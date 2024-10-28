import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from src.Dataset.data_util import batch_padding
from src.Model.model_utils import generate, get_llm_backbone, prepare_lora
from src.Train.dpo_loss_func import get_loss_func
from src.Train.sft_loss_func import SampleAveragedCrossEntropyLoss
from src.Train.metric import Perplexity

logger = logging.getLogger(__name__)


class DPO_LLM(nn.Module):
    def __init__(self, args: Any):
        super(DPO_LLM, self).__init__()
        self.args = args
        self.backbone = get_llm_backbone(self.args)
        if args.training_args.lora:
            self.backbone = prepare_lora(args, self.backbone)
        self.loss_fn = get_loss_func(self.args)

        if self.loss_fn.requires_reference_model:
            if self.args.training_args.lora and not self.args.training_args.lora_unfreeze_layers:
                self.reference_backbone = None
            else:
                logger.info("🔄 Duplicating backbone for reference model.")
                self.reference_backbone = get_llm_backbone(self.args)
                for _, param in self.reference_backbone.named_parameters():
                    param.requires_grad = False
                self.reference_backbone = self.reference_backbone.eval()
                
        if self.args.infer_args.metric == "Perplexity":
            self.perplexity = Perplexity(self.args, reduce=False)

    def init_deepspeed(self):
        self.backward = self.backbone.backward
        self.save_checkpoint = self.backbone.save_checkpoint
        self.save_16bit_model = self.backbone.save_16bit_model
        if self.args.training_args.lora:
            self.backbone.base_model.model.config = (
                self.backbone.base_model.model.module.config
            )
            self.backbone.base_model.model.generation_config = (
                self.backbone.base_model.model.module.generation_config
            )
        else:
            self.backbone.config = self.backbone.module.config
            self.backbone.generation_config = self.backbone.module.generation_config

    def generate(self, batch: Dict, args: Any, streamer=None):
        if args.env_args.use_deepspeed and args.training_args.lora:
            return generate(self.backbone.base_model.model, batch, args, streamer)
        else:
            return generate(self.backbone, batch, args, streamer)

    def get_position_ids(attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def forward(self, batch: Dict, padding: bool = True) -> Dict:
        """
        Forward pass of DPO model. 
        Computes logits for chosen and rejected answers with both 
        the policy model and the reference model.
        """
        # Disable cache if gradient checkpointing is enabled
        if self.args.training_args.gradient_checkpointing:
            self.backbone.config.use_cache = False

        outputs: Dict = {}
        logits_dict = {}
        labels_dict = {}

        # Iterate over chosen and rejected answers
        for answer in ["chosen", "rejected"]:
            mask_key = f"{answer}_attention_mask"
            pad_keys = [
                f"{answer}_input_ids",
                f"{answer}_attention_mask",
                f"{answer}_labels",
            ]
            
            # Apply padding if enabled
            if padding:
                batch = batch_padding(
                    self.args,
                    batch,
                    self.args.training_args,
                    mask_key=mask_key,
                    pad_keys=pad_keys,
                )

            # Compute logits for the policy model
            kwargs = {
                "input_ids": batch[f"{answer}_input_ids"],
                "attention_mask": batch[f"{answer}_attention_mask"],
            }
            logits = self.backbone(**kwargs).logits
            logits_dict[answer] = logits
            labels_dict[answer] = batch[f"{answer}_labels"]

            outputs[f"{answer}_logps"] = get_batch_logps(
                logits,
                labels_dict[answer],
                average_log_prob=self.loss_fn.loss_reduction,
            )

            # Compute logits for the reference model if required
            if self.loss_fn.requires_reference_model:
                with torch.no_grad():
                    if self.reference_backbone:
                        reference_logits = self.reference_backbone(                            
                            input_ids=batch[f"{answer}_input_ids"],
                            attention_mask=batch[f"{answer}_attention_mask"]
                        ).logits
                    else:
                        with self.backbone.disable_adapter():
                            reference_logits = self.backbone(
                                input_ids=batch[f"{answer}_input_ids"],
                                attention_mask=batch[f"{answer}_attention_mask"]
                            ).logits

                    outputs[f"{answer}_reference_logps"] = get_batch_logps(
                        reference_logits,
                        labels_dict[answer],
                        average_log_prob=self.loss_fn.loss_reduction,
                    )

        # Compute the loss
        if self.loss_fn.requires_reference_model:
            loss, chosen_rewards, rejected_rewards = self.loss_fn(
                policy_chosen_logps=outputs["chosen_logps"],
                policy_rejected_logps=outputs["rejected_logps"],
                reference_chosen_logps=outputs["chosen_reference_logps"],
                reference_rejected_logps=outputs["rejected_reference_logps"],
            )
        else:
            loss, chosen_rewards, rejected_rewards = self.loss_fn(
                policy_chosen_logps=outputs["chosen_logps"],
                policy_rejected_logps=outputs["rejected_logps"],
            )
        outputs["loss"] = loss

        # Additional logging
        outputs["additional_log_chosen_rewards"] = chosen_rewards.detach()
        outputs["additional_log_rejected_rewards"] = rejected_rewards.detach()
        outputs["additional_log_reward_margin"] = (chosen_rewards - rejected_rewards).detach()

        outputs["additional_log_chosen_cross_entropy_loss"] = (
            SampleAveragedCrossEntropyLoss(self.args)(logits_dict["chosen"], 
                                                      labels_dict["chosen"]).detach()
        )
        outputs["additional_log_rejected_cross_entropy_loss"] = (
            SampleAveragedCrossEntropyLoss(self.args)(logits_dict["rejected"], 
                                                      labels_dict["rejected"]).detach()
        )

        # Compute perplexity if required
        if not self.training and self.args.infer_args.metric == "Perplexity":
            outputs["perplexity"] = self.perplexity(logits_dict["chosen"], 
                                                    labels_dict["chosen"])
            outputs["additional_log_rejected_perplexity"] = self.perplexity(logits_dict["rejected"], 
                                                                            labels_dict["rejected"])

        # Re-enable cache if gradient checkpointing was enabled
        if self.args.training_args.gradient_checkpointing:
            self.backbone.config.use_cache = True

        return outputs


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """
    Based upon the official implementation of DPO:
    https://github.com/eric-mitchell/direct-preference-optimization

    Compute the log probabilities of the given labels under the given logits.
    Args:
        logits:
            Logits of the model (unnormalized).
            Shape: (batch_size, sequence_length, vocab_size)
        labels:
            Labels for which to compute the log probabilities.
            Label tokens with a value of -100 are ignored.
            Shape: (batch_size, sequence_length)
        average_log_prob:
            If True, return the average log probability per (non-masked) token.
            Otherwise, return the sum of the
            log probabilities of the (non-masked) tokens.
    Returns:
        A tensor of shape (batch_size,) containing the average/sum
        log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    # shift labels and logits to account for next token prediction
    # See also text_causal_language_modeling_losses.py
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens when loss_mask is applied
    # Needed to be able to apply torch.gather with index=labels.unsqueeze(2)
    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)