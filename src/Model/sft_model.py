import logging
from typing import *
from typing import Any, Dict

import torch.nn as nn

from src.Dataset.data_util import batch_padding
from src.Model.model_utils import generate, get_llm_backbone, prepare_lora
from src.Train.sft_loss_func import get_loss_func
from src.Train.metric import Perplexity

logger = logging.getLogger(__name__)

class LLM(nn.Module):
    def __init__(self, args: Any):
        super(LLM, self).__init__()
        self.args = args
        self.backbone = get_llm_backbone(self.args)
        if args.training_args.lora:
            self.backbone = prepare_lora(args, self.backbone)
        self.loss_fn = get_loss_func(self.args)

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
        # Disable cache if gradient checkpointing is enabled
        if self.args.training_args.gradient_checkpointing:
            self.backbone.config.use_cache = False

        outputs: Dict = {}
        mask_key = "attention_mask"
        pad_keys = [
            "input_ids",
            "attention_mask",
            "special_tokens_mask",
            "labels",
        ]
        if padding:
            batch = batch_padding(
                self.args,
                batch,
                self.training,
                mask_key=mask_key,
                pad_keys=pad_keys,
                padding_side=self.args.tokenizer.padding_side,
            )

        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            # "position_ids": self.get_position_ids(batch["attention_mask"]),
        }

        try:
            output = self.backbone(**kwargs)
        except TypeError:
            # some models do not have position_ids
            del kwargs["position_ids"]
            output = self.backbone(**kwargs)
            output = self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        if "labels" in batch:
            loss = self.loss_fn(output.logits, batch["labels"])
            outputs["loss"] = loss

        if not self.training and self.args.infer_args.metric == "Perplexity":
            outputs["perplexity"] = self.perplexity(output.logits, batch["labels"])

        # Re-enable cache if gradient checkpointing was enabled
        if self.args.training_args.gradient_checkpointing:
            self.backbone.config.use_cache = True

        return outputs