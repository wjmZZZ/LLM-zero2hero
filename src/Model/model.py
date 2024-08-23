import gc
import logging
import re
from typing import *
from typing import Any, Dict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.pytorch_utils import Conv1D as Conv1DTransformer

from Dataset.data_util import batch_padding
from Model.model_utils import generate, set_generation_config
from Train.loss_func import get_loss_func
from Train.metric import Perplexity

logger = logging.getLogger(__name__)


def get_llm_backbone(args: Any) -> nn.Module:
    """
    Get the LLM backbone model instance.

    Args:
        args: Configuration object containing all hyperparameters.

    Returns:
        nn.Module: LLM backbone model instance.
    """
    config = AutoConfig.from_pretrained(
        args.model_args.llm_backbone,
        trust_remote_code=args.model_args.trust_remote_code,
        use_fast=args.model_args.use_fast_tokenizer,
    )
    config = update_backbone_config(config, args)

    kwargs = {}
    quantization_config = None

    if args.model_args.backbone_dtype in ["int8", "int4"] and len(args.env_args.gpus):
        kwargs["device_map"] = {"": args.env_args._device}
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=0.0,
        )
        args.model_args.use_pretrained_model = True
        kwargs["torch_dtype"] = torch.float16
    elif args.model_args.backbone_dtype == "int4" and len(args.env_args.gpus):
        kwargs["device_map"] = {"": args.env_args._device}
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        args.model_args.use_pretrained_model = True
        kwargs["torch_dtype"] = torch.float16
    elif len(args.env_args.gpus) == 0 and args.model_args.backbone_dtype in [
        "int4",
        "int8",
    ]:
        logger.warning("Quantization not supported on CPU. Using float32.")
        args.model_args.backbone_dtype = "float32"
    else:
        kwargs["torch_dtype"] = getattr(torch, args.model_args.backbone_dtype)

    logger.info(
        f"🛠️ Using {args.model_args.backbone_dtype} as the data type for the backbone model"
    )

    kwargs["trust_remote_code"] = args.model_args.trust_remote_code

    # If Flash Attention 2 is configured to be used, attempt to import and set related parameters
    if args.training_args.use_flash_attention_2:
        try:
            import flash_attn  # noqa: F401

            # see https://github.com/fxmarty/transformers/blob/3f06a3a0aec8cc1ec3ad6bf66ebe277392c5ab37/src/transformers/configuration_utils.py#L380
            config._attn_implementation_internal = "flash_attention_2"
            logger.info("⚡ Using Flash Attention 2.")
        except ImportError:
            logger.warning(
                "Flash Attention 2.0 is not available. "
                "Please consider to run 'make setup' to install it."
            )

    if args.model_args.use_pretrained_model:
        logger.info(
            f">>> 📥 Loading {args.model_args.llm_backbone}. This may take a while."
        )

        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_args.llm_backbone,
            config=config,
            quantization_config=quantization_config,
            use_flash_attention_2=args.training_args.use_flash_attention_2,
            **kwargs,
        )
    else:
        kwargs.pop("token", None)
        backbone = AutoModelForCausalLM.from_config(config, **kwargs)

    if len(args.tokenizer) > config.vocab_size:
        backbone.resize_token_embeddings(len(args.tokenizer))

    backbone.model_parallel = False

    # ==================================================================
    # Configure LoRA
    # ==================================================================
    # If LoRA is used, enable gradient checkpointing and freeze base model layers
    if args.training_args.lora:
        # If used, gradient checkpointing will be enabled below
        loaded_in_kbit = getattr(backbone, "is_loaded_in_8bit", False) or getattr(
            backbone, "is_loaded_in_4bit", False
        )

        for name, param in backbone.named_parameters():
            param.requires_grad = False

        # Convert all non-INT8 parameters to fp32
        if loaded_in_kbit:
            for param in backbone.parameters():
                if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                    param.data = param.data.to(torch.float32)
    else:
        if args.model_args.backbone_dtype != "float32":
            if args.env_args.mixed_precision:
                logger.info(
                    "🚫 Disabling mixed precision as the data type is not set to float32."
                )
                args.env_args.mixed_precision = False
            if args.model_args.backbone_dtype != "bfloat16":
                logger.warning(
                    "Pure float16 or int8 training may lead to instability unless adapters are used."
                )

        for name, param in backbone.named_parameters():
            if any(
                freeze_layer in name
                for freeze_layer in args.training_args.freeze_layers
            ):
                logger.info(f"❄️ Freezing layer: {name}")
                param.requires_grad = False

    if args.training_args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()

    # Ensure special token ids in generation config match those in the tokenizer
    if backbone.generation_config.eos_token_id != config.eos_token_id:
        logger.warning(
            "EOS token id in generation config does not match that in the tokenizer. "
            "Using the tokenizer's id."
        )
        backbone.generation_config.eos_token_id = config.eos_token_id

    if backbone.generation_config.pad_token_id != config.pad_token_id:
        logger.warning(
            "PAD token id in generation config does not match that in the tokenizer. "
            "Using the tokenizer's id."
        )
        backbone.generation_config.pad_token_id = config.pad_token_id

    # No warning needed for bos_token_id as it is not used
    if backbone.generation_config.bos_token_id != config.bos_token_id:
        backbone.generation_config.bos_token_id = config.bos_token_id

    # Generation configuration
    backbone = set_generation_config(backbone, args)

    return backbone


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


def update_backbone_config(config: Any, args: Any):
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = args.model_args.intermediate_dropout
    if hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = args.model_args.intermediate_dropout
    # Issue a warning if the model configuration does not have dropout attributes and the intermediate dropout value is greater than 0
    if (
        not hasattr(config, "hidden_dropout_prob")
        and not hasattr(config, "attention_probs_dropout_prob")
        and args.model_args.intermediate_dropout > 0
    ):
        logger.warning(
            "The model configuration does not have dropout attributes. "
            f"Ignoring intermediate_dropout = {args.model_args.intermediate_dropout}."
        )
        args.model_args.intermediate_dropout = 0

    tokenizer = args.tokenizer

    if config.eos_token_id != tokenizer.eos_token_id:
        logger.warning(
            "EOS token id in the configuration does not match that in the tokenizer. "
            "Using the tokenizer's id."
        )
        config.eos_token_id = tokenizer.eos_token_id

    if config.pad_token_id != tokenizer.pad_token_id:
        logger.warning(
            "PAD token id in the configuration does not match that in the tokenizer. "
            "Using the tokenizer's id."
        )
        config.pad_token_id = tokenizer.pad_token_id
    # No warning needed for bos_token_id as it is not used
    if config.bos_token_id != tokenizer.bos_token_id:
        config.bos_token_id = tokenizer.bos_token_id

    # If the configuration contains specific training parameters, set the pretraining_tp attribute in the model configuration for tensor parallelism
    if hasattr(config, "pretraining_tp") and args.training_args.lora:
        logger.info("🔧 Setting pretraining_tp in the model configuration to 1.")
        config.pretraining_tp = 1

    return config


def prepare_lora(args, backbone):
    # Determine target modules for LoRA
    target_modules = (
        [
            lora_target_module.strip()
            for lora_target_module in args.training_args.lora_target_modules.strip().split(  # noqa: E501
                ","
            )
        ]
        if args.training_args.lora_target_modules
        else None
    )

    # Automatically determine target modules if not specified
    if target_modules is None:
        target_modules = []
        for name, module in backbone.named_modules():
            if (
                isinstance(
                    module, (torch.nn.Linear, torch.nn.Conv1d, Conv1DTransformer)
                )
                and "head" not in name
            ):
                name = name.split(".")[-1]
                if name not in target_modules:
                    target_modules.append(name)

    logger.info(f"🎯 LoRA module names: {target_modules}")

    # Set LoRA configuration
    lora_config = LoraConfig(
        use_dora=args.training_args.use_dora,
        r=args.training_args.lora_r,
        lora_alpha=args.training_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.training_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    backbone = get_peft_model(backbone, lora_config)
    for name, param in backbone.named_parameters():
        # unfreeze base model's layers
        if any(
            unfreeze_layer in name
            for unfreeze_layer in args.training_args.lora_unfreeze_layers
        ):
            logger.info(f"Unfreezing layer: {name}")
            param.requires_grad = True

    trainable_params, all_param = backbone.get_nb_trainable_parameters()
    logger.info(f"📊 Trainable parameters count: {trainable_params}")
    logger.info(f"📊 Total parameters count: {all_param}")
    logger.info(f"📊 Trainable %: {100 * trainable_params / all_param:.4f}%")

    return backbone


def load_checkpoint(
    args: Any, model: torch.nn.Module, strict: bool = True, weights_path: str = None
):
    """Load checkpoint

    Parameters:
        args: Configuration file
        model: Model to load weights into
        strict: Whether to strictly match weights
        weights_path: Custom path to weights. If None, use args.model_args.pretrained_weights
    Returns:
        epoch: Current training epoch
    """

    if weights_path is None:
        weights_path = args.model_argsdel_args.pretrained_weights

    model_weights = torch.load(weights_path, map_location="cpu")["model"]

    model = load_model_weights(model, model_weights, strict, args)

    del model_weights
    gc.collect()

    if args.env_args._local_rank == 0:
        logger.info(f"Loaded weights from: {weights_path}")


def load_model_weights(
    model: torch.nn.Module, model_weights: Dict, strict: bool, args: Any
):
    # load_model_weights function: Load model weights.
    orig_num_items = len(model_weights)
    model_state_dict = model.state_dict()

    # Need to load models trained in int4/int8 to other data types
    model_weights = {
        k: (
            v
            if not (
                args.model_args.backbone_dtype not in ("int4", "int8")
                and (v.dtype is torch.int8 or v.dtype is torch.uint8)
            )
            else model_state_dict[k]
        )
        for k, v in model_weights.items()
        if not (
            ("SCB" in k or "weight_format" in k)
            and args.model_args.backbone_dtype not in ("int4", "int8")
        )
    }

    # If the number of weights does not match, need to ignore int4/int8 weights, thus disabling strict loading
    if len(model_weights) != orig_num_items:
        strict = False

    model_weights = {re.sub(r"^module\.", "", k): v for k, v in model_weights.items()}
    model_weights = {k.replace("_orig_mod.", ""): v for k, v in model_weights.items()}

    # Manually fix int8 weights
    if args.model_args.backbone_dtype == "int8":
        model_weights = {
            k: v.to(args.env_args._device) if "weight_format" not in k else v
            for k, v in model_weights.items()
        }

    try:
        model.load_state_dict(OrderedDict(model_weights), strict=True)
    except Exception as e:
        if strict:
            raise e
        else:
            if args.env_args._local_rank == 0:
                logger.warning(
                    "Only partially loaded pretrained weights. "
                    "Some layers could not be initialized with pretrained weights: "
                    f"{e}"
                )

            for layer_name in re.findall("size mismatch for (.*?):", str(e)):
                model_weights.pop(layer_name, None)
            model.load_state_dict(OrderedDict(model_weights), strict=False)
    return model
