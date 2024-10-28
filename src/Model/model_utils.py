import logging
import os
import shutil
from typing import Any, Dict

import torch
import torch.nn as nn
import transformers
from deepspeed.utils.zero_to_fp32 import \
    get_fp32_state_dict_from_zero_checkpoint
from transformers import (GenerationMixin, StoppingCriteria,
                          StoppingCriteriaList)
from transformers.utils import logging as transformers_logging
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.pytorch_utils import Conv1D as Conv1DTransformer
from peft import LoraConfig, get_peft_model

from src.Dataset.data_util import batch_padding

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


class TokenStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria based on tokens.
    Will stop generation when each generated sample contains at least one of the
    stop_word_ids.
    """

    def __init__(self, stop_word_ids, prompt_input_ids_len):
        super().__init__()
        self.prompt_input_ids_len = prompt_input_ids_len
        if stop_word_ids is None:
            stop_word_ids = []
        self.stop_word_ids = stop_word_ids

    def should_stop(self, generated_ids: torch.Tensor, stop_word_id: torch.Tensor):
        if len(stop_word_id.shape) == 0:
            return (
                torch.mean(((generated_ids == stop_word_id).sum(1) > 0).float()) == 1
            ).item()
        else:
            return (
                self.get_num_vector_found_in_matrix_rows(stop_word_id, generated_ids)
                == generated_ids.shape[0]
            )

    @staticmethod
    def get_num_vector_found_in_matrix_rows(vector, matrix):
        """
        Count the number of times a vector is found in a matrix row.
        If the vector is found in a row, the search stops and the next row is searched.
        """
        assert len(vector.shape) == 1
        assert len(matrix.shape) == 2

        found = 0
        for row in matrix:
            # stride through the vector
            for i in range(len(row) - len(vector) + 1):
                # check if the vector contains the tensor
                if torch.all(row[i : i + len(vector)] == vector):
                    found += 1
                    break

        return found

    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor, **kwargs):
        generated_ids: torch.Tensor = input_ids[:, self.prompt_input_ids_len :]
        for stop_word_id in self.stop_word_ids:
            if self.should_stop(generated_ids, stop_word_id.to(generated_ids.device)):
                if generated_ids.shape[1] == 1:
                    logger.warning(
                        f"⚠️ Stopping criteria triggered for {stop_word_id} at first "
                        "generated token."
                    )
                return True
        return False


class EnvVariableStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria based on env variable.
    Useful to force stopping within the app.
    """

    stop_streaming_env: str = "STOP_STREAMING"

    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor, **kwargs):
        should_stop = self.stop_streaming_env in os.environ
        if should_stop:
            logger.info("🛑 Received signal to stop generating")
        return should_stop


def contains_nan(output: Dict):
    return (
        sum(
            [
                1
                for key, val in output.items()
                if isinstance(val, torch.Tensor)
                and torch.isnan(val.detach().cpu()).sum() > 0
            ]
        )
        > 0
    )


def unwrap_model(model: torch.nn.Module):
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

    while isinstance(model, options):
        model = model.module

    return model


def save_checkpoint(model: torch.nn.Module, path: str, args) -> None:
    """Saves a model checkpoint if the path is provided.

    Args:
        model: model to save
        path: path to save the checkpoint to
    """

    if not path:
        raise ValueError(f"Path must be provided. Received {path}.")

    if not os.path.exists(path):
        os.makedirs(path)

    if args.env_args.use_deepspeed:
        # gather model params from all ranks when using Deepspeed
        status = model.save_16bit_model(path, "checkpoint.pth")
        if status:
            if args.env_args._local_rank == 0:
                checkpoint = {
                    "model": torch.load(
                        os.path.join(path, "checkpoint.pth"), map_location="cpu"
                    )
                }
        else:
            logger.warning(
                "⚠️ deepspeed.save_16bit_model didn't save the model, since"
                " stage3_gather_16bit_weights_on_model_save=False."
                " Saving the full checkpoint instead"
            )
            model.save_checkpoint(os.path.join(path, "ds_checkpoint"))
            if args.env_args._local_rank == 0:
                # load to cpu
                state_dict = get_fp32_state_dict_from_zero_checkpoint(
                    os.path.join(path, "ds_checkpoint")
                )
                # save as normal checkpoint that can be loaded by `load_state_dict`
                checkpoint = {"model": state_dict}
                torch.save(checkpoint, os.path.join(path, "checkpoint.pth"))
                shutil.rmtree(os.path.join(path, "ds_checkpoint"))

    else:
        if args.env_args._local_rank == 0:
            model = unwrap_model(model)
            checkpoint = {"model": model.state_dict()}
            torch.save(checkpoint, os.path.join(path, "checkpoint.pth"))
            if (
                args.training_args.lora
                and len(args.training_args.lora_unfreeze_layers) == 0
            ):
                model.backbone.save_pretrained(os.path.join(path, "adapter_model"))


def generate(backbone, batch, args, streamer, remove_prompt=True):
    mask_key = "prompt_attention_mask"
    pad_keys = [
        "prompt_input_ids",
        "prompt_attention_mask",
    ]
    batch = batch_padding(
        args,
        batch,
        training=False,
        mask_key=mask_key,
        pad_keys=pad_keys,
    )
    input_ids = batch["prompt_input_ids"]
    attention_mask = batch["prompt_attention_mask"]
    # Adding GenerationMixin type annotation for faster lookup
    generation_function: GenerationMixin.generate = backbone.generate
    verbosity = transformers_logging.get_verbosity()
    stopping_criteria = StoppingCriteriaList(
        [
            TokenStoppingCriteria(
                stop_word_ids=[
                    torch.tensor(args.tokenizer.eos_token_id)
                ],  # _stop_words_ids,
                prompt_input_ids_len=input_ids.shape[1],
            ),
            EnvVariableStoppingCriteria(),
        ]
    )
    # force to use cache and disable gradient checkpointing if enabled
    backbone.config.use_cache = True
    if args.training_args.gradient_checkpointing:
        backbone.gradient_checkpointing_disable()
    transformers_logging.set_verbosity_error()
    output = generation_function(
        inputs=input_ids,
        attention_mask=attention_mask,
        generation_config=backbone.generation_config,
        stopping_criteria=stopping_criteria,
        renormalize_logits=True,
        return_dict_in_generate=False,
        use_cache=True,
        streamer=streamer,
    )
    transformers_logging.set_verbosity(verbosity)
    # enable checkpointing again
    if args.training_args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if remove_prompt:
        output = output[:, input_ids.shape[1] :]
    return output


def set_generation_config(backbone: torch.nn.Module, args: Any):
    backbone.generation_config.min_new_tokens = args.infer_args.min_length_inference
    backbone.generation_config.max_new_tokens = args.infer_args.max_length_inference
    backbone.generation_config.max_time = (
        args.infer_args.max_time if args.infer_args.max_time > 0 else None
    )
    backbone.generation_config.do_sample = args.infer_args.do_sample
    backbone.generation_config.num_beams = args.infer_args.num_beams
    backbone.generation_config.repetition_penalty = args.infer_args.repetition_penalty
    if args.infer_args.do_sample:
        backbone.generation_config.temperature = args.infer_args.temperature
        backbone.generation_config.top_k = args.infer_args.top_k
        backbone.generation_config.top_p = args.infer_args.top_p
    else:
        backbone.generation_config.temperature = None
        backbone.generation_config.top_k = None
        backbone.generation_config.top_p = None

    backbone.generation_config.transformers_version = transformers.__version__
    return backbone
