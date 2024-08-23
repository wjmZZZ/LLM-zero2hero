import logging
import os
import shutil
from typing import Any, Dict

import torch
import transformers
from deepspeed.utils.zero_to_fp32 import \
    get_fp32_state_dict_from_zero_checkpoint
from transformers import (GenerationMixin, StoppingCriteria,
                          StoppingCriteriaList)
from transformers.utils import logging as transformers_logging

from Dataset.data_util import batch_padding

logger = logging.getLogger(__name__)


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
