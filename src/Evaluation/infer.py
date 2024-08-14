import logging
from typing import Dict

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from Evaluation.infer_utils import batch_decode, cat_batches, contains_nan
from Model.model_utils import unwrap_model
from Others.exceptions import DataException, ModelException
from Train.train_utils import get_torch_dtype
from Utils.utils import TqdmTologger

logger = logging.getLogger(__name__)


def LLM_infer(
    args,
    model: torch.nn.Module,
    valid_dataloader,
    mode: str,
) -> Dict[str, list]:
    """Runs inference

    Args:
        args: config
        model: model
        valid_dataloader: custom valid_dataloader
        mode: mode for inference

    Returns:
        Dictionary with output

    """
    progress_bar = tqdm(
        total=len(valid_dataloader),
        disable=args.env_args._local_rank != 0,
        file=TqdmTologger(logger),
        ascii=True,
        desc=f"{mode} progress",
        mininterval=0,
    )
    final_output = dict()

    log_update_steps = max(len(valid_dataloader) // args.training_args.log_nums, 1)

    iter_dataloader = iter(valid_dataloader)
    for step in range(len(valid_dataloader)):
        try:
            data = next(iter_dataloader)
        except Exception:
            raise DataException("Dataloader reading error. Skipping inference.")

        if args.infer_args.batch_size_inference != 0:
            val_batch_size = args.infer_args.batch_size_inference
        else:
            val_batch_size = args.training_args.batch_size

        args.env_args._curr_val_step += val_batch_size * args.env_args._world_size

        batch = {key: value.to(args.env_args._device) for key, value in data.items()}

        if args.env_args.use_deepspeed:
            if args.infer_args.metric != "Perplexity":
                output = {}
                output["predicted_answer_ids"] = (
                    model.generate(batch, args).detach().cpu()
                )
            else:
                output = model.forward(batch)
        else:
            with autocast(
                enabled=args.env_args.mixed_precision,
                dtype=get_torch_dtype(args.env_args.mixed_precision_dtype),
            ):
                if args.infer_args.metric != "Perplexity":
                    output = {}
                    output["predicted_answer_ids"] = (
                        unwrap_model(model).generate(batch, args).detach().cpu()
                    )
                else:
                    output = model.forward(batch)

        if contains_nan(output) and args.env_args.mixed_precision:
            raise ModelException(
                "NaN caught during mixed precision inference. "
                "Please disable mixed precision inference. "
                "Alternatively, reducing learning rate or "
                "gradient clipping may help to stabilize training."
            )

        output = batch_decode(args, output=output)

        if "predicted_answer_ids" in output.keys():
            del output["predicted_answer_ids"]

        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu()

            if key not in final_output:
                final_output[key] = [val]
            else:
                final_output[key] += [val]

        if (step + 1) % log_update_steps == 0 or step == len(valid_dataloader) - 1:
            progress_bar.set_description(f"{mode} progress", refresh=False)
            if (step + 1) % log_update_steps == 0:
                progress_bar.update(log_update_steps)
            else:
                progress_bar.update(len(valid_dataloader) % log_update_steps)

        if args.env_args._distributed:
            torch.distributed.barrier()

    progress_bar.close()
    del progress_bar

    final_output = cat_batches(final_output)

    return final_output
