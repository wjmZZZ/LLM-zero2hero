import logging
from textwrap import dedent
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import wandb
from src.Evaluation.eval import LLM_eval
from src.Model.model_utils import save_checkpoint
from src.Others.exceptions import TrainingException
from src.Train.train_utils import get_torch_dtype
from src.Utils.utils import TqdmTologger, seed_everything

logger = logging.getLogger(__name__)


def LLM_train(
    args: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    valid_data: pd.DataFrame,
    metric_func: callable,
) -> Tuple[float, float]:
    """
    Run the training loop for a LLM.

    Args:
        args: Configuration object containing all training parameters.
        model: The LLM model to be trained.
        optimizer: The optimizer for model parameter updates.
        scheduler: Learning rate scheduler.
        train_dataloader: DataLoader for training data.
        valid_dataloader: DataLoader for validation data.
        valid_data: DataFrame containing validation dataset.
        metric_func: Function to calculate evaluation metrics.

    Returns:
        A tuple containing the final validation loss and metric.

    Raises:
        TrainingException: If NaN is encountered in loss during training.
    """
    scaler: GradScaler | None = None
    if args.env_args.mixed_precision:
        scaler = GradScaler(enabled=(args.env_args.mixed_precision_dtype == "float16"))

    optimizer.zero_grad(set_to_none=True)
    best_valid_metric = args.infer_args._best_valid_metric
    objective_op = args.infer_args._objective_op

    if args.training_args.evaluate_before_training:
        valid_loss, valid_metric = LLM_eval(
            args, model, valid_dataloader, valid_data, metric_func
        )

    for epoch in range(args.training_args.num_train_epochs):
        seed_everything(
            args.env_args.seed
            + epoch * args.env_args._world_size * args.env_args.number_of_workers
            + args.env_args._local_rank * args.env_args.number_of_workers
        )

        if (
            args.env_args._distributed
            and not args.env_args.use_deepspeed
            and hasattr(train_dataloader.sampler, "set_epoch")
        ):
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        losses = []

        log_update_steps = max(
            args.training_args._training_epoch_steps // args.training_args.log_nums, 1
        )
        evaluation_step = args.training_args._validation_steps
        logger.info(
            f"🚀 Train Epoch: {epoch + 1} / {args.training_args.num_train_epochs}"
        )

        progress_bar = tqdm(
            total=args.training_args._training_epoch_steps,
            disable=args.env_args._local_rank != 0,
            file=TqdmTologger(logger),
            ascii=True,
            desc="train loss",
            mininterval=0,
        )

        for step, data in enumerate(train_dataloader):
            args.env_args._curr_step += (
                args.training_args.batch_size * args.env_args._world_size
            )
            batch = {k: v.to(args.env_args._device) for k, v in data.items()}

            model.require_backward_grad_sync = (
                step % args.training_args.grad_accumulation == 0
            )

            with autocast(
                enabled=args.env_args.mixed_precision,
                dtype=get_torch_dtype(args.env_args.mixed_precision_dtype),
            ):
                output = model.forward(batch)

            loss = output["loss"]
            if ~np.isfinite(loss.item()) and (epoch > 0 or step > 20):
                raise TrainingException(
                    dedent(
                        """
                    NaN caught in loss during training. 
                    Please, reduce learning rate, change dtype, 
                    or disable mixed precision. Alternatively, 
                    gradient clipping may help to stabilize training.
                    """
                    )
                )
            losses.append(loss.item())

            if args.exp_args.use_wandb and args.env_args._local_rank == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/step": args.env_args._curr_step,
                    }
                )

            if args.training_args.grad_accumulation != 1:
                loss = loss / args.training_args.grad_accumulation

            if (
                args.env_args.mixed_precision
                and len(args.env_args.gpus)
                and not args.env_args.use_deepspeed
            ):
                scaler.scale(loss).backward()
                if step % args.training_args.grad_accumulation == 0:
                    if args.training_args.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.training_args.gradient_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                if args.env_args.use_deepspeed:
                    model.backward(loss)
                else:
                    loss.backward()

                if step % args.training_args.grad_accumulation == 0:
                    if args.training_args.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.training_args.gradient_clip
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if args.env_args._distributed:
                torch.cuda.synchronize(device=args.env_args._local_rank)

            if scheduler:
                scheduler.step()

                if (
                    (step + 1) % log_update_steps == 0
                    or step == args.training_args._training_epoch_steps - 1
                ):
                    progress_bar.set_description(
                        f"train loss: {np.mean(losses[-10:]):.2f}", refresh=False
                    )
                    if (step + 1) % log_update_steps == 0:
                        progress_bar.update(log_update_steps)
                    else:
                        progress_bar.update(
                            args.training_args._training_epoch_steps % log_update_steps
                        )

                del output

            if (step + 1) % evaluation_step == 0:
                if args.training_args.save_checkpoint == "last":
                    logger.info(
                        f"🏆 Saving last model checkpoint to {args.exp_args.output_dir}"
                    )
                    save_checkpoint(
                        model=model, path=args.exp_args.output_dir, args=args
                    )

                valid_loss, valid_metric = LLM_eval(
                    args, model, valid_dataloader, valid_data, metric_func
                )

                if args.exp_args.use_wandb and args.env_args._local_rank == 0:
                    wandb.log(
                        {
                            "valid/loss": valid_loss,
                            "valid/metric": valid_metric,
                            "valid/step": args.env_args._curr_step,
                        }
                    )

                if args.training_args.save_checkpoint == "best":
                    if objective_op(valid_metric, best_valid_metric):
                        logger.info(
                            f"🏆 [BEST MODEL] | "
                            f"✨ {args.infer_args.metric}: {valid_metric:.5f} | "
                            f"🚀 improvement:{best_valid_metric:.3f} -> {valid_metric:.3f} | "
                            f"📁 save to {args.exp_args.output_dir}"
                        )
                        logger.info(f"")
                        save_checkpoint(
                            model=model, path=args.exp_args.output_dir, args=args
                        )
                        best_valid_metric = valid_metric

                model.train()

        progress_bar.close()
        del progress_bar

        if args.env_args._distributed:
            torch.cuda.synchronize(device=args.env_args._local_rank)
            torch.distributed.barrier()

        logger.info(f"")
        logger.info(
            f"✅ Epoch {epoch + 1}/{args.training_args.num_train_epochs} completed. Current step: {args.env_args._curr_step}"
        )

    if args.env_args._distributed:
        torch.distributed.barrier()

    return valid_loss, valid_metric
