import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Enviroment.env_utils import sync_across_processes
from Evaluation.eval_utils import eval_infer_result, save_predictions
from Evaluation.infer import LLM_infer

logger = logging.getLogger(__name__)


def LLM_eval(
    args: Any,
    model: torch.nn.Module,
    valid_dataloader: DataLoader,
    valid_data: pd.DataFrame,
    metric_func: callable,
    mode: str = "validation",
) -> Tuple[float, float]:
    """
    Run the evaluation loop for the LLM model.

    This function performs inference on the validation data, calculates metrics,
    and optionally saves predictions.

    Args:
        args (Any): Configuration object containing evaluation settings.
        model (torch.nn.Module): The trained model to evaluate.
        valid_dataloader (DataLoader): DataLoader for validation data.
        valid_data (pd.DataFrame): Validation data.
        metric_func (callable): Function to calculate evaluation metrics.
        mode (str, optional): Evaluation mode. Defaults to "validation".

    Returns:
        Tuple[float, float]: A tuple containing:
            - valid_loss (float): The average validation loss.
            - valid_metric (float): The average validation metric score.
    """
    logger.info(f"")
    logger.info(f"🚀 Starting validation")

    with torch.no_grad():
        model.eval()
        infer_result: Dict[str, Any] = LLM_infer(args, model, valid_dataloader, mode)
        model.train(model.training)

    if args.env_args._distributed:
        for key, value in infer_result.items():
            infer_result[key] = sync_across_processes(
                value, args.env_args._world_size, group=args.env_args._cpu_comm
            )

    if args.env_args._local_rank != 0:
        if args.env_args._distributed:
            torch.distributed.barrier()
        return 0, 0

    infer_result = eval_infer_result(args, valid_data, infer_result, metric_func)

    valid_loss = np.mean(
        infer_result.get("loss", torch.tensor(0)).float().cpu().numpy()
    )
    valid_metric = np.mean(infer_result["metrics"])

    logger.info(
        f"🔍 {mode.capitalize()} | {args.infer_args.metric}: {valid_metric:.5f} | "
        f"Step: {args.env_args._curr_step}"
    )

    save_predictions(args, infer_result, valid_data, mode)

    if args.env_args._distributed:
        torch.distributed.barrier()

    return valid_loss, valid_metric
