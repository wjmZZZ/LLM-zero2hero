import logging
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch import nn

logger = logging.getLogger(__name__)


def sacrebleu_score(args: Any, results: Dict, valid_data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the BLEU score for predicted texts against target texts.

    Args:
        args (Any): Configuration object.
        results (Dict): Dictionary containing predicted and target texts.
        valid_data (pd.DataFrame): Validation data.

    Returns:
        np.ndarray: Array of BLEU scores for each text pair.
    """
    scores = []
    smooth = SmoothingFunction().method1

    for predicted_text, target_text in zip(
        results["predicted_text"], results["target_text"]
    ):
        if not target_text.strip():
            score = 0.0
        else:
            predicted_chars = list(predicted_text)
            target_chars = list(target_text)
            score = sentence_bleu(
                [target_chars], predicted_chars, smoothing_function=smooth
            )
        scores.append(score)

    return np.array(scores)


# TODO: Add OpenAI-style evaluation for SiliconCloud
# TODO: Add custom evaluations such as embedding similarity, text length, etc.


class Perplexity(nn.Module):
    """
    Perplexity calculation module.

    Args:
        args (Any): Configuration object.
        reduce (bool): Whether to reduce the perplexity to a single value.
    """

    def __init__(self, args: Any, reduce: bool = True):
        super().__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.reduce = reduce

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate perplexity for the given logits and labels.

        Args:
            logits (torch.Tensor): Predicted logits.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Calculated perplexity.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)
        perplexity = torch.exp(perplexity)
        if self.reduce:
            perplexity = torch.mean(perplexity)

        return perplexity


def perplexity(args: Any, results: Dict, val_df: pd.DataFrame) -> np.ndarray:
    """
    Extract perplexity from results.

    Args:
        args (Any): Configuration object.
        results (Dict): Dictionary containing perplexity results.
        val_df (pd.DataFrame): Validation dataframe.

    Returns:
        np.ndarray: Array of perplexity values.
    """
    return results["perplexity"].detach().float().cpu().numpy()


class Metrics:
    """
    Metrics factory class.
    """

    _metrics = {
        "Perplexity": (perplexity, "min", "mean"),
        "BLEU": (sacrebleu_score, "max", "mean"),
    }

    @classmethod
    def get_metric(cls, name: str) -> tuple:
        """
        Get metric function and associated information.

        Args:
            name (str): Name of the metric.

        Returns:
            tuple: (metric_function, optimization_direction, reduction_method)

        Raises:
            ValueError: If the metric name is not found.
        """
        metric = cls._metrics.get(name)
        if metric is None:
            raise ValueError(f"Metric '{name}' not found.")
        return metric


def get_metric(args: Any) -> Callable:
    """
    Prepare the metric function based on configuration.

    Args:
        args (Any): Configuration arguments.

    Returns:
        Callable: Metric function.
    """
    metric_func, metric_mode, metric_reduce = Metrics.get_metric(args.infer_args.metric)

    if metric_mode == "max":
        args.infer_args._best_valid_metric = -np.inf
        args.infer_args._objective_op = np.greater
    else:
        args.infer_args._best_valid_metric = np.inf
        args.infer_args._objective_op = np.less

    return metric_func
