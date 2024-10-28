import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch import nn
from tqdm import tqdm

from src.Evaluation.AI_utils import AIEvaluator, get_ai_template
from src.Utils.utils import TqdmTologger

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


# DONE Add OpenAI-style evaluation for SiliconCloud
# TODO Add custom evaluations such as embedding similarity, text length, etc.
def AI_eval_score(
    args,
    results: Dict[str, List[str]],
    valid_df,
) -> Union[float, Tuple[np.ndarray, List[str]]]:
    """
    Calculate AI score for given results.

    Args:
        args: Configuration parameters.
        results: Dictionary containing predicted and target texts.
        valid_df: Validation data DataFrame.

    Returns:
        If raw_results is True, returns a tuple of (scores, explanations).
        Otherwise, returns the average score.
    """

    # Get the specified AI evaluation template
    eval_template = get_ai_template(args.infer_args.AI_eval_template_name)

    # Prepare evaluation data
    prompts = []
    for item in valid_df:
        if isinstance(item, dict) and args.data_args.prompt_column in item:
            prompts.extend([
                prompt[len(args.data_args.prompt_prefix):]
                .rstrip(args.data_args.prompt_suffix)
                .strip()
                for prompt in item[args.data_args.prompt_column]
            ])
        else:
            # If the structure of valid_df is not as expected, record the error and use an empty string
            logger.error(f"❌ Unexpected item structure in valid_df: {item}")
            prompts.append(" ")

    # Ensure all data lengths are consistent
    min_length = min(len(prompts), len(results["predicted_text"]), len(results["target_text"]))
    
    eval_data = pd.DataFrame(
        {
            "prompt_": prompts[:min_length],
            "predicted_text_": results["predicted_text"][:min_length],
            "target_text_": results["target_text"][:min_length],
        }
    )
    
    # Fill the evaluation template
    eval_data["formatted_evaluation"] = eval_data.apply(
        lambda row: eval_template.format(
            prompt=row["prompt_"],
            predicted_text=row["predicted_text_"],
            target_text=row["target_text_"],
        ),
        axis=1,
    )

    # Set up progress bar
    tqdm_logger = TqdmTologger(logger)
    ai_evaluator = AIEvaluator(args)
    model = args.infer_args.AI_eval_model
    # Perform parallel evaluation
    eval_results = Parallel(n_jobs=8, backend="threading")(
        delayed(ai_evaluator.evaluate_response)(formatted_evaluation, model)
        for formatted_evaluation in tqdm(
            eval_data["formatted_evaluation"].values,
            file=tqdm_logger,
            desc=f" AI evaluation {args.infer_args.AI_eval_model}",
            total=len(eval_data),
        )
    )

    # Unpack results
    scores, explanations = zip(*eval_results)

    return np.array(scores), list(explanations)


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


def perplexity(args: Any, results: Dict, valid_df: pd.DataFrame) -> np.ndarray:
    """
    Extract perplexity from results.

    Args:
        args (Any): Configuration object.
        results (Dict): Dictionary containing perplexity results.
        valid_df (pd.DataFrame): Validation dataframe.

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
        "AI": (AI_eval_score, "max", "mean"),
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
