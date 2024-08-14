import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from Dataset.data_util import nested_dicts_to_dataframe

logger = logging.getLogger(__name__)


def clean_output(infer_result: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Clean the predicted text in the inference result.

    Args:
        infer_result (Dict[str, Any]): The inference result containing predicted text.
        args (Any): Arguments containing tokenizer information.

    Returns:
        Dict[str, Any]: The cleaned inference result.
    """
    infer_result["predicted_text"] = [
        text.strip()[: text.find(args.tokenizer.eos_token)].strip()
        if args.tokenizer.eos_token in text
        else text.strip()
        for text in infer_result["predicted_text"]
    ]
    return infer_result


def eval_infer_result(
    args: Any,
    valid_data: List[Dict[str, List[str]]],
    infer_result: Dict[str, Any],
    metric_func: callable,
) -> Dict[str, Any]:
    """
    Evaluate the inference result using the specified metric function.

    Args:
        args (Any): Arguments containing configuration information.
        valid_data (List[Dict[str, List[str]]]): Validation data.
        infer_result (Dict[str, Any]): Inference result to be evaluated.
        metric_func (callable): Function to calculate the evaluation metric.

    Returns:
        Dict[str, Any]: The evaluated inference result with metrics.
    """
    # Drop any extra observations
    for k, v in infer_result.items():
        infer_result[k] = v[: len(valid_data)] 
    if args.infer_args.metric != "Perplexity":
        infer_result = clean_output(infer_result, args)

    all_answers = [answer for item in valid_data for answer in item["responses"]]
    infer_result["target_text"] = all_answers

    metrics = metric_func(args, infer_result, valid_data)
    infer_result["metrics"] = metrics

    return infer_result


def save_predictions(
    args: Any, infer_result: Dict[str, Any], valid_data: pd.DataFrame, mode: str
) -> None:
    """
    Save the predictions to a CSV file.

    Args:
        args (Any): Configuration object.
        infer_result (Dict[str, Any]): Inference result containing predictions.
        valid_data (pd.DataFrame): Validation data.
        mode (str): The mode of evaluation (e.g., 'validation', 'test').
    """
    infer_result, valid_data = format_output(args, valid_data, infer_result)

    valid_res_path = os.path.join(args.exp_args.output_dir, "valid_result")
    # os.makedirs(valid_res_path, exist_ok=True)

    # if args.infer_args.metric == "BLEU":
    os.makedirs(valid_res_path, exist_ok=True)
    csv_preds_name = os.path.join(
        valid_res_path, f"{mode}_predictions_step{args.env_args._curr_step}.csv"
    )
    valid_data.to_csv(csv_preds_name, index=False)


def get_end_conversation_ids(conversations: List[Dict[str, List[Any]]]) -> List[int]:
    """
    Get the indices of the last element in each conversation.

    Args:
        conversations (List[Dict[str, List[Any]]]): List of conversations.

    Returns:
        List[int]: Indices of the last element in each conversation.
    """
    indices = []
    current_index = -1

    for item in conversations:
        num_elements = len(next(iter(item.values())))
        final_index = current_index + num_elements
        indices.append(final_index)
        current_index = final_index

    return indices


def format_output(
    args: Any, valid_data: pd.DataFrame, infer_result: Dict[str, Any]
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Format the output for saving predictions.

    Args:
        args (Any): Configuration object.
        valid_data (pd.DataFrame): Validation data.
        infer_result (Dict[str, Any]): Inference result.

    Returns:
        Tuple[Dict[str, Any], pd.DataFrame]: Formatted inference result and validation data.
    """
    infer_result = {
        key: value
        for key, value in infer_result.items()
        if key not in ["loss", "target", "losses"]
    }
    infer_result.pop("target_text", None)

    end_conversation_ids = get_end_conversation_ids(valid_data)
    valid_data = nested_dicts_to_dataframe(valid_data, args)

    if "predicted_text" in infer_result:
        infer_result["predicted_text"] = np.array(infer_result["predicted_text"])

    if "logits" in infer_result:
        infer_result["logits"] = np.array(infer_result["logits"].float())
        
    if "perplexity" in infer_result:
        infer_result["perplexity"] = np.array(infer_result["perplexity"].float())
        infer_result["perplexity"] = np.mean(infer_result["perplexity"], axis=1)

    prompt_columns = args.data_args.prompt_column
    if isinstance(prompt_columns, tuple):
        for col in prompt_columns:
            infer_result[col] = valid_data.loc[end_conversation_ids, col].values
    else:
        infer_result[prompt_columns] = valid_data.loc[
            end_conversation_ids, prompt_columns
        ].values

    if "predicted_text" in infer_result:
        pred_column = f"pred_{args.data_args.answer_column}"
        valid_data[pred_column] = "NO ANSWER GENERATED. ONLY LAST ANSWER OF A CONVERSATION IS PREDICTED."
        valid_data.loc[end_conversation_ids, pred_column] = infer_result["predicted_text"]
        
    if "perplexity" in infer_result:
        valid_data["mean_perplexity"] = -1
        valid_data.loc[end_conversation_ids, "mean_perplexity"] = infer_result["perplexity"]

    return infer_result, valid_data
