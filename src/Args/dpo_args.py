import json
import os
from dataclasses import dataclass, is_dataclass
from datetime import datetime
from typing import Any, Dict
import torch

from src.Utils.utils import get_logger
from src.Args.sft_args import *

# ===========================================================================
# Argument Configuration Classes
# ===========================================================================
@dataclass
class DPODatasetArguments(DatasetArguments):
    """
    DPO dataset configuration
    """
    answer_column: str = "chosen_response"
    rejected_answer_column: str = "rejected_response"



@dataclass
class DPOTrainingArguments(TrainingArguments):
    """
    DPO training configuration
    """
    learning_rate: float = 1e-4
    beta: float = 0.2
    simpo_gamma: float = 1.0
    gradient_clip: float = 10.0
    loss_function: str = "DPOLoss"
    optimizer: str = "AdamW"
    # Needs to be enabled as we need logits from original model, see forward pass
    lora: bool = True


@dataclass
class DPOArguments:
    exp_args: ExperimentArguments
    data_args: DPODatasetArguments
    model_args: ModelArguments
    training_args: DPOTrainingArguments
    infer_args: InferenceArguments
    env_args: EnvironmentArguments

    debug: bool = False

    def __post_init__(self):
        table = self.table_beauty()
        logger = get_logger(self)
        logger.info("\n" + table)

        if (
            self.exp_args.use_wandb
            and self.exp_args.wandb_name == self.exp_args.experiment_name
        ):
            logger.warning(
                "wandb_name is set to experiment_name. It is recommended to set a specific wandb_name for better management and differentiation of experiments."
            )

        if self.debug:
            logger.debug(
                "🔥🔥 Debug mode is enabled. Detailed debug information will be logged. 🔥🔥"
            )

    def table_beauty(self):
        """
        Generate a formatted table of experiment configuration.

        Returns:
            str: A string representation of the formatted table.
        """
        from tabulate import tabulate

        # Create a dictionary containing the information you want to display in the log
        log_info = {
            "Experiment Name": self.exp_args.experiment_name,
            "Experiment Task": self.exp_args.task,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Base Model": self.model_args.llm_backbone,
            "Training Seed": self.env_args.seed,
            "Training Data": self.data_args.train_data_dir,
            "Validation Data": self.data_args.valid_data_dir,
            "Batch Size": str(self.training_args.batch_size),
            "Number of Epochs": str(self.training_args.num_train_epochs),
            "Max_seq_length": str(self.training_args.max_seq_length),
            "Optimizer": self.training_args.optimizer,
            "Learning Rate": str(self.training_args.learning_rate),
            "Description": (
                "This experiment aims to improve model performance and accuracy through effective model training."
                if self.exp_args.experiment_description == ""
                else self.exp_args.experiment_description
            ),
        }
        if self.exp_args.sub_experiment_name:
            log_info["Sub-Experiment Name"] = self.exp_args.sub_experiment_name

        if "Sub-Experiment Name" in log_info:
            items = list(log_info.items())
            main_exp_index = next(
                i for i, (k, v) in enumerate(items) if k == "Experiment Name"
            )
            items.insert(
                main_exp_index + 1,
                ("Sub-Experiment Name", log_info["Sub-Experiment Name"]),
            )
            log_info = dict(items)

        return tabulate(
            log_info.items(), headers=["Configuration", ""], tablefmt="pretty"
        )


def to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert a dataclass object or nested structure to a dictionary.

    Args:
        obj (Any): The object to convert.

    Returns:
        Dict[str, Any]: A dictionary representation of the object.
    """
    if is_dataclass(obj):
        result = {}
        for field in obj.__dataclass_fields__:
            value = getattr(obj, field)
            if not isinstance(value, (torch._C._distributed_c10d.ProcessGroup)):
                result[field] = to_dict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def save_args(args: DPOArguments, file_format: str = "json") -> None:
    """
    Save the experiment arguments to a file.

    Args:
        args (Arguments): The experiment arguments to save.
        file_format (str, optional): The format to save the file in. Defaults to "json".

    Returns:
        None
    """
    if args.exp_args.sub_experiment_name:
        file_name = f"{args.exp_args.experiment_name}_{args.exp_args.sub_experiment_name}_cfg.{file_format}"
    else:
        file_name = f"{args.exp_args.experiment_name}_cfg.{file_format}"
    file_path = os.path.join(args.exp_args.output_dir, file_name)
    serializable_args = to_dict(args)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_args, f, ensure_ascii=False, indent=4)
