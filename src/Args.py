import json
import os
import sys
from dataclasses import dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from transformers import HfArgumentParser

from Others.exceptions import DataException
from Utils.utils import flatten_dict, get_logger


# ===================================================================================
# Get Arguments
# ===================================================================================
def get_args(args: Optional[Dict[str, Any]] = None):
    parser = HfArgumentParser(
        (
            ExperimentArguments,
            DatasetArguments,
            ModelArguments,
            TrainingArguments,
            InferenceArguments,
            EnvironmentArguments,
        )
    )

    if args is not None:
        (
            exp_args,
            data_args,
            model_args,
            training_args,
            infer_args,
            env_args,
        ) = parser.parse_dict(args)
    else:
        # Extract command line arguments
        remaining_args = []
        config_file = None

    for arg in sys.argv[1:]:
        if arg.endswith((".yaml", ".json")):
            config_file = arg
        else:
            remaining_args.append(arg)

    if config_file:
        config_path = Path(os.path.abspath(config_file))
        if config_file.endswith(".yaml"):
            nested_dict = yaml.safe_load(config_path.read_text())
        else:
            with config_path.open("r", encoding="utf-8") as file:
                nested_dict = json.load(file)
        flat_dict = flatten_dict(nested_dict)
        return parser.parse_dict(flat_dict)
    else:
        return parser.parse_args_into_dataclasses(remaining_args)


# ===========================================================================
# Argument Configuration Classes
# ===========================================================================
@dataclass
class ExperimentArguments:
    """
    Experiment configuration, including basic information and path settings for the current experiment
    """

    experiment_name: str = "experiment"
    sub_experiment_name: str = ""
    experiment_description: str = ""

    output_dir: str = "./outputs"
    log_file_name: str = "log.log"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        exp_output_dir = os.path.join(self.output_dir, self.experiment_name)
        os.makedirs(exp_output_dir, exist_ok=True)
        self.log_file_name = os.path.join(exp_output_dir, self.log_file_name)

        # If there is a sub-experiment
        if self.sub_experiment_name:
            sub_exp_dir = (
                self.output_dir
                + "/"
                + self.experiment_name
                + "/"
                + self.sub_experiment_name
            )
            os.makedirs(sub_exp_dir, exist_ok=True)
            self.log_file_name = os.path.join(sub_exp_dir, self.log_file_name)

        self.output_dir = exp_output_dir


@dataclass
class DatasetArguments:
    """
    Data configuration, including data-related content
    """

    train_data_dir: str = ""
    valid_data_dir: str = ""
    valid_strategy: str = "auto"
    valid_size: float = 0.01

    system_column: str = "None"
    prompt_column: str = "prompt"
    answer_column: str = "response"

    system_prefix: str = "<|system|>"
    system_defalut: str = "You are a helpful assistant."
    system_suffix: str = "<|system|>"
    prompt_prefix: str = "<|prompt|>"
    prompt_suffix: str = "<|prompt|>"
    response_prefix: str = "<|answer|>"
    response_suffix: str = "<|answer|>"

    mask_prompt_labels: bool = True

    def __post_init__(self):
        if self.valid_strategy in ("", None, "None"):
            raise DataException(
                "Validation strategy must be set, options are [custom, auto]"
            )


@dataclass
class ModelArguments:
    """
    Model configuration
    """

    pretrained: bool = True
    llm_backbone: str = ""
    use_pretrained_model: bool = True
    backbone_dtype: str = "float16"
    intermediate_dropout: float = 0
    pretrained_weights: str = ""

    use_fast_tokenizer: bool = True
    trust_remote_code: bool = True
    add_prefix_space: bool = False


@dataclass
class TrainingArguments:
    """
    Training configuration
    """

    num_train_epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 0.0001
    differential_learning_rate_layers: Tuple[str, ...] = ()
    differential_learning_rate: float = 0.00001
    max_seq_length: int = 512

    loss_function: str = "TokenAveragedCrossEntropy"
    optimizer: str = "AdamW"
    schedule: str = "Cosine"
    save_checkpoint: str = "last"

    freeze_layers: Tuple[str, ...] = ()
    use_flash_attention_2: bool = False
    drop_last_batch: bool = False

    warmup_epochs: float = 0.0
    log_nums: int = 20
    weight_decay: float = 0.0
    gradient_clip: float = 0.0
    grad_accumulation: int = 1
    gradient_checkpointing: bool = False

    lora: bool = False
    use_dora: bool = False
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = ""
    lora_unfreeze_layers: Tuple[str, ...] = ()

    num_validations_per_epoch: int = 1  # Number of validations per epoch
    evaluate_before_training: bool = False

    _training_epoch_steps: int = 0
    _validation_steps: int = -1  # Number of steps per validation

    def __post_init__(self):
        self.evaluate_before_training = (
            self.evaluate_before_training or self.num_train_epochs == 0
        )


@dataclass
class InferenceArguments:
    """
    Inference configuration
    """

    metric: str = "Perplexity"

    min_length_inference: int = 2
    max_length_inference: int = 256
    max_time: float = 0
    batch_size_inference: int = 0

    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 0.0
    repetition_penalty: float = 1.2
    stop_tokens: str = ""  # TODO
    top_k: int = 0
    top_p: float = 1.0

    _best_valid_metric: float = float("inf")
    _objective_op: Callable[[float, float], bool] = np.less


@dataclass
class EnvironmentArguments:
    use_deepspeed: bool = False
    deepspeed_method: str = "ZeRO2"
    seed: int = 42

    gpus: Tuple[str, ...] = tuple(str(x) for x in range(torch.cuda.device_count()))
    mixed_precision: bool = False
    mixed_precision_dtype: str = "bfloat16"

    compile_model: bool = False

    deepspeed_allgather_bucket_size: int = int(1e6)
    deepspeed_reduce_bucket_size: int = int(1e6)
    deepspeed_stage3_prefetch_bucket_size: int = int(1e6)
    deepspeed_stage3_param_persistence_threshold: int = int(1e6)

    find_unused_parameters: bool = False
    number_of_workers: int = 4

    # Private parameter configuration
    _distributed_inference: bool = False
    _distributed: bool = False
    _local_rank: int = 0
    _world_size: int = 1
    _curr_step: int = 0
    _curr_val_step: int = 0
    _rank: int = 0  # global rank
    _device: str = "cuda"
    _cpu_comm: Any = None


@dataclass
class Arguments:
    exp_args: ExperimentArguments
    data_args: DatasetArguments
    model_args: ModelArguments
    training_args: TrainingArguments
    infer_args: InferenceArguments
    env_args: EnvironmentArguments

    debug: bool = True

    def __post_init__(self):
        table = self.table_beauty()
        logger = get_logger(self.exp_args.log_file_name)
        logger.info("\n" + table)

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


def save_args(args: Arguments, file_format: str = "json") -> None:
    """
    Save the experiment arguments to a file.

    Args:
        args (Arguments): The experiment arguments to save.
        file_format (str, optional): The format to save the file in. Defaults to "json".

    Returns:
        None
    """
    file_path = os.path.join(
        args.exp_args.output_dir, f"{args.exp_args.experiment_name}_cfg.{file_format}"
    )
    serializable_args = to_dict(args)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_args, f, ensure_ascii=False, indent=4)
