import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional
from transformers import HfArgumentParser
import yaml


from src.Args.sft_args import ExperimentArguments, ModelArguments, InferenceArguments, EnvironmentArguments, to_dict
from src.Args.dpo_args import DPODatasetArguments, DPOTrainingArguments
from src.Utils.utils import flatten_dict

# ===================================================================================
# Get Arguments
# ===================================================================================
def get_args(args: Optional[Dict[str, Any]] = None):
    parser = HfArgumentParser(
        (
        ExperimentArguments,
        DPODatasetArguments,
        ModelArguments,
        DPOTrainingArguments,
        InferenceArguments,
        EnvironmentArguments,
        )
    )

    if args is not None:
        (
            exp_args,
            data_args,
            dpo_data_args,
            model_args,
            training_args,
            dpo_training_args,
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
        try:
            parsed_args = parser.parse_dict(flat_dict)
            return parsed_args
        except Exception as e:
            print(f"Error parsing arguments: {e}")
            print("Parser fields:")
            for field in parser.dataclass_types:
                print(f"  {field.__name__}: {field.__annotations__}")
            raise
    else:
        return parser.parse_args_into_dataclasses(remaining_args)


def save_args(args, file_format: str = "json") -> None:
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