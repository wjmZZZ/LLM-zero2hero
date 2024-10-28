import codecs
import logging
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import distributed
from torch.utils.data import Sampler

from src.Others.exceptions import DataException, MetricException
from src.Utils.utils import seed_everything

logger = logging.getLogger(__name__)


def batch_padding(
    args: Any,
    batch: Dict,
    training: bool = True,
    mask_key: str = "attention_mask",
    pad_keys: List[str] = ["input_ids", "attention_mask", "special_tokens_mask"],
    padding_side: str = "left",
) -> Dict:
    """Pads a batch according to set quantile, or cuts it at maximum length"""
    if args.env_args.compile_model:
        # logger.warning("Batch padding not functional with torch compile.")
        return batch
    elif batch[mask_key].sum() == 0:
        # continued pretraining
        return batch
    else:
        if padding_side == "left":
            idx = int(torch.where(batch[mask_key] == 1)[1].min())
        else:
            idx = int(torch.where(batch[mask_key] == 1)[1].max())

    if padding_side == "left":
        for key in pad_keys:
            if key in batch:
                batch[key] = batch[key][:, idx:].contiguous()
    else:
        idx += 1
        for key in pad_keys:
            if key in batch:
                batch[key] = batch[key][:, :idx].contiguous()

    return batch


def preprocess_conversation(args: Any, df: pd.DataFrame) -> List[Dict]:
    """
    Preprocess the conversation data from a DataFrame.
    Supports both ShareGPT format and DPO preference format.

    Args:
        args: Configuration arguments.
        df (pd.DataFrame): Input DataFrame containing conversation data.

    Returns:
        list: A list of preprocessed conversations, each containing prompts, responses, and system messages.

    Raises:
        DataException: If the number of systems, prompts, and responses don't match.
    """
    conversations = []
    if args.data_args.system_column is None:
        args.data_args.system_column = "system"
    
    for _, row in df.iterrows():
        systems = []
        prompts = []
        responses = []
        chosen_responses = []
        rejected_responses = []

        # Get system message from outer dict if exists (ShareGPT format)
        # Currently not supporting tools, function_call and observation
        try:
            current_system = (row[args.data_args.system_column] 
                            if not pd.isnull(row.get(args.data_args.system_column)) 
                            else args.data_args.system_default)
        except (KeyError, AttributeError):
            current_system = args.data_args.system_default
                   
        # Process conversation messages
        for conv in row["conversations"]:
            if conv["from"] == "human":
                systems.append(parse_system(args, current_system))
                prompts.append(parse_prompt(args, conv["value"]))
            elif conv["from"] == "assistant" or conv["from"] == "gpt":
                responses.append(parse_response(args, conv["value"]))
            elif conv["from"] == "chosen_gpt":
                chosen_responses.append(parse_response(args, conv["value"]))
            elif conv["from"] == "rejected_gpt":
                rejected_responses.append(parse_response(args, conv["value"]))

        # Determine the type of conversation and process accordingly
        if len(chosen_responses) > 0 and len(rejected_responses) > 0:
            # Preference type conversation
            if not (len(systems) == len(prompts) == len(chosen_responses) == len(rejected_responses)):
                raise DataException(f"Warning: Mismatch in preference type conversation - systems ({len(systems)}), prompts ({len(prompts)}), "
                      f"chosen responses ({len(chosen_responses)}), and rejected responses ({len(rejected_responses)}).")
                continue  # Error indicates data issue, can move continue to skip erroneous samples ⬆️
            conversations.append({
                f"{args.data_args.system_column}": systems,                
                f"{args.data_args.prompt_column}": prompts,
                f"{args.data_args.answer_column}": chosen_responses,
                f"{args.data_args.rejected_answer_column}": rejected_responses
            })
        else:
            # Standard type conversation
            if not (len(systems) == len(prompts) == len(responses)):
                continue
                raise DataException(f"Warning: Mismatch in standard type conversation - systems ({len(systems)}), "
                      f"prompts ({len(prompts)}), and responses ({len(responses)}).")
                continue  # Error indicates data issue, can move continue to skip erroneous samples ⬆️
            conversations.append({
                f"{args.data_args.system_column}": systems,                
                f"{args.data_args.prompt_column}": prompts,
                f"{args.data_args.answer_column}": responses,
            })

    return conversations


def parse_system(args: Any, system: str):
    if (
        args.data_args.system_prefix == "None"
        and args.data_args.system_suffix == "None"
    ):
        return system
    system_prefix = codecs.decode(args.data_args.system_prefix, "unicode_escape")
    system_suffix = codecs.decode(args.data_args.system_suffix, "unicode_escape")
    system = f"{system_prefix}{system}{system_suffix}"
    return system


def parse_prompt(args: Any, prompt: str):
    prompt_prefix = codecs.decode(args.data_args.prompt_prefix, "unicode_escape")
    prompt_suffix = codecs.decode(args.data_args.prompt_suffix, "unicode_escape")
    prompt = f"{prompt_prefix}{prompt}{prompt_suffix}"
    return prompt


def parse_response(args: Any, response: str):
    response_prefix = codecs.decode(args.data_args.response_prefix, "unicode_escape")
    response_suffix = codecs.decode(args.data_args.response_suffix, "unicode_escape")
    response = f"{response_prefix}{response}{response_suffix}"
    return response


def nested_dicts_to_dataframe(data, args):
    if args.exp_args.task == "SFT":
        expanded_data = {"system": [], "prompt": [], "response": []}
    elif args.exp_args.task == "DPO":
        expanded_data = {"system": [], "prompt": [], "chosen_response": [], "rejected_response": []}
    else:
        raise MetricException(f"Unsupported task type: {args.exp_args.task}")

    for item in data:
        systems = item.get(f"{args.data_args.system_column}", [])
        prompts = item.get(f"{args.data_args.prompt_column}", [])
        
        if args.exp_args.task == "SFT":
            responses = item.get(f"{args.data_args.answer_column}", [])
            if not (len(systems) == len(prompts) == len(responses)):
                logger.warning(
                    f"Data anomaly: Mismatch in the number of system messages ({len(systems)}), "
                    f"prompts ({len(prompts)}), and responses ({len(responses)})."
                )
                continue
            items_to_process = zip(systems, prompts, responses)
        else:  # DPO
            chosen_responses = item.get(f"{args.data_args.answer_column}", [])
            rejected_responses = item.get(f"{args.data_args.rejected_answer_column}", [])
            if not (len(systems) == len(prompts) == len(chosen_responses) == len(rejected_responses)):
                logger.warning(
                    f"Data anomaly: Mismatch in the number of system messages ({len(systems)}), "
                    f"prompts ({len(prompts)}), chosen responses ({len(chosen_responses)}), "
                    f"and rejected responses ({len(rejected_responses)})."
                )
                continue
            items_to_process = zip(systems, prompts, chosen_responses, rejected_responses)

        for items in items_to_process:
            # Process system message
            system = items[0].replace(args.data_args.system_prefix, "").replace(
                args.data_args.system_suffix, ""
            )
            expanded_data["system"].append(system)

            # Process prompt
            prompt = items[1].replace(args.data_args.prompt_prefix, "").replace(
                args.data_args.prompt_suffix, ""
            )
            expanded_data["prompt"].append(prompt)
            
            # Process response(s)
            if args.exp_args.task == "SFT":
                response = items[2].replace(args.data_args.response_prefix, "").replace(
                    args.data_args.response_suffix, ""
                )
                expanded_data["response"].append(response)
            else:  # DPO
                chosen = items[2].replace(args.data_args.response_prefix, "").replace(
                    args.data_args.response_suffix, ""
                )
                rejected = items[3].replace(args.data_args.response_prefix, "").replace(
                    args.data_args.response_suffix, ""
                )
                expanded_data["chosen_response"].append(chosen)
                expanded_data["rejected_response"].append(rejected)

    return pd.DataFrame(expanded_data)


def worker_init_fn(worker_id: int) -> None:
    """
    Set random seed for each worker process.

    Args:
        worker_id: ID of the corresponding worker process.
    """

    if "PYTHONHASHSEED" in os.environ:
        seed = int(os.environ["PYTHONHASHSEED"]) + worker_id
    else:
        seed = np.random.get_state()[1][0] + worker_id  # type: ignore
    seed_everything(seed)


class OrderedDistributedSampler(Sampler):
    """
    Distributed sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with torch.nn.parallel.DistributedDataParallel.
    In such a case, each process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a unique subset of the original dataset.
    Source:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/distributed_sampler.py
    """

    def __init__(
        self,
        dataset: Any,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """
        Args:
            dataset: Dataset used for sampling
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process within num_replicas
        """

        if num_replicas is None:
            if not distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = distributed.get_world_size()
        if rank is None:
            if not distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices += [0] * (self.total_size - len(indices))
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[
            self.rank * self.num_samples : self.rank * self.num_samples
            + self.num_samples
        ]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
