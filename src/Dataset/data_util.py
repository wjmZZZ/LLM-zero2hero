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

from Others.exceptions import DataException
from Utils.utils import seed_everything

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


def preprocess_conversation(args, df):
    """
    Preprocess the conversation data from a DataFrame.

    Args:
        args: Configuration arguments.
        df (pd.DataFrame): Input DataFrame containing conversation data.

    Returns:
        list: A list of preprocessed conversations, each containing prompts, responses, and system messages.

    Raises:
        DataException: If the number of systems, prompts, and responses don't match.
    """
    conversations = []

    for item in df["conversations"]:
        prompts = []
        responses = []
        systems = []

        for conv in item:
            current_system = ""
            if "system" in conv:
                current_system = parse_system(args, conv["system"])

            if conv["from"] == "human":
                systems.append(parse_system(args, current_system))
                prompts.append(parse_prompt(args, conv["value"]))
            elif conv["from"] == "assistant" or conv["from"] == "gpt":
                responses.append(parse_response(args, conv["value"]))

        # Check if the number of systems, prompts, and responses match
        if not (len(systems) == len(prompts) == len(responses)):
            continue
            raise DataException(
                f"Data anomaly: Mismatch in the number of systems ({len(systems)}), "
                f"prompts ({len(prompts)}), and responses ({len(responses)})."
            )

        conversations.append(
            {"prompts": prompts, "responses": responses, "systems": systems}
        )

    return conversations


def parse_system(args: Any, system: str):
    if (
        args.data_args.system_prefix == "None"
        and args.data_args.system_suffix == "None"
    ):
        return system
    if system == "":
        system = args.data_args.system_defalut
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


import re

import pandas as pd
def nested_dicts_to_dataframe(data, args):
    expanded_data = {"system": [], "prompt": [], "response": []}

    for item in data:
        systems = item.get("systems", [])
        prompts = item.get("prompts", [])
        responses = item.get("responses", [])

        # Check if the number of system messages, prompts, and responses match
        if not (len(systems) == len(prompts) == len(responses)):
            logger.warning(f"Data anomaly: Mismatch in the number of system messages ({len(systems)}), "
                           f"prompts ({len(prompts)}), and responses ({len(responses)}).")
            continue

        for system, prompt, response in zip(systems, prompts, responses):
            # Remove special characters, keep only the original text
            system = system.replace(args.data_args.system_prefix, "").replace(args.data_args.system_suffix, "")
            prompt = prompt.replace(args.data_args.prompt_prefix, "").replace(args.data_args.prompt_suffix, "")
            response = response.replace(args.data_args.response_prefix, "").replace(args.data_args.response_suffix, "")

            expanded_data["system"].append(system)
            expanded_data["prompt"].append(prompt)
            expanded_data["response"].append(response)

    return pd.DataFrame(expanded_data)

# def nested_dicts_to_dataframe(data, args):
#     expanded_data = {"system": [], "prompt": [], "response": []}

#     # 定义要移除的前缀和后缀
#     system_pattern = (
#         re.escape(args.data_args.system_prefix)
#         + "(.+?)"
#         + re.escape(args.data_args.system_suffix)
#     )
#     prompt_pattern = (
#         re.escape(args.data_args.prompt_prefix)
#         + "(.+?)"
#         + re.escape(args.data_args.prompt_suffix)
#     )
#     response_pattern = "(.+?)" + re.escape(args.data_args.response_suffix)

#     for item in data:
#         max_length = max(
#             len(item["prompts"]), len(item["responses"]), len(item["systems"])
#         )
#         for i in range(max_length):
#             # 使用正则表达式提取内容
#             system = (
#                 re.search(system_pattern, item["systems"][i]).group(1)
#                 if i < len(item["systems"])
#                 else None
#             )
#             prompt = (
#                 re.search(prompt_pattern, item["prompts"][i]).group(1)
#                 if i < len(item["prompts"])
#                 else None
#             )
#             response = (
#                 re.search(response_pattern, item["responses"][i]).group(1)
#                 if i < len(item["responses"])
#                 else None
#             )

#             expanded_data["system"].append(system)
#             expanded_data["prompt"].append(prompt)
#             expanded_data["response"].append(response)

#     return pd.DataFrame(expanded_data)


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
