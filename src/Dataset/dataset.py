import json
import logging
import os
from textwrap import dedent
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Sampler, SequentialSampler

from src.Dataset.data_util import (OrderedDistributedSampler,
                               preprocess_conversation, worker_init_fn)
from src.Others.exceptions import DataException

logger = logging.getLogger(__name__)


def load_data(args):
    """
    Load training and validation datasets based on configuration.

    Parameters:
        args: Configuration object containing dataset configuration.

    Returns:
        (train_data, valid_data)
    """
    if args.data_args.valid_strategy == "custom":
        if not os.path.exists(args.data_args.valid_data_dir):
            raise DataException(
                "Validation dataset not provided! Please provide a validation dataset or choose the auto validation strategy."
            )
        train_df = read_data(args.data_args.train_data_dir, args)
        valid_df = read_data(args.data_args.valid_data_dir, args)

    elif args.data_args.valid_strategy == "auto":
        df = read_data(args.data_args.train_data_dir, args)
        train_df, valid_df = train_test_split(
            df, test_size=args.data_args.valid_size, random_state=args.env_args.seed
        )
    else:
        raise DataException("No valid validation strategy provided.")

    train_df = preprocess_conversation(args, train_df)
    valid_df = preprocess_conversation(args, valid_df)
    
    if args.exp_args.task == "SFT":
        logger.info(
            f'\n\n{train_df[0][args.data_args.system_column][0]}\
                {train_df[0][args.data_args.prompt_column][0]}\
                {train_df[0][args.data_args.answer_column][0]}\n'
        )
    elif args.exp_args.task == "DPO":
        logger.info(
            f'\n\n{train_df[0][args.data_args.system_column][0]}\
                {train_df[0][args.data_args.prompt_column][0]}\n \
            chosen_response:\n {train_df[0][args.data_args.answer_column][0]}\n \
            rejected_response:\n {train_df[0][args.data_args.rejected_answer_column][0]}\n'
        )
        
    return train_df, valid_df


def read_data(file_path: str, args: Any) -> pd.DataFrame:
    """
    Read dataset.

    Parameters:
        file_path: Path to the dataset file.
        args: Configuration object containing dataset configuration.

    Returns:
        Data in DataFrame format.
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, lineterminator="\n").reset_index(drop=True)
    elif file_path.endswith(".pq") or file_path.endswith(".parquet"):
        try:
            df = pd.read_parquet(file_path, engine="pyarrow").reset_index(drop=True)
        except Exception:
            df = pd.read_parquet(file_path, engine="fastparquet").reset_index(drop=True)
    elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
        try:
            df = pd.read_json(file_path, lines=True).reset_index(drop=True)
        except:
            with open(file_path, "r") as f:
                df = pd.DataFrame(json.load(f))
    else:
        raise DataException(
            dedent(
                f"""
            Could not determine type of file {file_path}: 
            CSV (`.csv`), Parquet (`.pq` and `.parquet`), and JSON (`.json`) are supported.
        """
            )
        )

    if args.debug:
        if len(df) < 200:
            df = df  # Use all data
        else:
            df = df.sample(
                frac=0.1,  # Dynamically select 1% of the data
                random_state=args.env_args.seed,
            )
    return df


def get_train_dataloader(train_dataset: Any, args: Any):
    """
    Prepare training data loader.

    Parameters:
        train_dataset: Training dataset.
        args: Configuration object containing dataset configuration.

    Returns:
        Training data loader.
    """

    sampler: Sampler
    if args.env_args._distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.env_args._world_size,
            rank=args.env_args._local_rank,
            shuffle=True,
            seed=args.env_args.seed,
            drop_last=True,
        )
        sampler_length = len(sampler)
    else:
        sampler = None
        sampler_length = len(train_dataset)

    if (
        sampler_length < args.training_args.batch_size
        and args.training_args.drop_last_batch
    ):
        logger.warning(
            "Training data is too small when dropping the last batch. The number of samples should be at least the batch size multiplied by the number of GPUs."
            "Forcing to keep the last batch."
        )
        args.training_args.drop_last_batch = False
    if sampler_length <= 1:
        raise DataException("Data is too small to train the model.")

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=(sampler is None),
        batch_size=args.training_args.batch_size,
        num_workers=args.env_args.number_of_workers,
        pin_memory=True,
        collate_fn=None,
        drop_last=args.training_args.drop_last_batch,
        worker_init_fn=worker_init_fn,
    )

    logger.info(f"🚂 train_dataloader length: {len(train_dataloader)}")

    return train_dataloader


def get_valid_dataloader(valid_dataset: Any, args: Any):
    """
    Prepare validation data loader.

    Parameters:
        valid_dataset: Validation dataset.
        args: Configuration object containing dataset configuration.

    Returns:
        Validation data loader.
    """

    sampler: Sampler
    if args.env_args._distributed and args.infer_args.distributed_inference:
        sampler = OrderedDistributedSampler(
            valid_dataset,
            num_replicas=args.env_args._world_size,
            rank=args.env_args._local_rank,
        )
    else:
        sampler = SequentialSampler(valid_dataset)

    if args.infer_args.batch_size_inference != 0:
        batch_size = args.infer_args.batch_size_inference
    else:
        batch_size = args.training_args.batch_size

    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=args.env_args.number_of_workers,
        pin_memory=True,
        collate_fn=None,
        worker_init_fn=worker_init_fn,
    )

    logger.info(f"🔍 valid_dataloader length: {len(valid_dataloader)}")

    return valid_dataloader
