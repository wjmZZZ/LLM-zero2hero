import logging
import os
import shutil
from typing import Any, Tuple

import deepspeed
import numpy as np
import torch
from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from Dataset.dataset import OrderedDistributedSampler
from Enviroment.env_deepspeed import get_ds_config
from Enviroment.env_utils import seed_everything, sync_across_processes
from Others.exceptions import EnviromentException

logger = logging.getLogger(__name__)


def Prepare_environment(args: Any) -> None:
    """
    Prepare the training environment.

    This function sets up the distributed training environment, initializes the process group,
    sets the device, and ensures reproducibility by setting a global random seed.

    Args:
        args (Any): Configuration arguments.

    Raises:
        EnviromentException: If there's an incompatibility between backbone type and DeepSpeed.
    """
    if (
        args.model_args.backbone_dtype in ["int8", "int4"]
        and args.env_args.use_deepspeed
    ):
        raise EnviromentException(
            f"""
            ❌ Deepspeed does not support backbone type {args.model_args.backbone_dtype}.
            🔧 Please set backbone type to float16 or bfloat16 for using deepspeed.
            """
        )

    args.env_args._distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if args.env_args._distributed:
        args.env_args._local_rank = int(os.environ["LOCAL_RANK"])
        args.env_args._device = f"cuda:{args.env_args._local_rank}"
        if args.env_args.use_deepspeed:
            deepspeed.init_distributed()
        else:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

        args.env_args._cpu_comm = torch.distributed.new_group(backend="gloo")

        args.env_args._world_size = torch.distributed.get_world_size()
        args.env_args._rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.env_args._rank)
        logger.info(
            f"""
            🌟 Training in distributed mode with multiple processes
            💻 1 GPU per process. Process {args.env_args._rank}
            🌐 total: {args.env_args._world_size}
            🔢 local rank: {args.env_args._local_rank}
            """
        )

        args.env_args.seed = int(
            sync_across_processes(
                np.array([args.env_args.seed]),
                args.env_args._world_size,
                group=args.env_args._cpu_comm,
            )[0]
        )
    else:
        args.env_args._local_rank = 0
        if torch.cuda.is_available() and len(args.env_args.gpus) > 0:
            args.env_args._device = "cuda:0"
        else:
            args.env_args._device = "cpu"
            logger.warning("⚠️ Training on CPU. This will be slow.")

    seed_everything(args.env_args.seed)
    logger.info(f"🌱 Global random seed: {args.env_args.seed}")


def wrap_model_distributed(
    model: torch.nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    args: Any,
) -> Tuple[torch.nn.Module, Optimizer, DataLoader, DataLoader, _LRScheduler]:
    """
    Wrap the model for distributed training.

    This function prepares the model, optimizer, and dataloaders for distributed training,
    either using DeepSpeed or PyTorch's DistributedDataParallel.

    Args:
        model (torch.nn.Module): The model to be wrapped.
        optimizer (Optimizer): The optimizer.
        lr_scheduler (_LRScheduler): The learning rate scheduler.
        train_dataloader (DataLoader): The training data loader.
        valid_dataloader (DataLoader): The validation data loader.
        args (Any): Configuration object.

    Returns:
        Tuple[torch.nn.Module, Optimizer, DataLoader, DataLoader, _LRScheduler]:
        The wrapped model, optimizer, train dataloader, valid dataloader, and lr scheduler.
    """
    if args.env_args.use_deepspeed:
        ds_config = get_ds_config(args)

        if not args.training_args.lora:
            ds_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
                model=model.backbone,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                training_data=train_dataloader.dataset,
                config_params=ds_config,
            )
            model.backbone = ds_engine
        else:
            ds_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
                model=model.backbone.base_model.model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                training_data=train_dataloader.dataset,
                config_params=ds_config,
            )
            model.backbone.base_model.model = ds_engine

        model.init_deepspeed()

        valid_dataloader = DeepSpeedDataLoader(
            valid_dataloader.dataset,
            batch_size=valid_dataloader.batch_size,
            local_rank=args.env_args._local_rank,
            pin_memory=True,
            tput_timer=None,
            data_sampler=OrderedDistributedSampler(
                valid_dataloader.dataset,
                num_replicas=args.env_args._world_size,
                rank=args.env_args._local_rank,
            ),
        )
    else:
        find_unused_parameters = args.env_args.find_unused_parameters and not getattr(
            args.model_args, "gradient_checkpointing", False
        )
        model = DistributedDataParallel(
            model,
            device_ids=[args.env_args._local_rank],
            find_unused_parameters=find_unused_parameters,
        )

    return model, optimizer, train_dataloader, valid_dataloader, lr_scheduler


def check_disk_space(
    model: torch.nn.Module, path: str, use_deepspeed: bool = False
) -> None:
    """
    Check if there's enough disk space to save model weights.

    This function calculates the required disk space for saving the model weights
    and checks if there's enough free space available.

    Args:
        model (torch.nn.Module): The model whose weights need to be saved.
        path (str): The path where the weights will be saved.
        use_deepspeed (bool, optional): Whether DeepSpeed is being used. Defaults to False.

    Raises:
        EnviromentException: If there's not enough disk space to save the model weights.
    """
    total, used, free = shutil.disk_usage(path)

    model_size_in_bytes = sum(
        param.numel()
        * (
            1
            if param.data.dtype in [torch.int8, torch.uint8]
            else 2 if param.data.dtype in [torch.float16, torch.bfloat16] else 4
        )
        for param in model.parameters()
    )

    if use_deepspeed:
        model_size_in_bytes *= 2  # Double space for DeepSpeed engine conversion

    required_space = model_size_in_bytes * 1.03  # 3% margin
    if required_space < free:
        logger.info("💾 There is enough space to save model weights.")
    else:
        raise EnviromentException(
            f"""
            ❌ Not enough space to save model weights.
            🔢 Required space: {required_space / (1024 * 1024):.2f}MB, 
            💾 Available space: {free / (1024 * 1024):.2f}MB.
            """
        )
