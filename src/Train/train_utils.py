import logging
from typing import Dict, List, Union

import torch

logger = logging.getLogger(__name__)


def batch_to_device(
    batch: Union[Dict, List, torch.Tensor], device: str
) -> Union[Dict, List, torch.Tensor, str]:
    """
    Send batch data to the specified device.

    Args:
        batch (Union[Dict, List, torch.Tensor]): Input batch data.
        device (str): Target device to send data to.

    Returns:
        Union[Dict, List, torch.Tensor, str]: Batch data on the specified device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)) and all(
        isinstance(item, str) for item in batch
    ):
        return batch
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(item, device) for item in batch)
    else:
        return batch


def calculate_steps(args, train_dataloader, valid_dataloader):
    """
    Calculate various steps for training and validation.

    Args:
        args: Arguments object containing configuration parameters.
        train_dataloader: DataLoader for training data.
        valid_dataloader: DataLoader for validation data.

    Returns:
        Tuple[int, int, int]: A tuple containing:
            - total_training_steps: Total number of training steps.
            - validation_steps: Number of steps between validations.
    """
    total_training_steps = (
        len(train_dataloader)
        * args.training_args.num_train_epochs
        * args.training_args.batch_size
        * args.env_args._world_size
    )
    args.training_args._training_epoch_steps = len(train_dataloader)

    if args.infer_args.batch_size_inference != 0:
        val_batch_size = args.infer_args.batch_size_inference
    else:
        val_batch_size = args.training_args.batch_size

    num_validations_per_epoch = args.training_args.num_validations_per_epoch

    args.training_args._validation_steps = (
        len(train_dataloader) // num_validations_per_epoch
        if num_validations_per_epoch > 0
        else 0
    )
    return (
        total_training_steps,
        args.training_args._validation_steps,
    )


def compile_model(model: torch.nn.Module, args) -> torch.nn.Module:
    """
    Compile the model using torch.compile if configured.

    Args:
        model (torch.nn.Module): The model to be compiled.
        args: Arguments object containing configuration parameters.

    Returns:
        torch.nn.Module: The compiled model (or original model if compilation is skipped).
    """
    if args.env_args.compile_model:
        if args.env_args.use_deepspeed:
            logger.warning(
                "Deepspeed is activated, but it does not support torch.compile. "
                "Skipping compilation for this experiment."
            )
        else:
            if args.env_args._distributed:
                model.module.backbone = torch.compile(model.module.backbone)
            else:
                model.backbone = torch.compile(model.backbone)
    return model


def get_torch_dtype(dtype: str) -> torch.dtype:
    """
    Get the corresponding torch.dtype based on the input string.

    Args:
        dtype (str): String representation of the desired dtype.

    Returns:
        torch.dtype: The corresponding torch.dtype.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype, torch.float32)
