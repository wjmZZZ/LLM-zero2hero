from typing import Any

import torch
from transformers import (get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)


class Scheduler:
    """A class to manage different learning rate schedulers."""

    _schedulers = {
        "Constant": get_constant_schedule_with_warmup,
        "Cosine": get_cosine_schedule_with_warmup,
        "Linear": get_linear_schedule_with_warmup,
    }

    @classmethod
    def get_scheduler(cls, name: str) -> callable:
        """
        Get a scheduler function by name.

        Args:
            name (str): The name of the scheduler.

        Returns:
            callable: The scheduler function.

        Raises:
            ValueError: If the scheduler name is not found.
        """
        try:
            return cls._schedulers[name]
        except KeyError:
            raise ValueError(
                f"Scheduler '{name}' is not in the list of available schedulers."
            )


def get_scheduler(
    args: Any,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Prepare the learning rate scheduler.

    Args:
        args (Any): Input configuration containing training arguments.
        optimizer (torch.optim.Optimizer): Model optimizer.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Configured learning rate scheduler.
    """
    scheduler_func = Scheduler.get_scheduler(args.training_args.schedule)

    total_steps = (
        args.training_args.num_train_epochs * args.training_args._training_epoch_steps
    )
    warmup_steps = (
        args.training_args.warmup_epochs * args.training_args._training_epoch_steps
    )

    scheduler = scheduler_func(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    return scheduler
