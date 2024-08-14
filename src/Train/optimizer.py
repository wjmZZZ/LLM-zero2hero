from functools import partial
from typing import Any, Dict, List

import torch


class Optimizers:
    """Optimizers factory."""

    _optimizers = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": partial(torch.optim.SGD, momentum=0.9, nesterov=True),
        "RMSprop": partial(torch.optim.RMSprop, momentum=0.9, alpha=0.9),
        "Adadelta": torch.optim.Adadelta,
    }

    @classmethod
    def get_optimizer(cls, name: str) -> torch.optim.Optimizer:
        """
        Get the optimizer class by name.

        Args:
            name (str): The name of the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer class.

        Raises:
            ValueError: If the optimizer name is not found.
        """
        optimizer = cls._optimizers.get(name)
        if optimizer is None:
            raise ValueError(f"Optimizer '{name}' not found.")
        return optimizer


def get_optimizer(model: torch.nn.Module, args: Any) -> torch.optim.Optimizer:
    """
    Prepare the optimizer for the model.

    This function creates an optimizer with different parameter groups,
    applying different learning rates and weight decay settings based on
    the layer names and configuration.

    Args:
        model (torch.nn.Module): The model to optimize.
        args (Any): Configuration object containing training arguments.

    Returns:
        torch.optim.Optimizer: The configured optimizer.

    """
    no_decay = ["bias", "LayerNorm.weight"]
    differential_layers = args.training_args.differential_learning_rate_layers

    optimizer_class = Optimizers.get_optimizer(name=args.training_args.optimizer)

    def get_params(differential: bool, decay: bool) -> List[torch.nn.Parameter]:
        return [
            param
            for name, param in model.named_parameters()
            if (
                any(layer in name for layer in differential_layers) == differential
                and any(nd in name for nd in no_decay) == (not decay)
                and param.requires_grad
            )
        ]

    param_groups: List[Dict[str, Any]] = [
        {
            "params": get_params(differential=False, decay=True),
            "lr": args.training_args.learning_rate,
            "weight_decay": args.training_args.weight_decay,
        },
        {
            "params": get_params(differential=False, decay=False),
            "lr": args.training_args.learning_rate,
            "weight_decay": 0,
        },
        {
            "params": get_params(differential=True, decay=True),
            "lr": args.training_args.differential_learning_rate,
            "weight_decay": args.training_args.weight_decay,
        },
        {
            "params": get_params(differential=True, decay=False),
            "lr": args.training_args.differential_learning_rate,
            "weight_decay": 0,
        },
    ]

    optimizer = optimizer_class(
        param_groups,
        lr=args.training_args.learning_rate,
        weight_decay=args.training_args.weight_decay,
    )

    return optimizer
