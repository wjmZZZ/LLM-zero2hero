from typing import Any

import torch
from torch import nn


# ===============================================================================
# Casual Language Model
# ===============================================================================
class TokenAveragedCrossEntropyLoss(nn.Module):
    """
    Token-averaged cross entropy loss for casual language models.

    This loss function computes the average cross entropy loss across all tokens in the batch.

    Args:
        args (Any): Configuration object.
    """

    def __init__(self, args: Any):
        super().__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the token-averaged cross entropy loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
            labels (torch.Tensor): True labels of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Computed loss value.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )


class SampleAveragedCrossEntropyLoss(nn.Module):
    """
    Sample-averaged cross entropy loss for casual language models.

    This loss function computes the average cross entropy loss across all samples in the batch.

    Args:
        args (Any): Configuration object.
    """

    def __init__(self, args: Any):
        super().__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the sample-averaged cross entropy loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
            labels (torch.Tensor): True labels of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Computed loss value.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = 0
        for i in range(labels.shape[0]):
            loss += self.loss_fn(shift_logits[i], shift_labels[i])
        loss /= labels.shape[0]
        return loss


# ===============================================================================
# Classification Language Model
# ===============================================================================
class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss for classification tasks.

    Args:
        args (Any): Configuration object.
    """

    def __init__(self, args: Any):
        super().__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross entropy loss for classification.

        Args:
            logits (torch.Tensor): Predicted logits.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed loss value.
        """
        return self.loss_fn(logits, labels.reshape(-1).long())


class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary cross entropy loss for binary classification tasks.

    Args:
        args (Any): Configuration object.
    """

    def __init__(self, args: Any):
        super().__init__()
        self.args = args
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the binary cross entropy loss.

        Args:
            logits (torch.Tensor): Predicted logits.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed loss value.
        """
        return self.loss_fn(logits, labels)


class LossFunction:
    """Loss function factory."""

    _loss_functions = {
        "TokenAveragedCrossEntropyLoss": TokenAveragedCrossEntropyLoss,
        "SampleAveragedCrossEntropyLoss": SampleAveragedCrossEntropyLoss,
        "CrossEntropyLoss": CrossEntropyLoss,
        "BinaryCrossEntropyLoss": BinaryCrossEntropyLoss,
    }

    @classmethod
    def get_loss_function(cls, name: str) -> nn.Module:
        """
        Get the loss function class by name.

        Args:
            name (str): Name of the loss function.

        Returns:
            nn.Module: The loss function class.

        Raises:
            ValueError: If the loss function name is not found.
        """
        loss_function = cls._loss_functions.get(name)
        if loss_function is None:
            raise ValueError(f"Loss function '{name}' not found.")
        return loss_function


def get_loss_func(args: Any) -> nn.Module:
    """
    Load and instantiate the loss function based on configuration.

    Args:
        args (Any): Input configuration containing training arguments.

    Returns:
        nn.Module: Instantiated loss function.
    """
    return LossFunction.get_loss_function(args.training_args.loss_function)(args)
