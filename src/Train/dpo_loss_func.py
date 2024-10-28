from typing import Any, KeysView

import logging
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


# ===============================================================================
# DPO Loss Functions
# ===============================================================================
class DPOLoss(nn.Module):
    """
    Implements Direct Preference Optimization (DPO) loss.

    Args:
        cfg (Any): Configuration object for training settings.
    """
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.requires_reference_model = True
        self.loss_reduction = False

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        Computes DPO loss.

        Args:
            policy_chosen_logps (torch.FloatTensor): Log probabilities of chosen policy.
            policy_rejected_logps (torch.FloatTensor): Log probabilities of rejected policy.
            reference_chosen_logps (torch.FloatTensor): Log probabilities of chosen reference.
            reference_rejected_logps (torch.FloatTensor): Log probabilities of rejected reference.

        Returns:
            torch.Tensor: Mean of DPO loss, mean of chosen rewards, mean of rejected rewards.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        losses = self.get_losses(logits=pi_logratios - ref_logratios)

        chosen_rewards = self.cfg.training_args.beta * (
            policy_chosen_logps - reference_chosen_logps
        ).detach()
        rejected_rewards = self.cfg.training_args.beta * (
            policy_rejected_logps - reference_rejected_logps
        ).detach()

        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

    def get_losses(self, logits: torch.FloatTensor) -> torch.Tensor:
        """
        Computes the DPO losses based on logits.

        Args:
            logits (torch.FloatTensor): Logit differences between chosen and rejected samples.

        Returns:
            torch.Tensor: Computed loss values.
        """
        label_smoothing = 0
        losses = (
            -F.logsigmoid(self.cfg.training_args.beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-self.cfg.training_args.beta * logits) * label_smoothing
        )
        return losses


class LossFunction:
    """Loss function factory."""
    _loss_functions = {
        "DPOLoss": DPOLoss,
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


def get_loss_func(cfg: Any) -> nn.Module:
    """
    Load and instantiate the loss function based on configuration.

    Args:
        cfg (Any): Input configuration containing training arguments.

    Returns:
        nn.Module: Instantiated loss function.
    """
    return LossFunction.get_loss_function(cfg.training_args.loss_function)(cfg)