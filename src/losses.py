"""
losses.py

Loss functions for SIGNAL-Preop-style multitask binary classification.

Implemented components:
1. Binary focal loss for class imbalance.
2. Uncertainty-based task weighting for multitask learning.
3. Combined multitask focal loss used by the Longformer multitask model.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TASKS = ["DD", "GAD", "IND", "SR"]


class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss for imbalanced binary classification.

    Args:
        alpha: Optional class-balancing factor. If None, no alpha weighting is used.
        gamma: Focusing parameter. Larger values emphasize hard examples.
        reduction: "mean", "sum", or "none".
    """

    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', or 'none'.")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw logits with shape [batch_size].
            targets: Binary labels with shape [batch_size].
        """
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        focal_weight = (1.0 - p_t).pow(self.gamma)
        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


class MultitaskUncertaintyFocalLoss(nn.Module):
    """
    Multitask focal loss with uncertainty-based task weighting.

    The multitask objective is:

        total_loss = sum_i exp(-s_i) * L_i + s_i

    where L_i is the focal loss for task i and s_i is a learnable log variance.

    This implementation follows the uncertainty-weighting idea commonly used for
    balancing multiple task losses. The learnable log_vars should be optimized
    together with model parameters.
    """

    def __init__(
        self,
        tasks: Sequence[str] = DEFAULT_TASKS,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        use_uncertainty_weighting: bool = True,
    ) -> None:
        super().__init__()

        self.tasks = list(tasks)
        self.use_uncertainty_weighting = use_uncertainty_weighting

        self.focal_loss = BinaryFocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction="mean",
        )

        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(len(self.tasks)))
        else:
            self.register_buffer("log_vars", torch.zeros(len(self.tasks)))

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: Tensor of shape [batch_size, num_tasks].
            labels: Tensor of shape [batch_size, num_tasks].

        Returns:
            Dictionary containing total loss and task-specific losses.
        """
        if logits.shape != labels.shape:
            raise ValueError(
                f"logits and labels must have the same shape, got "
                f"{logits.shape} and {labels.shape}."
            )

        if logits.shape[1] != len(self.tasks):
            raise ValueError(
                f"Expected {len(self.tasks)} task logits, got {logits.shape[1]}."
            )

        task_losses = []
        output: Dict[str, torch.Tensor] = {}

        for idx, task in enumerate(self.tasks):
            task_loss = self.focal_loss(
                logits=logits[:, idx],
                targets=labels[:, idx],
            )
            task_losses.append(task_loss)
            output[f"loss_{task}"] = task_loss.detach()

        total_loss = torch.zeros((), device=logits.device)

        for idx, task_loss in enumerate(task_losses):
            if self.use_uncertainty_weighting:
                precision = torch.exp(-self.log_vars[idx])
                weighted_loss = precision * task_loss + self.log_vars[idx]
            else:
                weighted_loss = task_loss

            total_loss = total_loss + weighted_loss

        output["loss"] = total_loss
        return output


def build_multitask_loss(
    tasks: Sequence[str] = DEFAULT_TASKS,
    alpha: Optional[float] = 0.25,
    gamma: float = 2.0,
    use_uncertainty_weighting: bool = True,
) -> MultitaskUncertaintyFocalLoss:
    """
    Convenience function for building the multitask loss.
    """
    return MultitaskUncertaintyFocalLoss(
        tasks=tasks,
        alpha=alpha,
        gamma=gamma,
        use_uncertainty_weighting=use_uncertainty_weighting,
    )