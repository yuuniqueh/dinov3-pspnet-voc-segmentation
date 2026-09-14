"""Loss functions for PASCAL VOC semantic segmentation."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def foreground_dice_loss(logits: Tensor, targets: Tensor, ignore_index: int = 255, epsilon: float = 1e-6) -> Tensor:
    """Soft Dice loss over foreground classes that occur in the current batch.

    Cross-entropy already supervises every pixel, including background.  This
    extra term focuses on overlapping foreground regions, so a batch containing
    no instance of a class does not incorrectly reward that class.
    """
    num_classes = logits.shape[1]
    valid = targets.ne(ignore_index)
    safe_targets = targets.masked_fill(~valid, 0)
    probabilities = F.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).to(probabilities.dtype)
    valid_mask = valid.unsqueeze(1).to(probabilities.dtype)
    probabilities, one_hot = probabilities * valid_mask, one_hot * valid_mask

    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present_foreground = one_hot.sum(dim=(0, 2, 3))[1:] > 0
    if not torch.any(present_foreground):
        return logits.sum() * 0.0
    dice = (2.0 * intersection[1:] + epsilon) / (denominator[1:] + epsilon)
    return 1.0 - dice[present_foreground].mean()


class SegmentationCriterion(nn.Module):
    """Cross-entropy optionally regularized with foreground soft Dice loss."""

    def __init__(self, ignore_index: int = 255, dice_weight: float = 0.0) -> None:
        super().__init__()
        if dice_weight < 0:
            raise ValueError("dice_weight must be non-negative.")
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        loss = self.cross_entropy(logits, targets)
        if self.dice_weight:
            loss = loss + self.dice_weight * foreground_dice_loss(logits, targets, self.ignore_index)
        return loss
