"""Useful semantic-segmentation metrics derived from a VOC confusion matrix."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class SegmentationMetrics:
    """Accumulate predictions and calculate VOC-relevant segmentation metrics.

    The confusion matrix stores ground-truth classes as rows and predicted
    classes as columns. Void pixels (VOC label 255) are never counted.
    """

    def __init__(self, num_classes: int = 21, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: Tensor, target: Tensor) -> None:
        prediction = logits.argmax(dim=1).detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        encoded = self.num_classes * target[valid] + prediction[valid]
        self.confusion += torch.bincount(encoded, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        """Return mIoU, pixel accuracy, mAcc, FWIoU, and per-class diagnostics."""
        confusion = self.confusion.float()
        true_count = confusion.sum(dim=1)
        predicted_count = confusion.sum(dim=0)
        intersection = torch.diag(confusion)
        union = true_count + predicted_count - intersection
        class_iou = torch.where(union > 0, intersection / union, torch.nan)
        class_accuracy = torch.where(true_count > 0, intersection / true_count, torch.nan)
        total = true_count.sum()
        pixel_accuracy = intersection.sum() / total if total > 0 else torch.tensor(torch.nan)
        frequencies = true_count / total if total > 0 else torch.zeros_like(true_count)
        fw_iou = torch.nansum(frequencies * class_iou)
        return {
            "miou": float(torch.nanmean(class_iou)),
            "pixel_accuracy": float(pixel_accuracy),
            "mean_accuracy": float(torch.nanmean(class_accuracy)),
            "fw_iou": float(fw_iou),
            "class_iou": [float(value) for value in class_iou],
            "class_accuracy": [float(value) for value in class_accuracy],
            "confusion_matrix": self.confusion.tolist(),
        }
