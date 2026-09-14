"""DINOv3 ViT-S/16 backbone and a PSPNet segmentation decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    """A convolution followed by BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PyramidPoolingModule(nn.Module):
    """PSPNet context aggregation at several spatial-bin sizes."""

    def __init__(self, in_channels: int, reduction_channels: int, bins: Iterable[int] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    ConvNormAct(in_channels, reduction_channels),
                )
                for bin_size in bins
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        pooled = [x]
        for stage in self.stages:
            pooled.append(F.interpolate(stage(x), size=(height, width), mode="bilinear", align_corners=False))
        return torch.cat(pooled, dim=1)


class PSPHead(nn.Module):
    """The trainable PSPNet head that maps dense DINO features to VOC logits."""

    def __init__(self, in_channels: int = 384, channels: int = 256, num_classes: int = 21) -> None:
        super().__init__()
        self.reduce = ConvNormAct(in_channels, channels, kernel_size=1)
        self.ppm = PyramidPoolingModule(channels, channels // 4)
        self.fuse = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=3),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.fuse(self.ppm(self.reduce(features)))


class DINOv3PSPNet(nn.Module):
    """DINOv3 ViT-S/16 plus a PSPNet decoder with optional last-block tuning.

    Args:
        dinov3_dir: Local clone/extraction of the official DINOv3 repository.
        backbone_weights: Required path to the approved ViT-S/16 checkpoint. Set
            ``None`` only for a structural smoke test; it must never be used for
            a real experiment.
        freeze_backbone: Keep all DINOv3 parameters fixed.
        trainable_backbone_blocks: Number of final ViT blocks to fine-tune.
    """

    def __init__(
        self,
        dinov3_dir: str | Path,
        backbone_weights: str | Path | None,
        num_classes: int = 21,
        freeze_backbone: bool = True,
        feature_layers: Iterable[int] = (11,),
        trainable_backbone_blocks: int = 0,
    ) -> None:
        super().__init__()
        dinov3_dir = Path(dinov3_dir).resolve()
        if not (dinov3_dir / "hubconf.py").is_file():
            raise FileNotFoundError(f"Official DINOv3 source not found: {dinov3_dir}")

        # The official loader normalizes local checkpoint paths to file:// URLs
        # and caches them. Keep that cache in this project rather than the
        # system user profile, which also makes the experiment self-contained.
        hub_cache = dinov3_dir.parent.parent / ".cache" / "torch"
        hub_cache.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_cache))

        self.backbone = torch.hub.load(
            str(dinov3_dir),
            "dinov3_vits16",
            source="local",
            pretrained=backbone_weights is not None,
            weights=str(Path(backbone_weights).resolve()) if backbone_weights else None,
        )
        self.patch_size = 16
        self.feature_layers = tuple(feature_layers)
        if not self.feature_layers:
            raise ValueError("feature_layers must contain at least one DINOv3 block index.")
        block_count = len(self.backbone.blocks)
        if any(layer < 0 or layer >= block_count for layer in self.feature_layers):
            raise ValueError(
                f"feature_layers must be between 0 and {block_count - 1} for this DINOv3 backbone."
            )
        if trainable_backbone_blocks < 0 or trainable_backbone_blocks > block_count:
            raise ValueError(f"trainable_backbone_blocks must be between 0 and {block_count}.")
        if not freeze_backbone and trainable_backbone_blocks == 0:
            trainable_backbone_blocks = block_count
        self.trainable_backbone_blocks = trainable_backbone_blocks
        self.freeze_backbone = trainable_backbone_blocks == 0

        # Start from a completely frozen pretrained backbone, then expose only
        # the requested final blocks and final LayerNorm to the optimizer.
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if trainable_backbone_blocks:
            for block in self.backbone.blocks[-trainable_backbone_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad_(True)
        # Each selected ViT-S/16 layer contributes a 384-channel dense map.  A
        # 1x1 projection at the start of PSPHead fuses and reduces the concat.
        self.decoder = PSPHead(in_channels=384 * len(self.feature_layers), num_classes=num_classes)

    def train(self, mode: bool = True) -> "DINOv3PSPNet":
        super().train(mode)
        # Frozen blocks stay in evaluation mode.  The selected final blocks use
        # training mode only when the surrounding model is training.
        self.backbone.eval()
        if mode and self.trainable_backbone_blocks:
            for block in self.backbone.blocks[-self.trainable_backbone_blocks:]:
                block.train()
            self.backbone.norm.train()
        return self

    def forward(self, images: Tensor) -> Tensor:
        input_size = images.shape[-2:]
        if input_size[0] % self.patch_size or input_size[1] % self.patch_size:
            raise ValueError(f"Input size {input_size} must be divisible by {self.patch_size}.")

        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.get_intermediate_layers(images, n=self.feature_layers, reshape=True)
        else:
            features = self.backbone.get_intermediate_layers(images, n=self.feature_layers, reshape=True)
        fused_features = torch.cat(features, dim=1)
        logits = self.decoder(fused_features)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
