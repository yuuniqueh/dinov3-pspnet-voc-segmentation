"""PASCAL VOC 2012 semantic-segmentation dataset and paired transforms."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IGNORE_INDEX = 255
VOC_NUM_CLASSES = 21  # background plus the 20 PASCAL foreground categories
VOC_CLASS_NAMES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VOCTransform:
    """Apply spatial operations identically to the RGB image and label mask."""

    def __init__(self, image_size: int, training: bool) -> None:
        if image_size % 16:
            raise ValueError("image_size must be divisible by DINOv3's patch size (16).")
        self.image_size = image_size
        self.training = training

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        if self.training:
            scale = random.uniform(0.5, 2.0)
            width, height = image.size
            new_size = (max(self.image_size, round(height * scale)), max(self.image_size, round(width * scale)))
            image = TF.resize(image, new_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
            mask = TF.resize(mask, new_size, interpolation=InterpolationMode.NEAREST)

            pad_height = max(0, self.image_size - image.height)
            pad_width = max(0, self.image_size - image.width)
            if pad_height or pad_width:
                image = TF.pad(image, (0, 0, pad_width, pad_height), fill=0)
                mask = TF.pad(mask, (0, 0, pad_width, pad_height), fill=IGNORE_INDEX)

            top = random.randint(0, image.height - self.image_size)
            left = random.randint(0, image.width - self.image_size)
            image = TF.crop(image, top, left, self.image_size, self.image_size)
            mask = TF.crop(mask, top, left, self.image_size, self.image_size)
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
        else:
            image = TF.resize(image, (self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR, antialias=True)
            mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST)

        image_tensor = TF.normalize(TF.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return image_tensor, mask_tensor


class VOCDataset(VOCSegmentation):
    """torchvision's VOC2012 semantic masks with a paired image/mask transform."""

    def __init__(
        self,
        root: str | Path,
        image_set: Literal["train", "val"],
        image_size: int,
        download: bool = False,
    ) -> None:
        super().__init__(root=str(root), year="2012", image_set=image_set, download=download)
        self.paired_transform = VOCTransform(image_size, training=image_set == "train")

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])
        return self.paired_transform(image, mask)


def make_voc_loaders(
    root: str | Path,
    image_size: int = 512,
    batch_size: int = 2,
    workers: int = 0,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Construct Windows-safe VOC2012 train/validation data loaders."""
    train_set = VOCDataset(root, "train", image_size, download=download)
    val_set = VOCDataset(root, "val", image_size, download=download)
    loader_options = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_options),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_options),
    )
