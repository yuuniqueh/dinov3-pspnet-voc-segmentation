"""Train frozen-DINOv3 + PSPNet on PASCAL VOC 2012 semantic segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import IGNORE_INDEX, VOC_CLASS_NAMES, VOC_NUM_CLASSES, make_voc_loaders
from src.losses import SegmentationCriterion
from src.metrics import SegmentationMetrics
from src.model import DINOv3PSPNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dinov3-dir", type=Path, default=Path("third_party/dinov3-main"))
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        default=Path("checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vits16_pspnet"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 first on Windows; raise only after a successful run.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=0,
        help="Number of final DINOv3 ViT blocks to fine-tune (0 keeps the backbone frozen).",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help="Learning rate for unfrozen DINOv3 blocks; used only when --unfreeze-last-blocks is positive.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="Clip trainable gradient norm after AMP unscaling; 0 disables clipping.",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.0,
        help="Add this multiple of foreground soft Dice loss to cross-entropy (0 keeps CE only).",
    )
    parser.add_argument(
        "--feature-layers",
        type=int,
        nargs="+",
        default=[11],
        help="Zero-based DINOv3 block indices to concatenate before PSPNet; ViT-S/16 has 0-11.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="For a fast smoke test only.")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="For a fast smoke test only.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None,
) -> tuple[float, dict[str, object]]:
    model.eval()
    metric, loss_total, sample_count = SegmentationMetrics(VOC_NUM_CLASSES, IGNORE_INDEX), 0.0, 0
    for batch_index, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks)
        metric.update(logits, masks)
        loss_total += loss.item() * images.size(0)
        sample_count += images.size(0)
    return loss_total / max(sample_count, 1), metric.compute()


def append_overall_metrics(path: Path, epoch: int, train_loss: float, val_loss: float, train: dict[str, object], val: dict[str, object], learning_rate: float) -> None:
    """Append one compact, Excel-friendly row of epoch-level results."""
    row = {
        "epoch": epoch, "learning_rate": learning_rate,
        "train_loss": train_loss, "val_loss": val_loss,
        "train_miou": train["miou"], "val_miou": val["miou"],
        "train_pixel_accuracy": train["pixel_accuracy"], "val_pixel_accuracy": val["pixel_accuracy"],
        "train_mean_accuracy": train["mean_accuracy"], "val_mean_accuracy": val["mean_accuracy"],
        "train_fw_iou": train["fw_iou"], "val_fw_iou": val["fw_iou"],
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_per_class_metrics(path: Path, epoch: int, split: str, metrics: dict[str, object]) -> None:
    """Append one row per VOC class so weak categories can be diagnosed."""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "split", "class_id", "class_name", "iou", "class_accuracy"])
        if write_header:
            writer.writeheader()
        for class_id, class_name in enumerate(VOC_CLASS_NAMES):
            writer.writerow({
                "epoch": epoch, "split": split, "class_id": class_id, "class_name": class_name,
                "iou": metrics["class_iou"][class_id],
                "class_accuracy": metrics["class_accuracy"][class_id],
            })


def save_best_diagnostics(output_dir: Path, epoch: int, metrics: dict[str, object]) -> None:
    """Write best validation metrics in formats useful for a report."""
    summary = {key: value for key, value in metrics.items() if key != "confusion_matrix"}
    summary["epoch"] = epoch
    with (output_dir / "best_validation_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    with (output_dir / "best_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_class\\predicted_class", *VOC_CLASS_NAMES])
        for class_name, row in zip(VOC_CLASS_NAMES, metrics["confusion_matrix"]):
            writer.writerow([class_name, *row])


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this project configuration.")
    if not args.backbone_weights.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {args.backbone_weights}")
    if args.image_size % 16:
        raise ValueError("--image-size must be divisible by 16.")

    set_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    amp_enabled = not args.no_amp
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Choose a new --output-dir to preserve experiment records."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.output_dir / "tensorboard")
    train_loader, val_loader = make_voc_loaders(
        args.data_root, args.image_size, args.batch_size, args.workers, download=False
    )
    model = DINOv3PSPNet(
        args.dinov3_dir,
        args.backbone_weights,
        VOC_NUM_CLASSES,
        freeze_backbone=args.unfreeze_last_blocks == 0,
        feature_layers=args.feature_layers,
        trainable_backbone_blocks=args.unfreeze_last_blocks,
    ).to(device)
    criterion = SegmentationCriterion(IGNORE_INDEX, dice_weight=args.dice_weight)
    optimizer_groups: list[dict[str, object]] = [{"params": model.decoder.parameters(), "lr": args.lr}]
    if args.unfreeze_last_blocks:
        backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
        optimizer_groups.append({"params": backbone_parameters, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_miou, global_step = -1.0, 0

    print(
        f"device={torch.cuda.get_device_name(0)} | train={len(train_loader.dataset)} | "
        f"val={len(val_loader.dataset)} | feature_layers={args.feature_layers} | dice_weight={args.dice_weight} | "
        f"unfreeze_last_blocks={args.unfreeze_last_blocks}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_metric, running_loss, seen = SegmentationMetrics(VOC_NUM_CLASSES, IGNORE_INDEX), 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch_index, (images, masks) in enumerate(progress):
            if args.limit_train_batches is not None and batch_index >= args.limit_train_batches:
                break
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=args.grad_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            train_metric.update(logits, masks)
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(seen, 1)
        train_metrics = train_metric.compute()
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, amp_enabled, args.limit_val_batches
        )
        # Record the rate actually used in this epoch, before the scheduler changes it.
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        for metric_name, label in (("miou", "mIoU"), ("pixel_accuracy", "pixel_accuracy"), ("mean_accuracy", "mean_class_accuracy"), ("fw_iou", "frequency_weighted_iou")):
            writer.add_scalars(label, {"train": train_metrics[metric_name], "val": val_metrics[metric_name]}, epoch)
        writer.add_scalar("learning_rate", learning_rate, epoch)
        for class_id, class_name in enumerate(VOC_CLASS_NAMES):
            writer.add_scalars(
                f"class_iou/{class_name}",
                {"train": train_metrics["class_iou"][class_id], "val": val_metrics["class_iou"][class_id]},
                epoch,
            )
        append_overall_metrics(args.output_dir / "metrics.csv", epoch, train_loss, val_loss, train_metrics, val_metrics, learning_rate)
        append_per_class_metrics(args.output_dir / "per_class_metrics.csv", epoch, "train", train_metrics)
        append_per_class_metrics(args.output_dir / "per_class_metrics.csv", epoch, "val", val_metrics)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, args.output_dir / "last.pth")
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save(checkpoint, args.output_dir / "best.pth")
            save_best_diagnostics(args.output_dir, epoch, val_metrics)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_mIoU={train_metrics['miou']:.4f} "
            f"val_loss={val_loss:.4f} val_mIoU={val_metrics['miou']:.4f} "
            f"val_pixel_acc={val_metrics['pixel_accuracy']:.4f} val_mAcc={val_metrics['mean_accuracy']:.4f} "
            f"val_FWIoU={val_metrics['fw_iou']:.4f} best={best_miou:.4f}"
        )
    writer.close()


if __name__ == "__main__":
    main()
