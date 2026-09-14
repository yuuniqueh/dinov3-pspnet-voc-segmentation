# DINOv3 ViT-S/16 + PSPNet on PASCAL VOC 2012

本课程项目使用官方 DINOv3 最小 ViT-S/16 预训练视觉骨干和自行实现的 PSPNet 分割头，在 PASCAL VOC 2012 语义分割任务上训练与验证。

## 最终结果

最终模型在 VOC2012 验证集上达到：

| Metric | Result |
| --- | ---: |
| mIoU | **86.77%** |
| Pixel Accuracy | 97.24% |
| Mean Class Accuracy | 92.97% |
| Frequency-Weighted IoU | 94.87% |

最佳 checkpoint 出现在第 42 个 epoch。该结果在本项目的验证协议下获得：验证图像与对应标签均使用双线性/最近邻插值缩放到 `640 x 640`，再计算指标。因此，不应将其直接与采用原始分辨率、滑窗推理或测试服务器评估的论文结果横向比较。

## Final configuration

```text
Backbone:       DINOv3 ViT-S/16, LVD-1689M pretrained checkpoint
Decoder:        PSPNet with PPM bins (1, 2, 3, 6)
Dataset:        PASCAL VOC 2012 semantic segmentation
Split:          1,464 train images / 1,449 validation images
Input size:     640 x 640
Loss:           CrossEntropy + 0.2 x foreground soft Dice loss
Optimizer:      AdamW
Decoder LR:     5e-4
Backbone LR:    1e-5
Weight decay:   1e-4
Fine-tuning:    final two DINOv3 Transformer blocks and final LayerNorm
Epochs:         60 with cosine learning-rate decay
Gradient clip:  1.0
```

## Ablation results

| Experiment | Main change | Best mIoU |
| --- | --- | ---: |
| Baseline | 512 input, frozen DINOv3, CrossEntropy, 40 epochs | 80.51% |
| Learning-rate tuning | `lr=5e-4` | 80.78% |
| Higher resolution | 640 input, CrossEntropy | 81.82% |
| Dice loss | 640 input, CrossEntropy + 0.2 Dice | 83.13% |
| Longer schedule | 640 input, Dice, 60 epochs | 83.80% |
| Final model | Fine-tune final two DINOv3 blocks | **86.77%** |

## Architecture

```text
RGB image
  -> DINOv3 ViT-S/16
  -> dense patch feature map [B, 384, H/16, W/16]
  -> 1x1 channel reduction
  -> PSPNet pyramid pooling module: 1, 2, 3, 6 bins
  -> 21-class logits
  -> bilinear upsampling to the input size
```

During the final experiment, the first ten Transformer blocks remain frozen. The final two blocks and the final LayerNorm use the smaller backbone learning rate.

## Project layout

```text
src/
  data.py       VOC dataset, paired augmentations, and data loaders
  losses.py     CrossEntropy and foreground soft Dice loss
  metrics.py    mIoU, Pixel Accuracy, mAcc, FWIoU, and confusion matrix
  model.py      DINOv3 wrapper and PSPNet decoder
train.py        training, validation, logging, and checkpoint entry point
```

The following local directories are intentionally excluded from Git: `data/`, `checkpoints/`, `runs/`, `.cache/`, and `third_party/`.

## Setup

Tested environment:

```text
Python 3.11.16
PyTorch 2.8.0 + CUDA 12.8
torchvision 0.23.0 + CUDA 12.8
GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
```

Install PyTorch and torchvision with CUDA support first, then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

Download the official DINOv3 source repository into `third_party/dinov3-main/` and place the approved ViT-S/16 checkpoint at:

```text
checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

Place VOC2012 in the standard torchvision layout:

```text
data/VOCdevkit/VOC2012/
```

## Run the final experiment

```powershell
D:\anaconda\envs\dinov3_psp\python.exe train.py `
  --image-size 640 `
  --epochs 60 `
  --lr 0.0005 `
  --backbone-lr 0.00001 `
  --unfreeze-last-blocks 2 `
  --grad-clip-norm 1.0 `
  --weight-decay 0.0001 `
  --dice-weight 0.2 `
  --output-dir runs\input640_dice02_ftlast2_60ep
```

Each run writes `metrics.csv`, per-class metrics, TensorBoard logs, a best checkpoint, and a confusion matrix to its selected output directory.

## References

- [DINOv3 official repository](https://github.com/facebookresearch/dinov3)
- [Pyramid Scene Parsing Network (PSPNet)](https://arxiv.org/abs/1612.01105)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
