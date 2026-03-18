import os
# ==========================================
# 1. 显卡与绘图后端设置 (必须在最前面)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 根据需要修改 GPU 编号

import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端

import json
import torch
import numpy as np
import argparse
import random
import time
import logging
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# MONAI 组件
from monai.losses import DiceCELoss
# [修改 1] 导入 AttentionUnet
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, RandFlipd, RandRotated, RandShiftIntensityd, EnsureTyped,
    AsDiscreted, SpatialPadd, Spacingd, AsDiscrete
)
from monai.data import CacheDataset, DataLoader
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# -------------------------------------------------------------------------
# 工具函数 (保持与 VSNet 一致)
# -------------------------------------------------------------------------
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path

def get_logger(save_dir):
    # [修改日志名]
    logger = logging.getLogger("AttentionUNet_Training")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def get_args():
    parser = argparse.ArgumentParser(description="Attention UNet Training Script")
    parser.add_argument("--project", default="runs/train", help="Save results to project/name")
    
    # [修改 2] 修改默认 name，避免覆盖 Baseline 结果
    parser.add_argument("--name", default="attention_unet", help="Save results to project/name")
    
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset root")
    parser.add_argument("--dataset_json", type=str, default="dataset_thin.json", help="Dataset json file name")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Maximum number of epochs")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval")
    
    # 梯度累积设置
    parser.add_argument("--batch_size", type=int, default=4, help="Physical batch size per GPU")
    parser.add_argument("--target_batch_size", type=int, default=16, help="Target effective batch size")
    
    parser.add_argument("--num_samples", type=int, default=4, help="Patches per volume")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patch_size", type=tuple, default=(96, 96, 96), help="Input patch size")
    return parser.parse_args()

def get_transforms(patch_size, num_samples):
    keys = ["image", "label"]
    
    train_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=(1.0, 1.0, 1.0), 
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0, a_max=200.0,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        SpatialPadd(keys=keys, spatial_size=patch_size),
        RandCropByPosNegLabeld(
            keys=keys, label_key="label", spatial_size=patch_size,
            pos=1, neg=1, num_samples=num_samples, image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandRotated(
            keys=keys, range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5, 
            mode=["bilinear", "nearest"]
        ),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=keys),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=200.0, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=keys),
    ])
    return train_transforms, val_transforms

def main():
    args = get_args()
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化目录 (基于新的 args.name = "attention_unet")
    save_dir = increment_path(Path(args.project) / args.name, mkdir=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(save_dir)
    writer = SummaryWriter(log_dir=str(save_dir))
    
    logger.info(f"🚀 Attention UNet Training started. Results saved to {save_dir}")
    logger.info(f"Using device: {device}")
    
    with open(save_dir / 'opt.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 2. 数据准备
    json_path = os.path.join(args.data_dir, args.dataset_json)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")
        
    with open(json_path, "r") as f:
        data_json = json.load(f)

    train_files = []
    for item in data_json["training"]:
        train_files.append({
            "image": os.path.join(args.data_dir, item["image"]),
            "label": os.path.join(args.data_dir, item["label"])
        })

    random.seed(42)
    random.shuffle(train_files)
    split_idx = int(len(train_files) * 0.8)
    val_files = train_files[split_idx:]
    train_files = train_files[:split_idx]

    logger.info(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}")

    train_transforms, val_transforms = get_transforms(args.patch_size, args.num_samples)

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # 3. 模型构建 (使用 AttentionUnet)
    # [修改 3] 替换为 AttentionUnet
    # 注意：AttentionUnet 通常不包含 residual units，它是标准的 U-Net 加上 Attention Gates
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,             # 背景, HV, PV
        channels=(32, 64, 128, 256, 512), # 保持大容量配置以对齐复现参数量
        strides=(2, 2, 2, 2),
    ).to(device)
    
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"AttentionUnet Model parameters: {params_num / 1e6:.2f}M")

    # [核心] Loss 定义
    loss_seg_fn = DiceCELoss(
        softmax=True, 
        to_onehot_y=True, 
        include_background=False, 
        batch=True,
    )
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

    best_dice = -1
    best_epoch = -1

    # 计算累积步数 (梯度累积逻辑保持不变)
    accumulation_steps = args.target_batch_size // args.batch_size
    if args.target_batch_size % args.batch_size != 0:
        logger.warning(f"⚠️ Target batch size {args.target_batch_size} implies fractional accumulation with batch size {args.batch_size}.")
    
    logger.info(f"⚡ Gradient Accumulation enabled: {accumulation_steps} steps (Effective Batch Size: {args.target_batch_size})")

    # 4. 训练循环
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epochs}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        running_loss = 0
        steps = 0
        
        optimizer.zero_grad()

        for i, batch_data in enumerate(pbar):
            steps += 1
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_seg_fn(outputs, labels)
                
                # 梯度累积：Loss 除以步数
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # 还原 Loss 数值用于显示
            current_loss = loss.item() * accumulation_steps
            running_loss += current_loss

            pbar.set_postfix({
                "Loss": f"{current_loss:.4f}"
            })

        if (len(train_loader) % accumulation_steps) != 0:
             scaler.step(optimizer)
             scaler.update()
             optimizer.zero_grad()

        avg_loss = running_loss / steps
        writer.add_scalar("Train/Total_Loss", avg_loss, epoch)
        
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(ckpt, weights_dir / "last.pth")

        # 5. 验证循环
        if epoch % args.val_interval == 0:
            model.eval()
            
            vis_dir = save_dir / "vis_check"
            vis_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
                for i, val_data in enumerate(val_loader_tqdm):
                    val_inputs = val_data["image"].to(device, non_blocking=True)
                    val_labels = val_data["label"].to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda'):
                        val_outputs = sliding_window_inference(
                            inputs=val_inputs, 
                            roi_size=args.patch_size, 
                            sw_batch_size=4, 
                            predictor=model,
                            overlap=0.5
                        )
                    
                    if i == 0:
                        slice_idx = val_inputs.shape[-1] // 2
                        img_show = val_inputs[0, 0, :, :, slice_idx].cpu().numpy()
                        lbl_show = val_labels[0, 0, :, :, slice_idx].cpu().numpy()
                        pred_show = torch.argmax(val_outputs, dim=1)[0, :, :, slice_idx].detach().cpu().numpy()

                        plt.figure(figsize=(12, 4), dpi=100)
                        plt.subplot(1, 3, 1); plt.imshow(img_show, cmap="gray"); plt.title("Image"); plt.axis('off')
                        plt.subplot(1, 3, 2); plt.imshow(lbl_show, cmap="jet", interpolation='nearest'); plt.title("Label"); plt.axis('off')
                        plt.subplot(1, 3, 3); plt.imshow(pred_show, cmap="jet", interpolation='nearest'); plt.title(f"Pred Epoch {epoch}"); plt.axis('off')
                        plt.savefig(vis_dir / f"epoch_{epoch}_check.png")
                        plt.close()

                    val_outputs_onehot = AsDiscrete(argmax=True, to_onehot=3, dim=1)(val_outputs)
                    val_labels_onehot = AsDiscrete(to_onehot=3, dim=1)(val_labels)

                    dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

                dice_score = dice_metric.aggregate()
                dice_hv = dice_score[0].item()
                dice_pv = dice_score[1].item()
                mean_dice = dice_score.mean().item()
                dice_metric.reset()

                log_msg = (f"\nEpoch {epoch} | Val Mean Dice: {mean_dice:.4f} | "
                           f"HV: {dice_hv:.4f} | PV: {dice_pv:.4f}")
                logger.info(log_msg)
                
                writer.add_scalar("Val/Mean_Dice", mean_dice, epoch)

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_epoch = epoch
                    torch.save(ckpt, weights_dir / "best.pth")
                    logger.info(f"🏆 New Best Model saved! (Dice: {best_dice:.4f})")

    logger.info(f"\nTraining completed. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    writer.close()

if __name__ == "__main__":
    main()