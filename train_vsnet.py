import os
# ==========================================
# 1. 显卡与绘图后端设置 (必须在最前面)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 根据需要修改 GPU 编号

import matplotlib
matplotlib.use('Agg')

import json
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import logging
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F

# MONAI 组件
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, RandFlipd, RandRotated, RandShiftIntensityd, EnsureTyped,
    AsDiscreted, SpatialPadd, Spacingd, AsDiscrete
)
from monai.data import CacheDataset, DataLoader
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# 导入修复后的 VSNet
from VSNet import VSNet

# -------------------------------------------------------------------------
# 工具函数
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
    logger = logging.getLogger("VSNet_Training")
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
    parser = argparse.ArgumentParser(description="VSNet Training Script")
    parser.add_argument("--project", default="runs/train", help="Save results to project/name")
    parser.add_argument("--name", default="vsnet", help="Save results to project/name")
    
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset root")
    parser.add_argument("--dataset_json", type=str, default="dataset_full_thin.json", help="Dataset json file name")
    
    parser.add_argument("--max_epochs", type=int, default=2000, help="Maximum number of epochs")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval")
    
    parser.add_argument("--batch_size", type=int, default=4, help="Physical batch size per GPU")
    parser.add_argument("--target_batch_size", type=int, default=16, help="Target effective batch size (Paper: 16)")
    parser.add_argument("--num_samples", type=int, default=4, help="Patches per volume (Paper: 4)")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (Paper: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (Paper: 1e-5)")
    parser.add_argument("--patch_size", type=tuple, default=(96, 96, 96), help="Input patch size (Paper: 96x96x96)")
    
    # Loss 权重 (论文 4.5 节最优配置)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for centerline regression (L_cr)")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for edge segmentation (L_ec)")
    parser.add_argument("--gamma", type=float, default=0.1, help="Weight for deep supervision (L_deep)")
    
    return parser.parse_args()

def get_transforms(patch_size, num_samples):
    # VSNet 需要 4 个数据源
    keys = ["image", "label", "edge", "reg"]
    # 空间插值模式：连续值(image, reg)用bilinear，离散值(label, edge)用nearest
    modes = ("bilinear", "nearest", "nearest", "bilinear")
    
    train_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=modes),
        
        # 只对 CT 图像进行窗宽窗位截断归一化
        ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=200.0, b_min=0.0, b_max=1.0, clip=True
        ),
        
        SpatialPadd(keys=keys, spatial_size=patch_size),
        
        # 依据主要任务的 label 进行裁剪
        RandCropByPosNegLabeld(
            keys=keys, label_key="label", spatial_size=patch_size,
            pos=1, neg=1, num_samples=num_samples, image_key="image", image_threshold=0,
        ),
        
        # 空间几何增强 (严格保证 4 个特征对齐)
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandRotated(keys=keys, range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5, mode=modes),
        
        # 仅对图像进行强度漂移
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=keys, track_meta=False),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]), # 验证时只需要图像和标签
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=200.0, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ])
    return train_transforms, val_transforms

def main():
    args = get_args()
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 目录初始化
    save_dir = increment_path(Path(args.project) / args.name, mkdir=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(save_dir)
    writer = SummaryWriter(log_dir=str(save_dir))
    logger.info(f"🚀 VSNet Training started. Results saved to {save_dir}")
    
    with open(save_dir / 'opt.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 2. 从 JSON 文件解析数据集
    json_path = os.path.join(args.data_dir, args.dataset_json)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")
        
    with open(json_path, "r") as f:
        data_json = json.load(f)

    def parse_files(file_list):
        parsed = []
        for item in file_list:
            # os.path.normpath 可以清理路径中的 "./" 以确保路径干净
            parsed.append({
                "image": os.path.normpath(os.path.join(args.data_dir, item["image"])),
                "label": os.path.normpath(os.path.join(args.data_dir, item["label"])),
                "edge": os.path.normpath(os.path.join(args.data_dir, item["edge"])),
                "reg": os.path.normpath(os.path.join(args.data_dir, item["centerline"]))  # 将 centerline 映射到 reg
            })
        return parsed

    train_files = parse_files(data_json.get("training", []))
    val_files = parse_files(data_json.get("validation", []))

    # 如果 JSON 中 validation 列表为空，则从 training 列表切分 80/20
    if len(val_files) == 0:
        logger.info("Validation list in JSON is empty. Performing automatic 80/20 split.")
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

    # 3. 模型构建 (论文：参数量约 38.1M)
    model = VSNet(
        in_channels=1, 
        out_channels=3, 
        img_size=args.patch_size[0],
        training=True
    ).to(device)
    
    # 4. 损失函数定义
    criterion_seg = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False, batch=True)
    criterion_edge = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False, batch=True)
    criterion_reg = nn.MSELoss()
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    best_dice = -1
    best_epoch = -1
    accumulation_steps = args.target_batch_size // args.batch_size
    logger.info(f"⚡ Gradient Accumulation: {accumulation_steps} steps (Effective BS: {args.target_batch_size})")

    # 5. 训练循环
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epochs}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        running_loss = 0.0
        optimizer.zero_grad()

        for i, batch_data in enumerate(pbar):
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)
            edges = batch_data["edge"].to(device, non_blocking=True)
            regs = batch_data["reg"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                # VSNet 返回: 主分割, 中心线回归, 边缘分割, 深监督2, 深监督3
                seg_v, reg, seg_e, deep2, deep3 = model(inputs)
                
                # ----------------- 替换之前的代码 -----------------
                L_se = criterion_seg(seg_v, labels)
                L_cr = criterion_reg(reg, regs)
                L_ec = criterion_edge(seg_e, edges)
                
                # [新增修复] 按照论文要求，将 labels 缩小到与 deep2 和 deep3 相同的尺寸
                # 由于 labels 包含的是类别索引 (0, 1, 2)，必须使用 mode='nearest' 防止产生小数
                labels_d2 = F.interpolate(labels.float(), size=deep2.shape[2:], mode='nearest').to(labels.dtype)
                labels_d3 = F.interpolate(labels.float(), size=deep3.shape[2:], mode='nearest').to(labels.dtype)
                
                # 使用缩放后的标签计算深监督 Loss
                L_deep = (criterion_seg(deep2, labels_d2) + criterion_seg(deep3, labels_d3)) / 2.0
                
                loss = L_se + args.alpha * L_cr + args.beta * L_ec + args.gamma * L_deep
                loss = loss / accumulation_steps
                # ------------------------------------------------

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            pbar.set_postfix({"Loss": f"{loss.item() * accumulation_steps:.4f}"})

        if (len(train_loader) % accumulation_steps) != 0:
             scaler.step(optimizer)
             scaler.update()
             optimizer.zero_grad()

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Train/Total_Loss", avg_loss, epoch)
        
        # 6. 验证循环
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
                            inputs=val_inputs, roi_size=args.patch_size, 
                            sw_batch_size=4, predictor=model, overlap=0.5
                        )
                    
                    if i == 0:
                        slice_idx = val_inputs.shape[-1] // 2
                        img_show = val_inputs[0, 0, :, :, slice_idx].cpu().numpy()
                        lbl_show = val_labels[0, 0, :, :, slice_idx].cpu().numpy()
                        pred_show = torch.argmax(val_outputs, dim=1)[0, :, :, slice_idx].cpu().numpy()

                        plt.figure(figsize=(12, 4), dpi=100)
                        plt.subplot(1, 3, 1); plt.imshow(img_show, cmap="gray"); plt.title("Image"); plt.axis('off')
                        plt.subplot(1, 3, 2); plt.imshow(lbl_show, cmap="jet", interpolation='nearest'); plt.title("Label"); plt.axis('off')
                        plt.subplot(1, 3, 3); plt.imshow(pred_show, cmap="jet", interpolation='nearest'); plt.title(f"Pred E{epoch}"); plt.axis('off')
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

                logger.info(f"\nEpoch {epoch} | Val Mean Dice: {mean_dice:.4f} | HV: {dice_hv:.4f} | PV: {dice_pv:.4f}")
                writer.add_scalar("Val/Mean_Dice", mean_dice, epoch)

                ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                torch.save(ckpt, weights_dir / "last.pth")
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_epoch = epoch
                    torch.save(ckpt, weights_dir / "best.pth")
                    logger.info(f"🏆 New Best Model saved! (Dice: {best_dice:.4f})")

    logger.info(f"\n🎉 Training completed. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    writer.close()

if __name__ == "__main__":
    main()