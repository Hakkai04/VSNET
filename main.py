import os
# ==========================================
# 1. 显卡与绘图后端设置 (必须在最前面)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 强制使用物理 GPU 1

import matplotlib
matplotlib.use('Agg') # 设置非交互式后端，防止 "main thread is not in main loop" 报错

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
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from monai.losses import DiceCELoss
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, RandFlipd, RandRotated, RandShiftIntensityd, EnsureTyped,
    AsDiscreted, SpatialPadd, Spacingd, 
    AsDiscrete 
)
from monai.data import CacheDataset, DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# 导入模型文件
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
    logger = logging.getLogger("VSNet")
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
    parser.add_argument("--name", default="exp", help="Save results to project/name")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset root")
    parser.add_argument("--dataset_json", type=str, default="dataset_thin.json", help="Dataset json file name")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Maximum number of epochs")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval")
    
    # [修改] Batch Size 增大到 8 (4090 24G 显存足够)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (volumes per batch)")
    
    parser.add_argument("--num_samples", type=int, default=4, help="Patches per volume (Paper uses 4)")
    
    # [修改] 学习率降低到 1e-4，防止 Loss 震荡不收敛
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patch_size", type=tuple, default=(96, 96, 96), help="Input patch size")
    return parser.parse_args()

def get_transforms(patch_size, num_samples):
    train_transforms = Compose([
        LoadImaged(keys=["image", "label", "edge", "reg"]),
        EnsureChannelFirstd(keys=["image", "label", "edge", "reg"]),
        Orientationd(keys=["image", "label", "edge", "reg"], axcodes="RAS"),
        
        # === [新增] 体素重采样 (Voxel Resampling) ===
        # 将所有图像统一重采样到 (1.0mm, 1.0mm, 1.5mm)
        # mode: image/reg 用 bilinear (双线性插值)，label/edge 用 nearest (最近邻，防止产生小数标签)
        Spacingd(
            keys=["image", "label", "edge", "reg"],
            pixdim=(1.0, 1.0, 1.5), 
            mode=("bilinear", "nearest", "nearest", "bilinear")
        ),
        # ============================================
        
        # [修改] 窗宽优化：0-200 HU，过滤骨骼和空气，增强血管对比度
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0, 
            a_max=200.0, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        
        SpatialPadd(keys=["image", "label", "edge", "reg"], spatial_size=patch_size),
        RandCropByPosNegLabeld(
            keys=["image", "label", "edge", "reg"], label_key="label", spatial_size=patch_size,
            pos=1, neg=1, num_samples=num_samples, image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=["image", "label", "edge", "reg"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label", "edge", "reg"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label", "edge", "reg"], prob=0.5, spatial_axis=2),
        RandRotated(
            keys=["image", "label", "edge", "reg"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5, 
            mode=["bilinear", "nearest", "nearest", "bilinear"]
        ),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label", "edge", "reg"]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # === [新增] 验证集也要加！确保和训练集一致 ===
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.5),
            mode=("bilinear", "nearest")
        ),
        # ============================================
        # [修改] 保持与训练一致的窗宽
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=200.0, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train_transforms, val_transforms

def main():
    args = get_args()
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化目录
    save_dir = increment_path(Path(args.project) / args.name, mkdir=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(save_dir)
    writer = SummaryWriter(log_dir=str(save_dir))
    
    logger.info(f"🚀 Training started. Results saved to {save_dir}")
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
            "label": os.path.join(args.data_dir, item["label"]),
            "edge": os.path.join(args.data_dir, "preprocessed", "edge", os.path.basename(item["label"])),
            "reg": os.path.join(args.data_dir, "preprocessed", "reg", os.path.basename(item["label"]))
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

    # 3. 模型
    model = VSNet(in_channels=1, out_channels=3, img_size=96, training=True).to(device)
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {params_num / 1e6:.2f}M")

    # [修改] 暂时关闭辅助任务 Loss，让模型专心学分割
    loss_alpha, loss_beta, loss_gamma = 0.0, 0.0, 0.1
    
    # [新增] 类别权重：降低背景权重(0.1)，提高血管权重(1.0)
    # 这能解决模型倾向于预测全背景的问题
    class_weights = torch.tensor([0.1, 1.0, 1.0]).to(device)

    # [修改] 传入 weight 参数
    loss_seg_fn = DiceCELoss(
        softmax=True, to_onehot_y=True, include_background=False, batch=True,
        weight=class_weights 
    )
    loss_edge_fn = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False, batch=True)
    loss_reg_fn = MSELoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 使用新版 Scaler 写法
    scaler = torch.amp.GradScaler('cuda')

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    # 暂时不需要 HD95 Metric 实例，为了速度先不用它

    best_dice = -1
    best_epoch = -1

    # 4. 训练循环
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        model.training = True 
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epochs}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        running_losses = {"total": 0, "seg": 0, "reg": 0, "edge": 0, "deep": 0}
        steps = 0

        for batch_data in pbar:
            steps += 1
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)
            edges = batch_data["edge"].to(device, non_blocking=True)
            regs = batch_data["reg"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                seg_v, reg_out, seg_e, deep2, deep3 = model(inputs)

                l_se = loss_seg_fn(seg_v, labels)
                l_cr = loss_reg_fn(reg_out, regs)
                l_ec = loss_edge_fn(seg_e, edges)

                target_deep2 = F.interpolate(labels.float(), size=deep2.shape[2:], mode='nearest')
                target_deep3 = F.interpolate(labels.float(), size=deep3.shape[2:], mode='nearest')
                l_deep = loss_seg_fn(deep2, target_deep2) + loss_seg_fn(deep3, target_deep3)

                loss = l_se + (loss_alpha * l_cr) + (loss_beta * l_ec) + (loss_gamma * l_deep)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_losses["total"] += loss.item()
            running_losses["seg"] += l_se.item()
            running_losses["reg"] += l_cr.item()
            running_losses["edge"] += l_ec.item()
            running_losses["deep"] += l_deep.item()

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Seg": f"{l_se.item():.4f}", 
                "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        avg_loss = {k: v / steps for k, v in running_losses.items()}
        writer.add_scalar("Train/Total_Loss", avg_loss["total"], epoch)
        writer.add_scalar("Train/Seg_Loss", avg_loss["seg"], epoch)
        
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss["total"],
        }
        torch.save(ckpt, weights_dir / "last.pth")

        # 5. 验证循环
        if epoch % args.val_interval == 0:
            model.eval()
            model.training = False 
            
            vis_dir = save_dir / "vis_check"
            vis_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
                for i, val_data in enumerate(val_loader_tqdm):
                    val_inputs = val_data["image"].to(device, non_blocking=True)
                    val_labels = val_data["label"].to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda'):
                        val_outputs = sliding_window_inference(
                            inputs=val_inputs, roi_size=args.patch_size, sw_batch_size=4, predictor=model
                        )
                    
                    # === 可视化检查 (只看第一个 Batch) ===
                    if i == 0:
                        unique_labels = torch.unique(val_labels).cpu().numpy()
                        # logger.info(f"\n[Check] Epoch {epoch} Label Values: {unique_labels}")
                        
                        slice_idx = val_inputs.shape[-1] // 2
                        img_show = val_inputs[0, 0, :, :, slice_idx].cpu().numpy()
                        lbl_show = val_labels[0, 0, :, :, slice_idx].cpu().numpy()
                        pred_show = torch.argmax(val_outputs, dim=1)[0, :, :, slice_idx].detach().cpu().numpy()

                        plt.figure(figsize=(12, 4), dpi=100)
                        
                        plt.subplot(1, 3, 1)
                        plt.imshow(img_show, cmap="gray")
                        plt.title(f"Image\nMin:{img_show.min():.1f}, Max:{img_show.max():.1f}")
                        plt.axis('off')

                        plt.subplot(1, 3, 2)
                        plt.imshow(lbl_show, cmap="jet", interpolation='nearest') 
                        plt.title("Label (GT)")
                        plt.axis('off')

                        plt.subplot(1, 3, 3)
                        plt.imshow(pred_show, cmap="jet", interpolation='nearest')
                        plt.title(f"Prediction Epoch {epoch}")
                        plt.axis('off')
                        
                        save_path = vis_dir / f"epoch_{epoch}_check.png"
                        plt.savefig(save_path)
                        plt.close()
                    # =================================

                    # 计算 Dice
                    val_outputs_onehot = AsDiscrete(argmax=True, to_onehot=3, dim=1)(val_outputs)
                    val_labels_onehot = AsDiscrete(to_onehot=3, dim=1)(val_labels)

                    dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)
                    
                    # [修改] 暂时注释掉 HD95 以加速验证
                    # hd95_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

                dice_score = dice_metric.aggregate()
                
                dice_hv = dice_score[0].item()
                dice_pv = dice_score[1].item()
                mean_dice = dice_score.mean().item()
                
                # 暂时设为 0.0
                mean_hd95 = 0.0 

                dice_metric.reset()

                log_msg = (f"\nEpoch {epoch} | Val Mean Dice: {mean_dice:.4f} | "
                           f"HV: {dice_hv:.4f} | PV: {dice_pv:.4f} | HD95: {mean_hd95:.2f}")
                logger.info(log_msg)
                
                writer.add_scalar("Val/Mean_Dice", mean_dice, epoch)
                writer.add_scalar("Val/HV_Dice", dice_hv, epoch)
                writer.add_scalar("Val/PV_Dice", dice_pv, epoch)

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_epoch = epoch
                    torch.save(ckpt, weights_dir / "best.pth")
                    logger.info(f"🏆 New Best Model saved! (Dice: {best_dice:.4f})")

    logger.info(f"\nTraining completed. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    logger.info(f"Final weights saved to: {weights_dir}")
    writer.close()

if __name__ == "__main__":
    main()