import os
import json
import torch
import numpy as np
import argparse
from torch.optim import AdamW
from torch.nn import MSELoss, Interpolate
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    RandShiftIntensityd,
    EnsureTyped,
    AsDiscreted,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# 导入模型文件
from VSNet import VSNet

def get_args():
    parser = argparse.ArgumentParser(description="VSNet Training Script")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset root")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Maximum number of epochs")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (volumes per batch)")
    parser.add_argument("--num_samples", type=int, default=4, help="Patches per volume (Paper uses 4)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patch_size", type=tuple, default=(96, 96, 96), help="Input patch size")
    return parser.parse_args()

def get_transforms(patch_size, num_samples):
    """
    数据增强与预处理，严格参照论文 Implementation Details 。
    """
    train_transforms = Compose(
        [
            # 读取 Image, Label, Edge, Reg 四种数据
            LoadImaged(keys=["image", "label", "edge", "reg"]),
            EnsureChannelFirstd(keys=["image", "label", "edge", "reg"]),
            Orientationd(keys=["image", "label", "edge", "reg"], axcodes="RAS"),
            
            # 强度归一化 (CT 常用处理)
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
            ),
            
            # 随机裁剪：保证裁剪区域包含前景 (Label > 0)
            # Paper: "four patches with a size of 96x96x96" [cite: 395]
            RandCropByPosNegLabeld(
                keys=["image", "label", "edge", "reg"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            
            # 数据增强：Flip, Rotation, Intensity Shift [cite: 396]
            RandFlipd(keys=["image", "label", "edge", "reg"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label", "edge", "reg"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label", "edge", "reg"], prob=0.5, spatial_axis=2),
            RandRotated(
                keys=["image", "label", "edge", "reg"],
                range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5, mode=["bilinear", "nearest", "nearest", "bilinear"]
            ),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            
            EnsureTyped(keys=["image", "label", "edge", "reg"]),
        ]
    )
    
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms

def main():
    args = get_args()
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备数据
    json_path = os.path.join(args.data_dir, "dataset.json")
    with open(json_path, "r") as f:
        data_json = json.load(f)

    # 路径映射：将相对路径转换为绝对路径
    train_files = []
    for item in data_json["training"]:
        train_files.append({
            "image": os.path.join(args.data_dir, item["image"]),
            "label": os.path.join(args.data_dir, item["label"]),
            "edge": os.path.join(args.data_dir, "preprocessed", "edge", os.path.basename(item["label"])),
            "reg": os.path.join(args.data_dir, "preprocessed", "reg", os.path.basename(item["label"]))
        })
        
    # 简单的划分，实际建议在dataset.json中定义好 folds
    val_files = train_files[-int(len(train_files)*0.2):] 
    train_files = train_files[:-int(len(train_files)*0.2)]

    train_transforms, val_transforms = get_transforms(args.patch_size, args.num_samples)

    # 使用 CacheDataset 加速读取
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=4)
    # Batch size 为 volumes 数量，由于每个 volume 采样 num_samples 个 patch，
    # 这里的实际 batch size = args.batch_size * args.num_samples
    # Paper uses 4 volumes * 4 patches = 16 patches total.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # 2. 定义模型
    # out_channels=3 (BG, HV, PV)
    model = VSNet(
        in_channels=1, 
        out_channels=3, 
        img_size=96, 
        training=True
    ).to(device)

    # 3. 定义 Loss 和 优化器
    # Loss weights from Paper: alpha=1, beta=1, gamma=0.1
    loss_alpha = 1.0 # Centerline Regression Weight
    loss_beta = 1.0  # Edge Segmentation Weight
    loss_gamma = 0.1 # Deep Supervision Weight

    # Segmentation Loss: Dice + CrossEntropy [cite: 346]
    # 用于 Main task, Edge task, Deep Supervision
    loss_seg_fn = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False, batch=True)
    # Edge 需要特定的 DiceCE, 因为 edge label 可能是 0/1 单通道，需要转 onehot
    loss_edge_fn = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False, batch=True)
    
    # Regression Loss: MSE [cite: 350]
    loss_reg_fn = MSELoss()

    # Optimizer: AdamW 
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    best_metric = -1
    best_metric_epoch = -1

    # 4. 训练循环
    for epoch in range(args.max_epochs):
        model.train()
        model.training = True # 确保 VSNet forward 返回 tuple
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            # 将 (Batch, Samples, C, H, W) -> (Batch*Samples, C, H, W)
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            edges = batch_data["edge"].to(device)
            regs = batch_data["reg"].to(device)

            # Flatten batch and samples dimensions
            inputs = inputs.view(-1, *inputs.shape[2:])
            labels = labels.view(-1, *labels.shape[2:])
            edges = edges.view(-1, *edges.shape[2:])
            regs = regs.view(-1, *regs.shape[2:])

            optimizer.zero_grad()

            # Forward Pass
            # returns: seg_v (Main), reg (Center), seg_e (Edge), deep2, deep3
            seg_v, reg_out, seg_e, deep2, deep3 = model(inputs)

            # --- Calculate Losses ---
            
            # 1. Main Segmentation Loss (HV/PV)
            # labels range: 0, 1, 2. seg_v output: 3 channels
            l_se = loss_seg_fn(seg_v, labels)

            # 2. Auxiliary Task 1: Centerline Regression (MSE) [cite: 350]
            # reg_out is Sigmoid activated (0-1), regs is ground truth distance map (0-1)
            l_cr = loss_reg_fn(reg_out, regs)

            # 3. Auxiliary Task 2: Edge Segmentation [cite: 330]
            # seg_e output: 2 channels (Softmax). edges label: 1 channel (0/1)
            l_ec = loss_edge_fn(seg_e, edges)

            # 4. Deep Supervision 
            # Deep outputs are typically smaller feature maps.
            # We must downsample GT labels to match deep2 and deep3 spatial dimensions.
            # deep2 and deep3 dimensions vary based on architecture, we assume they need matching.
            
            target_deep2 = F.interpolate(labels.float(), size=deep2.shape[2:], mode='nearest')
            target_deep3 = F.interpolate(labels.float(), size=deep3.shape[2:], mode='nearest')
            
            l_deep2 = loss_seg_fn(deep2, target_deep2)
            l_deep3 = loss_seg_fn(deep3, target_deep3)
            l_deep = l_deep2 + l_deep3

            # Total Loss [cite: 365]
            loss = l_se + (loss_alpha * l_cr) + (loss_beta * l_ec) + (loss_gamma * l_deep)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f} "
                  f"(SE: {l_se.item():.4f}, CR: {l_cr.item():.4f}, EC: {l_ec.item():.4f}, DS: {l_deep.item():.4f})", end="\r")

        epoch_loss /= step
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}, Average Loss: {epoch_loss:.4f}")

        # 5. 验证循环 (Every 5 epochs)
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            model.training = False # VSNet forward returns only seg_v
            
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    # Sliding Window Inference for 3D volumes
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs, 
                        roi_size=args.patch_size, 
                        sw_batch_size=4, 
                        predictor=model
                    )
                    
                    # Discretize outputs for metric calculation
                    val_outputs = AsDiscreted(argmax=True)(val_outputs)
                    
                    # Compute Metrics
                    # DiceMetric and Hausdorff expect (B, C, H, W, D) one-hot or channel-first labels if specified
                    # We convert labels to one-hot for metrics: 3 channels (BG, HV, PV)
                    val_labels_onehot = AsDiscreted(to_onehot=3)(val_labels)
                    val_outputs_onehot = AsDiscreted(to_onehot=3)(val_outputs)

                    dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)
                    hd95_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

                # Aggregate metrics
                # Metric returns tensor of shape (Num_Classes - 1,) if include_background=False
                # So index 0 is HV, index 1 is PV
                dice_score = dice_metric.aggregate()
                hd95_score = hd95_metric.aggregate()
                
                dice_hv = dice_score[0].item()
                dice_pv = dice_score[1].item()
                mean_dice = dice_score.mean().item()
                
                print(f"Validation Epoch {epoch + 1}:")
                print(f"  Mean Dice: {mean_dice:.4f}")
                print(f"  Hepatic Vein (HV) Dice: {dice_hv:.4f}")
                print(f"  Portal Vein (PV) Dice: {dice_pv:.4f}")
                print(f"  Mean HD95: {hd95_score.mean().item():.4f}")

                # Save best model
                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join("best_metric_model.pth"))
                    print("  Saved new best model!")

                dice_metric.reset()
                hd95_metric.reset()

    print(f"Training completed. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")

if __name__ == "__main__":
    main()