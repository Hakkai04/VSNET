import os
import glob
import re
import csv
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# MONAI 组件
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
# 注意：移除了 HausdorffDistanceMetric，因为它太慢且在训练期容易导致内存溢出
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandCropByPosNegLabeld, RandRotate90d, RandFlipd, ToTensord,
    EnsureTyped, SpatialPadd
)
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference

# 导入 VSNet 模型
from VSNet import VSNet

# ================= 1. YOLO风格实验管理类 =================
class ExperimentManager:
    def __init__(self, project_root="runs/train", name="exp"):
        self.project_root = project_root
        self.name = name
        self.save_dir = self._increment_path()
        self.weights_dir = os.path.join(self.save_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        
        self.log_path = os.path.join(self.save_dir, "results.csv")
        # [修改] 移除了 95hd_mm 列，只保留核心指标
        self.header = [
            "epoch", 
            "train_loss", "val_loss", 
            "dice_mean", "dice_hv", "dice_pv", 
            "precision", "sensitivity"
        ]
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            
        self._setup_logger()
        print(f"🚀 Training started. Results saved to {self.save_dir}")

    def _increment_path(self):
        if not os.path.exists(self.project_root):
            return os.path.join(self.project_root, f"{self.name}1")
        
        dirs = glob.glob(f"{self.project_root}/{self.name}*")
        nums = []
        for d in dirs:
            try:
                basename = os.path.basename(d)
                n = int(re.search(f"{self.name}(\d+)", basename).group(1))
                nums.append(n)
            except:
                continue
        
        if not nums:
            return os.path.join(self.project_root, f"{self.name}1")
        
        new_num = max(nums) + 1
        return os.path.join(self.project_root, f"{self.name}{new_num}")

    def _setup_logger(self):
        logging.basicConfig(
            filename=os.path.join(self.save_dir, "train.log"),
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def log_results(self, data):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)
    
    def save_model(self, model, optimizer, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.weights_dir, "last.pth"))
        if is_best:
            torch.save(state, os.path.join(self.weights_dir, "best.pth"))

# ================= 2. 数据集构建 =================
def get_dataloader(batch_size=2, train_val_ratio=0.8):
    img_dir = "./dataset/imagesTr"
    lbl_dir = "./dataset/labelsTr"
    reg_dir = "./dataset/preprocessed/reg"
    edge_dir = "./dataset/preprocessed/edge"

    if not os.path.exists(reg_dir) or not os.path.exists(edge_dir):
        raise FileNotFoundError("预处理数据未找到，请先运行预处理脚本生成 reg 和 edge 数据！")

    files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    data_list = []

    for img_path in files:
        fname = os.path.basename(img_path)
        lbl_path = os.path.join(lbl_dir, fname)
        reg_path = os.path.join(reg_dir, fname)
        edge_path = os.path.join(edge_dir, fname)

        if os.path.exists(lbl_path) and os.path.exists(reg_path) and os.path.exists(edge_path):
            data_list.append({
                "image": img_path,
                "seg": lbl_path,
                "reg": reg_path,
                "edge": edge_path
            })
        else:
            print(f"Warning: Missing files for {fname}, skipping.")

    random.seed(42)
    random.shuffle(data_list)
    split_idx = int(len(data_list) * train_val_ratio)
    train_files = data_list[:split_idx]
    val_files = data_list[split_idx:]
    
    print(f"Dataset Split: Train={len(train_files)}, Val={len(val_files)}")

    train_transforms = Compose([
        LoadImaged(keys=["image", "seg", "reg", "edge"]),
        EnsureChannelFirstd(keys=["image", "seg", "reg", "edge"]),
        ScaleIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "seg", "reg", "edge"], spatial_size=(96, 96, 96)),
        RandCropByPosNegLabeld(
            keys=["image", "seg", "reg", "edge"],
            label_key="seg",
            spatial_size=(96, 96, 96),
            pos=1, neg=1, num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        RandRotate90d(keys=["image", "seg", "reg", "edge"], prob=0.5, spatial_axes=[0, 2]),
        RandFlipd(keys=["image", "seg", "reg", "edge"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "seg", "reg", "edge"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "seg", "reg", "edge"], prob=0.5, spatial_axis=2),
        EnsureTyped(keys=["image", "seg", "reg", "edge"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "seg"]),
        EnsureChannelFirstd(keys=["image", "seg"]),
        ScaleIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "seg"], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=["image", "seg"]),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # [关键优化] Windows下 num_workers 设为 0 避免多进程死锁和内存复制开销
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    return train_loader, val_loader

# ================= 3. 多任务损失函数 =================
class VSNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0 
        self.beta = 1.0
        self.gamma = 0.1
        
        self.dice_ce_loss = DiceCELoss(softmax=False, to_onehot_y=True, lambda_dice=1.0, lambda_ce=1.0)
        self.edge_loss = DiceCELoss(softmax=False, to_onehot_y=True, lambda_dice=1.0, lambda_ce=1.0)
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        pred_seg, pred_reg, pred_edge, pred_deep2, pred_deep3 = outputs
        gt_seg, gt_reg, gt_edge = targets['seg'], targets['reg'], targets['edge']

        l_se = self.dice_ce_loss(pred_seg, gt_seg)
        l_cr = self.mse_loss(pred_reg, gt_reg)
        l_ec = self.edge_loss(pred_edge, gt_edge)

        target_deep2 = F.interpolate(gt_seg.float(), size=pred_deep2.shape[2:], mode='nearest')
        target_deep3 = F.interpolate(gt_seg.float(), size=pred_deep3.shape[2:], mode='nearest')
        
        l_deep2 = self.dice_ce_loss(pred_deep2, target_deep2)
        l_deep3 = self.dice_ce_loss(pred_deep3, target_deep3)
        l_deep = l_deep2 + l_deep3

        total_loss = l_se + (self.alpha * l_cr) + (self.beta * l_ec) + (self.gamma * l_deep)
        return total_loss, l_se, l_cr, l_ec

# ================= 4. 训练主流程 =================
def main():
    # [优化] 自动选择设备并打印
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    BATCH_SIZE = 2
    EPOCHS = 300
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    VAL_INTERVAL = 10
    ROI_SIZE = (96, 96, 96)
    
    set_determinism(seed=42)

    exp = ExperimentManager()
    train_loader, val_loader = get_dataloader(batch_size=BATCH_SIZE)

    model = VSNet(
        in_channels=1, 
        out_channels=3, 
        img_size=96, 
        training=True
    ).to(device)

    loss_function = VSNetLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # [新增] 梯度缩放器，用于混合精度训练
    scaler = torch.amp.GradScaler('cuda')

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
    conf_metric = ConfusionMatrixMetric(include_background=False, metric_name=["precision", "sensitivity"], reduction="mean")
    # [修改] 移除了 HausdorffDistanceMetric

    best_metric = -1
    
    for epoch in range(EPOCHS):
        model.train()
        model.training = True
        epoch_loss = 0
        step = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_data in pbar:
            step += 1
            inputs = batch_data["image"].to(device)
            targets = {
                "seg": batch_data["seg"].to(device),
                "reg": batch_data["reg"].to(device),
                "edge": batch_data["edge"].to(device)
            }

            optimizer.zero_grad()
            
            # [修复] 正确的混合精度调用方式 (PyTorch 2.x+)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = model(inputs)
                loss, l_se, l_cr, l_ec = loss_function(outputs, targets)
            
            # 使用 scaler 进行反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item():.4f} (Seg:{l_se:.3f} Reg:{l_cr:.3f})")

        mean_loss = epoch_loss / step

        # 验证循环
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            model.training = False
            
            with torch.no_grad():
                for val_data in tqdm(val_loader, desc="Validation"):
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["seg"].to(device)
                    
                    # [优化] sw_batch_size=4 提高推理速度，显存不够则改回 2 或 1
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs, 
                        roi_size=ROI_SIZE, 
                        sw_batch_size=4, 
                        predictor=model,
                        overlap=0.5 
                    )
                    
                    val_outputs_onehot = [
                        F.one_hot(torch.argmax(i, dim=0), num_classes=3).permute(3, 0, 1, 2) 
                        for i in decollate_batch(val_outputs)
                    ]
                    val_labels_onehot = [
                        F.one_hot(i[0].long(), num_classes=3).permute(3, 0, 1, 2) 
                        for i in decollate_batch(val_labels)
                    ]

                    dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)
                    dice_metric_batch(y_pred=val_outputs_onehot, y=val_labels_onehot)
                    conf_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)
                    # [修改] 移除了 hd95_metric 调用

                metric_dice = dice_metric.aggregate().item()
                metric_batch = dice_metric_batch.aggregate()
                dice_hv = metric_batch[0].item()
                dice_pv = metric_batch[1].item()
                
                conf_matrix = conf_metric.aggregate() 
                precision = conf_matrix[0].item()
                sensitivity = conf_matrix[1].item()
                
                dice_metric.reset()
                dice_metric_batch.reset()
                conf_metric.reset()
                
                print(
                    f"\nVal - Mean Dice: {metric_dice:.4f} | HV: {dice_hv:.4f} | PV: {dice_pv:.4f}\n"
                    f"Prec: {precision:.4f} | Sens: {sensitivity:.4f}"
                )

                if metric_dice > best_metric:
                    best_metric = metric_dice
                    exp.save_model(model, optimizer, epoch + 1, is_best=True)
                    print(">>> New Best Model Saved!")
                else:
                    exp.save_model(model, optimizer, epoch + 1, is_best=False)

                exp.log_results([
                    epoch + 1, mean_loss, "N/A", 
                    metric_dice, dice_hv, dice_pv, 
                    precision, sensitivity
                ])
        else:
            exp.log_results([epoch + 1, mean_loss] + [""] * 6)
            exp.save_model(model, optimizer, epoch + 1, is_best=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()