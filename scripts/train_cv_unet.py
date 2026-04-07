import os
import sys
import yaml
import json
import torch
import argparse
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete

# 把主目录加入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models import build_model
from utils import build_loss
from utils.data_utils import get_transforms

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

def get_logger(save_dir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    
    file_handler = logging.FileHandler(os.path.join(save_dir, "train_cv.log"))
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 避免重复添加handler
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_unet.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.exists(config_path) and not os.path.isabs(config_path):
        config_path = os.path.join(parent_dir, args.config)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if 'patch_size' in config:
        config['patch_size'] = tuple(config['patch_size'])
            
    return config

def get_cv_dataloaders(config, train_files, val_files):
    batch_size = config.get("batch_size", 2)
    train_transforms, val_transforms = get_transforms(config)

    def worker_init_fn(worker_id):
        np.random.seed(torch.initial_seed() % (2**32) + worker_id)

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn,
        persistent_workers=True
    )
    
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    return train_loader, val_loader

def evaluate_best_model(model, val_loader, config, device, logger, fold_dir):
    """单独加载最好模型，进行四项指标详尽评估：Dice, Precision, Sensitivity, HD95"""
    model.eval()
    patch_size = config.get("patch_size", (96, 96, 96))
    
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch", get_not_nans=False)
    conf_matrix_metric = ConfusionMatrixMetric(
        include_background=False, 
        metric_name=["precision", "sensitivity"], 
        reduction="mean_batch", 
        get_not_nans=False
    )
    with torch.no_grad():
        val_tqdm = tqdm(val_loader, desc=f"Evaluating Best Model", leave=False)
        for i, val_data in enumerate(val_tqdm):
            val_inputs = val_data["image"].to(device, non_blocking=True)
            val_labels = val_data["label"].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                val_outputs = sliding_window_inference(
                    inputs=val_inputs, 
                    roi_size=patch_size, 
                    sw_batch_size=4, 
                    predictor=model,
                    overlap=0.5
                )
            
            if isinstance(val_outputs, tuple):
                val_outputs = val_outputs[0]
                
            val_outputs_onehot = AsDiscrete(argmax=True, to_onehot=3, dim=1)(val_outputs)
            val_labels_onehot = AsDiscrete(to_onehot=3, dim=1)(val_labels)

            dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)
            hd95_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)
            conf_matrix_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

    # 聚合指标
    dice_score = dice_metric.aggregate()
    hd95_score = hd95_metric.aggregate()
    conf_metrics = conf_matrix_metric.aggregate()

    mean_dice = torch.nanmean(dice_score).item()
    mean_hd95 = torch.nanmean(hd95_score).item()
    
    if isinstance(conf_metrics, (tuple, list)):
        precision_score = torch.nanmean(conf_metrics[0]).item()
        sensitivity_score = torch.nanmean(conf_metrics[1]).item()
    else:
        precision_score = torch.nanmean(conf_metrics).item()
        sensitivity_score = 0.0

    logger.info("--------------------------------------------------")
    logger.info(f"Fold Best Model Detailed Metrics:")
    logger.info(f"Mean Dice: {mean_dice:.4f}")
    logger.info(f"Precision: {precision_score:.4f}")
    logger.info(f"Sensitivity: {sensitivity_score:.4f}")
    logger.info(f"HD95: {mean_hd95:.4f}")
    logger.info("--------------------------------------------------")

    return {
        "dice": mean_dice,
        "precision": precision_score,
        "sensitivity": sensitivity_score,
        "hd95": mean_hd95
    }

def main():
    config = get_config()
    
    # 随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    project = config.get("project", "runs/train")
    name = config.get("name", "unet_cv")
    save_dir = increment_path(Path(project) / name, mkdir=True)
    
    logger = get_logger(save_dir, "CV_Training")
    logger.info(f"🚀 5-Fold Cross Validation started. Results saved to {save_dir}")
    logger.info(f"Using device: {device}")
    
    with open(save_dir / 'opt_cv.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # 1. 解析数据
    data_dir = config.get("data_dir", "./dataset")
    dataset_json = config.get("dataset_json", "dataset_full_thin.json")
    json_path = os.path.join(data_dir, dataset_json)
    
    with open(json_path, "r") as f:
        data_json = json.load(f)

    # 只用 training 数据进行五折
    raw_files = data_json.get("training", [])
    parsed_files = []
    for item in raw_files:
        parsed_files.append({
            "image": os.path.normpath(os.path.join(data_dir, item["image"])),
            "label": os.path.normpath(os.path.join(data_dir, item["label"]))
        })
    parsed_files = np.array(parsed_files)
    
    k_folds = config.get("k_folds", 5)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    all_fold_metrics = []

    # 2. 开始每一折的训练
    for fold, (train_idx, val_idx) in enumerate(kf.split(parsed_files)):
        logger.info(f"\n{'='*50}\nStarting Fold {fold + 1}/{k_folds}\n{'='*50}")
        fold_dir = os.path.join(save_dir, f"fold_{fold+1}")
        weights_dir = os.path.join(fold_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        train_files = parsed_files[train_idx].tolist()
        val_files = parsed_files[val_idx].tolist()
        logger.info(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}")
        
        train_loader, val_loader = get_cv_dataloaders(config, train_files, val_files)
        
        model = build_model(config, device=device)
        criterion = build_loss(config).to(device)
        
        lr = config.get("lr", 1e-3)
        weight_decay = config.get("weight_decay", 1e-4)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler('cuda')
        
        max_epochs = config.get("max_epochs", 10000)
        patience = config.get("patience", 1000)
        val_interval = config.get("val_interval", 5)
        
        batch_size = config.get("batch_size", 2)
        target_batch_size = config.get("target_batch_size", 16)
        accumulation_steps = max(1, target_batch_size // batch_size)
        patch_size = config.get("patch_size", (96, 96, 96))
        
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        writer = SummaryWriter(log_dir=fold_dir)
        
        best_dice = -1.0
        best_epoch = -1
        
        # 为了普通验证快速进行，这里只算Dice
        fast_dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

        for epoch in range(1, max_epochs + 1):
            # -------- 训练阶段 --------
            model.train()
            running_loss = 0.0
            steps = 0
            
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch}/{max_epochs}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            optimizer.zero_grad()
            
            for i, batch_data in enumerate(pbar):
                steps += 1
                inputs = batch_data["image"].to(device, non_blocking=True)
                targets = {k: v.to(device, non_blocking=True) for k, v in batch_data.items() if k != "image"}
                targets["epoch"] = epoch
                
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss, loss_dict = criterion(outputs, targets)
                    loss = loss / accumulation_steps
                    
                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                current_loss = loss.item() * accumulation_steps
                running_loss += current_loss
                
                postfix_dict = {"Loss": f"{current_loss:.4f}"}
                for k, v in loss_dict.items():
                    postfix_dict[k] = f"{v:.4f}"
                pbar.set_postfix(postfix_dict)
                
            if (len(train_loader) % accumulation_steps) != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            avg_loss = running_loss / steps
            writer.add_scalar("Train/Total_Loss", avg_loss, epoch)
            scheduler.step()

            # -------- 验证阶段 --------
            if epoch % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_tqdm = tqdm(val_loader, desc="Validating", leave=False)
                    for val_data in val_tqdm:
                        val_inputs = val_data["image"].to(device, non_blocking=True)
                        val_labels = val_data["label"].to(device, non_blocking=True)
                        
                        with torch.amp.autocast('cuda'):
                            val_outputs = sliding_window_inference(
                                inputs=val_inputs, 
                                roi_size=patch_size, 
                                sw_batch_size=4, 
                                predictor=model,
                                overlap=0.5
                            )
                        if isinstance(val_outputs, tuple):
                            val_outputs = val_outputs[0]
                            
                        val_outputs_onehot = AsDiscrete(argmax=True, to_onehot=3, dim=1)(val_outputs)
                        val_labels_onehot = AsDiscrete(to_onehot=3, dim=1)(val_labels)
                        fast_dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

                    dice_score = fast_dice_metric.aggregate()
                    mean_dice = torch.nanmean(dice_score).item()
                    fast_dice_metric.reset()
                
                logger.info(f"Fold {fold+1} Epoch {epoch} | Val Mean Dice: {mean_dice:.4f}")
                writer.add_scalar("Val/Mean_Dice", mean_dice, epoch)
                
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(weights_dir, "best.pth"))
                    logger.info(f"🏆 Fold {fold+1} New Best! (Dice: {best_dice:.4f})")
                
                if best_epoch > 0 and (epoch - best_epoch) >= patience:
                    logger.info(f"⚠️ Early stopping triggered for Fold {fold+1}!")
                    break

            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        logger.info(f"Fold {fold+1} Training completed. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
        writer.close()
        
        # -------- Fold结束：详尽评估最佳模型 --------
        best_model_path = os.path.join(weights_dir, "best.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            fold_metrics = evaluate_best_model(model, val_loader, config, device, logger, fold_dir)
            all_fold_metrics.append(fold_metrics)
        else:
            logger.error(f"Best model path not found: {best_model_path}")

    # -------- 五折结束：打印均值 --------
    if len(all_fold_metrics) > 0:
        logger.info(f"\n\n{'='*50}\n5-Fold Cross Validation Final Results\n{'='*50}")
        
        avg_metrics = {k: np.mean([m[k] for m in all_fold_metrics]) for k in all_fold_metrics[0].keys()}
        std_metrics = {k: np.std([m[k] for m in all_fold_metrics]) for k in all_fold_metrics[0].keys()}
        
        for k in avg_metrics.keys():
            logger.info(f"{k.capitalize():>12}: {avg_metrics[k]:.4f} ± {std_metrics[k]:.4f}")
            for f_idx, m in enumerate(all_fold_metrics):
                logger.info(f"    Fold {f_idx+1}: {m[k]:.4f}")
                
        logger.info(f"{'='*50}")

if __name__ == "__main__":
    main()
