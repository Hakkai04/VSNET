import os
# ==========================================
# 1. 显卡与绘图后端设置 (必须在最前面)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 根据需要修改 GPU 编号
import sys
import yaml
import torch
import argparse
import logging
from pathlib import Path
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models import build_model
from utils import get_dataloader, build_loss, Trainer
from scripts.train import get_logger

def main():
    parser = argparse.ArgumentParser(description="Resume Training Script")
    parser.add_argument("--run_dir", type=str, default="runs/train/vsnet2", help="Path to the training directory to resume")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    # 兼容相对于项目根目录的路径
    if not run_dir.is_absolute():
        run_dir = Path(parent_dir) / args.run_dir

    if not run_dir.exists():
        print(f"Error: Directory {run_dir} does not exist.")
        return

    # 1. 加载配置
    config_path = run_dir / 'opt.yaml'
    if not config_path.exists():
        print(f"Error: Could not find {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.unsafe_load(f)

    if 'patch_size' in config:
        config['patch_size'] = tuple(config['patch_size'])

    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化日志和tensorboard
    model_name = config.get("model_name", "vsnet")
    logger = get_logger(run_dir, model_name)
    writer = SummaryWriter(log_dir=str(run_dir))
    
    logger.info(f"🔄 Resuming {model_name.upper()} Training from {run_dir}")
    logger.info(f"Using device: {device}")

    # 2. 数据与模型准备
    train_loader, val_loader = get_dataloader(config)
    model = build_model(config, device=device)

    # 3. 损失与优化器
    criterion = build_loss(config).to(device)
    optimizer = AdamW(
        model.parameters(), 
        lr=config.get("lr", 1e-3), 
        weight_decay=config.get("weight_decay", 1e-5)
    )
    scaler = torch.amp.GradScaler('cuda')

    # 4. 加载检查点
    start_epoch = 1
    best_dice = -1.0
    ckpt_path = run_dir / "weights" / "last.pth"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # 恢复 best_dice
        best_ckpt_path = run_dir / "weights" / "best.pth"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            best_dice = best_ckpt.get('dice', -1.0)
            logger.info(f"✅ Recovered best dice: {best_dice:.4f}")
    else:
        logger.warning(f"⚠️ No checkpoint found at {ckpt_path}, starting from scratch.")

    # 5. 初始化引擎
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        logger=logger,
        writer=writer,
        save_dir=str(run_dir),
        device=device
    )
    
    trainer.start_epoch = start_epoch
    if best_dice > 0:
        trainer.best_dice = best_dice

    # 6. 开始训练
    trainer.fit()

if __name__ == "__main__":
    main()
