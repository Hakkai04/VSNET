import os

# ==========================================
# 1. 显卡与绘图后端设置 (必须在最前面)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 根据需要修改 GPU 编号
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import yaml
import json
import torch
import argparse
import logging
from pathlib import Path
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism

# 添加主目录到 sys.path，以便导入 models 和 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models import build_model
from utils import get_dataloader, build_loss, Trainer

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

def get_logger(save_dir, model_name):
    logger = logging.getLogger(f"{model_name.upper()}_Training")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    
    file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def get_config():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    
    # Allow overriding common args
    parser.add_argument("--model_name", type=str, help="Model to train (e.g. vsnet, unet, vnet, attention_unet)")
    parser.add_argument("--batch_size", type=int, help="Physical batch size per GPU")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--weights", type=str, help="Path to pretrained weights for fine-tuning")
    
    args = parser.parse_args()
    
    config_path = args.config
    # 兼容相对路径
    if not os.path.exists(config_path) and not os.path.isabs(config_path):
        config_path = os.path.join(parent_dir, args.config)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    for key, value in vars(args).items():
        if value is not None and key != "config":
            config[key] = value
            
    if 'patch_size' in config:
        config['patch_size'] = tuple(config['patch_size'])
            
    return config

def main():
    config = get_config()
    
    # 固定全局随机种子，确保每次运行的数据划分及网络初始化结果绝对一致
    seed = 42
    import random
    import numpy as np
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
    
    # 1. 目录初始化
    model_name = config.get("model_name", "vsnet")
    project = config.get("project", "runs/train")
    # 如果配置中没写name，或者使用默认的name=vsnet，根据模型覆盖
    name = config.get("name", model_name)
    if name == "vsnet" and model_name != "vsnet":
        name = model_name
        
    save_dir = increment_path(Path(project) / name, mkdir=True)
    
    logger = get_logger(save_dir, model_name)
    writer = SummaryWriter(log_dir=str(save_dir))
    
    logger.info(f"🚀 {model_name.upper()} Training started. Results saved to {save_dir}")
    logger.info(f"Using device: {device}")
    
    # 保存整合后的配置
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # 2. 数据与模型准备
    train_loader, val_loader = get_dataloader(config)
    model = build_model(config, device=device)
    
    weights_path = config.get("weights", "")
    if weights_path and os.path.exists(weights_path):
        logger.info(f"🔄 Loading pretrained weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # ---- 自动前缀映射 ----
        # 如果模型是 EdgeGuidedSwinUNETR（内部将 SwinUNETR 包装在 self.swin_unetr 中），
        # 而预训练权重来自独立训练的 SwinUNETR（key 没有 'swin_unetr.' 前缀），
        # 则需要自动添加前缀使 key 匹配。
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # 检测是否需要添加前缀：如果 checkpoint 的 key 都不在模型中，
        # 但加上 'swin_unetr.' 前缀后能匹配，则执行映射
        if len(model_keys & ckpt_keys) == 0:
            prefixed = {f"swin_unetr.{k}": v for k, v in state_dict.items()}
            if len(model_keys & set(prefixed.keys())) > 0:
                logger.info("🔧 检测到权重 key 前缀不匹配，自动添加 'swin_unetr.' 前缀进行映射")
                state_dict = prefixed
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"✅ Pretrained weights loaded. Missing: {len(missing)} keys, Unexpected: {len(unexpected)} keys")
        if missing:
            logger.info(f"   Missing keys (新增模块，将随机初始化): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        logger.info("✅ Optimizer will start fresh!")
    
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {params_num / 1e6:.2f}M")

    # 3. 损失与优化器
    criterion = build_loss(config).to(device)
    optimizer = AdamW(
        model.parameters(), 
        lr=config.get("lr", 1e-3), 
        weight_decay=config.get("weight_decay", 1e-5)
    )
    scaler = torch.amp.GradScaler('cuda')

    # 4. 初始化引擎
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
        save_dir=str(save_dir),
        device=device
    )
    
    # 5. 开始训练
    trainer.fit()

if __name__ == "__main__":
    main()
