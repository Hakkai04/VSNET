import os
import json
import torch
import numpy as np
import random
import logging
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, RandFlipd, RandRotated, RandShiftIntensityd, EnsureTyped,
    SpatialPadd, Spacingd
)

logger = logging.getLogger("DataUtils")

def get_transforms(config):
    patch_size = config.get("patch_size", (96, 96, 96))
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)
        
    num_samples = config.get("num_samples", 4)
    model_name = config.get("model_name", "vsnet").lower()

    # 根据模型决定加载哪些键
    if model_name in ["vsnet", "lynet"]:
        keys = ["image", "label", "edge", "reg"]
        modes = ("bilinear", "nearest", "nearest", "bilinear")
    else:
        keys = ["image", "label"]
        modes = ("bilinear", "nearest")

    train_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=modes),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=config.get("a_min", 0.0), a_max=config.get("a_max", 200.0), 
            b_min=config.get("b_min", 0.0), b_max=config.get("b_max", 1.0), clip=True
        ),
        SpatialPadd(keys=keys, spatial_size=patch_size),
        RandCropByPosNegLabeld(
            keys=keys, label_key="label", spatial_size=patch_size,
            pos=1, neg=1, num_samples=num_samples, image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandRotated(keys=keys, range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5, mode=modes),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=keys, track_meta=False),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]), 
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=config.get("a_min", 0.0), a_max=config.get("a_max", 200.0), 
            b_min=config.get("b_min", 0.0), b_max=config.get("b_max", 1.0), clip=True
        ),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ])
    
    return train_transforms, val_transforms

def get_dataloader(config):
    data_dir = config.get("data_dir", "./dataset")
    dataset_json = config.get("dataset_json", "dataset_full_thin.json")
    model_name = config.get("model_name", "vsnet").lower()
    batch_size = config.get("batch_size", 4)
    
    json_path = os.path.join(data_dir, dataset_json)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")
        
    with open(json_path, "r") as f:
        data_json = json.load(f)

    # 统一文件路径解析
    def parse_files(file_list):
        parsed = []
        for item in file_list:
            if model_name in ["vsnet", "lynet"]:
                parsed.append({
                    "image": os.path.normpath(os.path.join(data_dir, item["image"])),
                    "label": os.path.normpath(os.path.join(data_dir, item["label"])),
                    "edge": os.path.normpath(os.path.join(data_dir, item["edge"])),
                    "reg": os.path.normpath(os.path.join(data_dir, item["centerline"]))
                })
            else:
                parsed.append({
                    "image": os.path.normpath(os.path.join(data_dir, item["image"])),
                    "label": os.path.normpath(os.path.join(data_dir, item["label"]))
                })
        return parsed

    train_files = parse_files(data_json.get("training", []))
    val_files = parse_files(data_json.get("validation", []))

    # 根据要求："请将数据集按 4:1 的比例随机划分为训练集和测试集 （Hold-out validation 策略）"
    # 我们合并所有数据并进行固定随机种子的 4:1 划分
    all_files = train_files + val_files
    random.seed(42)  # 严格使用固定种子打乱
    random.shuffle(all_files)
    
    # 4:1 划分即 80% 训练集, 20% 测试/验证集
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    logger.info(f"Hold-out Validation Strategy (4:1 split) -> Train samples: {len(train_files)}, Val/Test samples: {len(val_files)}")

    train_transforms, val_transforms = get_transforms(config)

    def worker_init_fn(worker_id):
        np.random.seed(torch.initial_seed() % (2**32) + worker_id)

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn,
        persistent_workers=True
    )
    
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader
