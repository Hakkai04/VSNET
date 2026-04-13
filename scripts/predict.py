import os
import sys
import torch
import yaml
import argparse
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, EnsureTyped, Invertd
)
from monai.inferers import sliding_window_inference

# Setup path parsing so we can safely import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models import build_model

def get_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Predict single 3D CT scan for 3D reconstruction and QA")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml (e.g. config_cga_unet.yaml)")
    parser.add_argument("--weight", type=str, required=True, help="Path to trained weights (e.g. runs/train/cga_unet/best.pth)")
    parser.add_argument("--image", type=str, required=True, help="Path to input raw image .nii.gz")
    parser.add_argument("--output", type=str, default="prediction.nii.gz", help="Path to save the predicted mask .nii.gz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config(args.config)
    
    # Force tuple parsing for yaml lists
    patch_size = config.get("patch_size", (96, 96, 96))
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)

    # 1. Build Model & Load Weights
    print(f"🔧 Building model [{config.get('model_name', 'vsnet')}]...")
    model = build_model(config, device=device)
    
    if not os.path.exists(args.weight):
        print(f"❌ Error: Weights file not found: {args.weight}")
        return
        
    print(f"📥 Loading weights from {args.weight}...")
    checkpoint = torch.load(args.weight, map_location=device)
    
    # Compatibility with checkpoints that save epoch/optimizer wrapper
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("✅ Model initialized and ready.")

    # 2. Image Transformations (Must be mathematically identical to Validation pipeline)
    print(f"💽 Pre-processing image: {args.image}")
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=config.get("a_min", 0.0), a_max=config.get("a_max", 200.0), 
            b_min=config.get("b_min", 0.0), b_max=config.get("b_max", 1.0), clip=True
        ),
        EnsureTyped(keys=["image"], track_meta=True),
    ])

    data = {"image": args.image}
    try:
        data = transforms(data)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
        
    inputs = data["image"].unsqueeze(0).to(device) # Shape: (1, C, D, H, W)

    # 3. Sliding Window Inference
    print("🧠 Running 3D sliding window inference... (This might take a minute)")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            outputs = sliding_window_inference(
                inputs, roi_size=patch_size, sw_batch_size=4, predictor=model, overlap=0.5
            )
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 使用 keepdim=True 保持通道维度，形状变为 (1, 1, H, W, D)
            discrete_preds = torch.argmax(outputs, dim=1, keepdim=True)

    # --- 以下为修改后的后处理与保存逻辑 ---
    from monai.transforms import Invertd, SaveImaged

    # 将预测结果放回字典，去掉 batch 维度，形状变为 (1, H, W, D)
    data["pred"] = discrete_preds.squeeze(0) 

    print("🔄 Inverting transforms to match original CT spacing and orientation...")
    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=transforms,      # 传入你之前的预处理 transform
            orig_keys="image",         # 参照原始图像的元数据进行还原
            nearest_interp=True,       # 极其重要：分割标签必须使用最近邻插值，否则会产生 0.5 这样的脏标签！
            to_tensor=True
        ),
        SaveImaged(
            keys="pred", 
            output_dir=os.path.dirname(os.path.abspath(args.output)) or ".", 
            output_postfix="",         # 不添加额外的后缀
            output_ext=".nii.gz", 
            separate_folder=False      # 不要为每个文件创建独立文件夹
        )
    ])

    # 4. 执行反向变换并自动保存
    # MONAI 会自动接管 Affine 矩阵，并确保与原始 NIfTI 头文件 100% 对齐
    print(f"💾 Exporting aligned predictions to: {args.output}")
    try:
        # 为了让 SaveImaged 按我们指定的文件名保存，我们需要临时修改元数据中的文件名
        data["pred"].meta["filename_or_obj"] = args.output
        data = post_transforms(data)
        print("🎉 Done! 预测文件现在与原始 CT 的物理空间完全一致。")
    except Exception as e:
        print(f"❌ Error during post-processing/saving: {e}")

if __name__ == "__main__":
    main()