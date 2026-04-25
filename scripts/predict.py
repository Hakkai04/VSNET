"""
predict.py — 单例推理脚本
============================================================
用法:
  python scripts/predict.py \
      --config config_edge_swin.yaml \
      --weights runs/train/edge_guided_swin_unetr/weights/best.pth \
      --input  /path/to/patient.nii.gz \
      --output runs/predict

  也可以传入一个目录，批量预测里面所有 .nii.gz 文件:
  python scripts/predict.py \
      --config config_edge_swin.yaml \
      --weights runs/train/edge_guided_swin_unetr/weights/best.pth \
      --input  /path/to/nifti_folder/ \
      --output runs/predict

输出:
  - 保留原始 NIfTI 的 affine 和 header（空间坐标系完全一致）
  - 在 3D Slicer 中可以直接叠加到原始 CT 上查看
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import yaml
import glob
import torch
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRanged, EnsureTyped, Spacingd
)
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader

# 添加主目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models import build_model


def get_args():
    parser = argparse.ArgumentParser(description="VSNet Inference Script")
    parser.add_argument("--config", type=str, required=True, help="训练时使用的 config yaml 文件路径")
    parser.add_argument("--weights", type=str, required=True, help="训练好的模型权重路径 (.pth)")
    parser.add_argument("--input", type=str, required=True, help="输入 .nii.gz 文件或包含 .nii.gz 文件的目录")
    parser.add_argument("--output", type=str, default="runs/predict", help="预测结果输出目录")
    parser.add_argument("--overlap", type=float, default=0.5, help="滑动窗口推理的重叠率")
    parser.add_argument("--sw_batch_size", type=int, default=8, help="滑动窗口内部的 batch size")
    parser.add_argument("--tta", action="store_true", help="是否使用 TTA (Test-Time Augmentation)")
    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path) and not os.path.isabs(config_path):
        config_path = os.path.join(parent_dir, config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'patch_size' in config:
        config['patch_size'] = tuple(config['patch_size'])
    return config


def get_inference_transforms(config):
    """构建与训练时完全一致的预处理流水线（不含随机增强）"""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config.get("a_min", -200.0),
            a_max=config.get("a_max", 250.0),
            b_min=config.get("b_min", 0.0),
            b_max=config.get("b_max", 1.0),
            clip=True,
        ),
        EnsureTyped(keys=["image"], track_meta=False),
    ])


def predict_single(model, data, patch_size, device, overlap=0.5, sw_batch_size=8, use_tta=False):
    """对单个样本进行滑动窗口推理"""
    inputs = data["image"].unsqueeze(0).to(device)  # (1, C, D, H, W)

    sw_args = {
        "roi_size": patch_size,
        "sw_batch_size": sw_batch_size,
        "predictor": model,
        "overlap": overlap,
        "mode": "gaussian",
    }

    with torch.amp.autocast('cuda'):
        # Pass 1: 原始
        out1 = sliding_window_inference(inputs=inputs, **sw_args)
        if isinstance(out1, (tuple, list)):
            out1 = out1[0]

        if use_tta:
            import torch.nn.functional as F
            # Pass 2: Flip Depth
            out2 = sliding_window_inference(inputs=torch.flip(inputs, [2]), **sw_args)
            if isinstance(out2, (tuple, list)): out2 = out2[0]
            prob2 = torch.flip(F.softmax(out2, dim=1), [2])

            # Pass 3: Flip Height
            out3 = sliding_window_inference(inputs=torch.flip(inputs, [3]), **sw_args)
            if isinstance(out3, (tuple, list)): out3 = out3[0]
            prob3 = torch.flip(F.softmax(out3, dim=1), [3])

            # Pass 4: Flip Width
            out4 = sliding_window_inference(inputs=torch.flip(inputs, [4]), **sw_args)
            if isinstance(out4, (tuple, list)): out4 = out4[0]
            prob4 = torch.flip(F.softmax(out4, dim=1), [4])

            # TTA 平均
            avg_prob = (F.softmax(out1, dim=1) + prob2 + prob3 + prob4) / 4.0
            pred = torch.argmax(avg_prob, dim=1).squeeze(0)  # (D, H, W)
        else:
            pred = torch.argmax(out1, dim=1).squeeze(0)  # (D, H, W)

    return pred.cpu().numpy().astype(np.uint8)


def save_prediction(pred_array, reference_path, output_path):
    """
    将预测结果保存为 NIfTI 文件，完整保留原始图像的空间信息。
    这样在 3D Slicer 中打开时，预测结果会自动与原始 CT 对齐。
    """
    # 读取原始文件获取 affine 和 header
    ref_img = nib.load(reference_path)
    ref_affine = ref_img.affine
    ref_header = ref_img.header.copy()

    # 由于推理时做了 Spacing resampling (1mm isotropic) 和 Orientation (RAS)，
    # 预测结果的空间尺寸可能和原始图像不完全一致。
    # 我们需要将预测结果重采样回原始空间。
    original_shape = ref_img.shape[:3]
    pred_shape = pred_array.shape

    if pred_shape != original_shape:
        # 使用最近邻插值将预测结果重采样回原始分辨率
        from scipy.ndimage import zoom
        zoom_factors = [o / p for o, p in zip(original_shape, pred_shape)]
        pred_array = zoom(pred_array, zoom_factors, order=0)  # order=0 = 最近邻

    # 构建 NIfTI 图像
    pred_nifti = nib.Nifti1Image(pred_array, affine=ref_affine, header=ref_header)

    # 设置数据类型为无符号整型（标签图）
    pred_nifti.header.set_data_dtype(np.uint8)

    # 保存
    nib.save(pred_nifti, output_path)


def main():
    args = get_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = config.get("patch_size", (96, 96, 96))

    # ============ 1. 收集输入文件 ============
    input_path = args.input
    if os.path.isdir(input_path):
        nii_files = sorted(glob.glob(os.path.join(input_path, "*.nii.gz")))
        if not nii_files:
            nii_files = sorted(glob.glob(os.path.join(input_path, "*.nii")))
        if not nii_files:
            print(f"❌ 在目录 {input_path} 中没有找到任何 .nii.gz 或 .nii 文件")
            return
    else:
        if not os.path.exists(input_path):
            print(f"❌ 文件不存在: {input_path}")
            return
        nii_files = [input_path]

    print(f"📂 找到 {len(nii_files)} 个待预测文件")

    # ============ 2. 创建输出目录 ============
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 预测结果将保存至: {output_dir}")

    # ============ 3. 加载模型 ============
    print(f"🔧 构建模型: {config.get('model_name', 'unknown')}")
    model = build_model(config, device=device)

    print(f"🔄 加载权重: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    params_num = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型加载完成 | 参数量: {params_num / 1e6:.2f}M")

    # ============ 4. 预处理 ============
    transforms = get_inference_transforms(config)

    # ============ 5. 逐例推理 ============
    print(f"🚀 开始推理... (TTA: {'开启' if args.tta else '关闭'})\n")

    for nii_path in tqdm(nii_files, desc="Predicting"):
        filename = os.path.basename(nii_path)
        stem = filename.replace(".nii.gz", "").replace(".nii", "")

        # 预处理
        data = transforms({"image": nii_path})

        # 推理
        with torch.no_grad():
            pred_array = predict_single(
                model=model,
                data=data,
                patch_size=patch_size,
                device=device,
                overlap=args.overlap,
                sw_batch_size=args.sw_batch_size,
                use_tta=args.tta,
            )

        # 保存（保留原始空间信息）
        out_path = os.path.join(output_dir, f"{stem}_pred.nii.gz")
        save_prediction(pred_array, reference_path=nii_path, output_path=out_path)

        print(f"  ✅ {filename} → {os.path.basename(out_path)}")

    print(f"\n🎉 推理完成! 共预测 {len(nii_files)} 个文件，结果保存在: {output_dir}")
    print("💡 在 3D Slicer 中: File → Add Data → 选择预测文件 → 在 Volume 中设置为 LabelMap 即可叠加查看")


if __name__ == "__main__":
    main()
