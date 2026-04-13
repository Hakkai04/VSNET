import os
import sys
import torch
import yaml
import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from numba import njit
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, EnsureTyped, AsDiscrete,
    Invertd, SaveImaged
)
from monai.metrics import DiceMetric
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
    parser = argparse.ArgumentParser(description="Batch Predict and Evaluate all 3D CT scans from Dataset JSON")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--weight", type=str, required=True, help="Path to trained weights (best.pth)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to evaluate: 'val', 'training', or 'test'")
    parser.add_argument("--save_preds", action="store_true", help="If flag is set, physically save all NIfTI predictions")
    parser.add_argument("--output_dir", type=str, default="predict_all_results", help="Directory to save predictions if --save_preds is used")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config(args.config)
    
    # 1. Parse JSON for the split
    json_path = os.path.join(config.get("data_dir", "./dataset"), config.get("dataset_json", "dataset_full_thin.json"))
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)
    
    if args.split not in dataset_info:
        print(f"❌ Error: Split '{args.split}' not found in {json_path}")
        return
        
    data_list = dataset_info[args.split]
    print(f"📂 Loaded {len(data_list)} images from split '{args.split}'.")

    # 2. Build Model & Load Weights
    print(f"🔧 Building model [{config.get('model_name', 'vsnet')}]...")
    model = build_model(config, device=device)
    checkpoint = torch.load(args.weight, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. Validation Transforms (including Label for Metric calculation)
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=config.get("a_min", 0.0), a_max=config.get("a_max", 200.0), 
            b_min=config.get("b_min", 0.0), b_max=config.get("b_max", 1.0), clip=True
        ),
        EnsureTyped(keys=["image", "label"], track_meta=True),
    ])

    patch_size = config.get("patch_size", (96, 96, 96))
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)

    # 4. Metric Setup
    dice_metric = DiceMetric(include_background=False, reduction="none")
    post_pred = AsDiscrete(to_onehot=3, dim=1)
    post_label = AsDiscrete(to_onehot=3, dim=1)

    if args.save_preds:
        os.makedirs(args.output_dir, exist_ok=True)

    results = []
    
    # 5. Batch Inference Loop
    print("\n🚀 Starting global evaluation...\n")
    pbar = tqdm(data_list, desc="Evaluating", bar_format='{l_bar}{bar:20}{r_bar}')
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for item in pbar:
                img_rel_path = item["image"]
                lbl_rel_path = item["label"]
                basename = os.path.basename(img_rel_path)
                
                # Combine absolute path
                data_dict = {
                    "image": os.path.join(config.get("data_dir", "./dataset"), img_rel_path),
                    "label": os.path.join(config.get("data_dir", "./dataset"), lbl_rel_path)
                }
                
                try:
                    processed_data = transforms(data_dict)
                except Exception as e:
                    print(f"Error loading {basename}: {e}")
                    continue
                    
                inputs = processed_data["image"].unsqueeze(0).to(device)
                labels = processed_data["label"].unsqueeze(0).to(device)

                # Inference
                outputs = sliding_window_inference(
                    inputs, roi_size=patch_size, sw_batch_size=4, predictor=model, overlap=0.5
                )
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                # Component selection (argmax)
                preds_argmax = torch.argmax(outputs, dim=1, keepdim=True)

                # Compute Dice
                val_outputs_onehot = post_pred(preds_argmax)
                val_labels_onehot = post_label(labels)
                
                dice_scores = dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)
                # dice_scores shape: (1, 2) for two foreground classes
                dice_hv = dice_scores[0, 0].item()
                dice_pv = dice_scores[0, 1].item()
                mean_pd = torch.nanmean(dice_scores[0]).item()
                
                results.append({
                    "filename": basename,
                    "mean_dice": mean_pd,
                    "hv_dice": dice_hv,
                    "pv_dice": dice_pv
                })
                
                pbar.set_postfix({"Dice": f"{mean_pd:.4f}"})

                if args.save_preds:
                    # Put prediction back into dict and remove batch dim: (1, 1, D, H, W) -> (1, D, H, W)
                    processed_data["pred"] = preds_argmax.squeeze(0)
                    
                    # Dynamically construct post-processing pipeline for alignment
                    post_transforms = Compose([
                        Invertd(
                            keys="pred",
                            transform=transforms,
                            orig_keys="image",
                            nearest_interp=True, # crucial for discrete labels
                            to_tensor=True
                        ),
                        SaveImaged(
                            keys="pred",
                            output_dir=os.path.abspath(args.output_dir),
                            output_postfix="", 
                            output_ext=".nii.gz",
                            separate_folder=False
                        )
                    ])
                    # Force SaveImaged to use our exact intended filename
                    out_path = os.path.join(args.output_dir, basename)
                    processed_data["pred"].meta["filename_or_obj"] = out_path
                    
                    # Run inverse transform & Save in one go
                    processed_data = post_transforms(processed_data)

    # 6. Result Aggregation and Reporting
    # Sort by mean dice descending
    results.sort(key=lambda x: x["mean_dice"], reverse=True)
    
    global_mean_dice = sum(r["mean_dice"] for r in results) / len(results)
    
    print("\n" + "="*50)
    print(f"📊 EVALUATION SUMMARY (Total Images: {len(results)})")
    print(f"🌍 Global Average Dice: {global_mean_dice:.4f}")
    print("="*50)
    
    print("\n🏆 THE BEST PERFORMING IMAGES:")
    for i, res in enumerate(results[:5]):
         print(f"  {i+1}. {res['filename']} -> Mean Dice: {res['mean_dice']:.4f} (HV: {res['hv_dice']:.4f}, PV: {res['pv_dice']:.4f})")
         
    print("\n💔 THE WORST PERFORMING IMAGES (Needs Inspection):")
    for i, res in enumerate(reversed(results[-5:])): # Lowest to slightly higher
         print(f"  {i+1}. {res['filename']} -> Mean Dice: {res['mean_dice']:.4f} (HV: {res['hv_dice']:.4f}, PV: {res['pv_dice']:.4f})")
         
    if args.save_preds:
         print(f"\n📂 All predictions have been saved sequentially to '{args.output_dir}'")

if __name__ == "__main__":
    main()
