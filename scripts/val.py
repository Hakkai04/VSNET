import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from utils.data_utils import get_dataloader
from monai.utils import set_determinism
from utils.losses import SoftClDiceLoss3D

def main():
    parser = argparse.ArgumentParser(description="Standalone Validation Script for VSNet")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pth weights file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 0. 固定全局随机种子与底层 CuDNN，确保推理图对齐 engine.py
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

    # 1. 严格按照训练集的划分方式挂载数据 (固定 seed 导致划分必与之前一致)
    _, val_loader = get_dataloader(config)

    # 2. 组装网络结构并导入权重文件
    model = build_model(config).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 3. 准备评价指标容器
    dice_metric_raw = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    dice_metric_post = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch", get_not_nans=False)
    conf_metric = ConfusionMatrixMetric(
        include_background=False, 
        metric_name=["precision", "sensitivity"], 
        reduction="mean_batch", 
        get_not_nans=False
    )
    
    cldice_metric = SoftClDiceLoss3D(iter_=3)
    cldice_sum = 0.0
    val_steps = 0

    patch_size = config.get("patch_size", (96, 96, 96))
    if isinstance(patch_size, list): 
        patch_size = tuple(patch_size)

    # 开启后处理模块：只保留血管网络最大的连通孤岛，消灭周围飞蚊假阳性噪点
    # applied_labels=[1, 2] 表示分别对类别1(HV)和类别2(PV)执行
    post_transform = KeepLargestConnectedComponent(applied_labels=[1, 2], is_onehot=False)

    print(f"\n🚀 开始全面离线验证...")
    print(f"📌 加载权重池: {args.weights}")
    print(f"🔧 启用推理技术: Test-Time Augmentation (TTA), 高斯融合滑窗 (Gaussian Blending), 空间连通去噪 (Post-Proc)\n")

    with torch.no_grad():
        eval_tqdm = tqdm(val_loader, desc="Evaluating", bar_format='{l_bar}{bar:40}{r_bar}')
        for val_data in eval_tqdm:
            inputs = val_data["image"].to(device, non_blocking=True)
            labels = val_data["label"].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                sw_args = {
                    "roi_size": patch_size, 
                    "sw_batch_size": 8, 
                    "predictor": model, 
                    "overlap": 0.5, 
                    "mode": "gaussian"
                }
                
                # --- TTA 4-Pass 矩阵重组交汇推理 ---
                out1 = sliding_window_inference(inputs=inputs, **sw_args)
                if isinstance(out1, tuple): out1 = out1[0]
                prob1 = F.softmax(out1, dim=1)
                
                out2 = sliding_window_inference(inputs=torch.flip(inputs, [2]), **sw_args)
                if isinstance(out2, tuple): out2 = out2[0]
                prob2 = torch.flip(F.softmax(out2, dim=1), [2])
                
                out3 = sliding_window_inference(inputs=torch.flip(inputs, [3]), **sw_args)
                if isinstance(out3, tuple): out3 = out3[0]
                prob3 = torch.flip(F.softmax(out3, dim=1), [3])
                
                out4 = sliding_window_inference(inputs=torch.flip(inputs, [4]), **sw_args)
                if isinstance(out4, tuple): out4 = out4[0]
                prob4 = torch.flip(F.softmax(out4, dim=1), [4])
                
                probs = (prob1 + prob2 + prob3 + prob4) / 4.0

            # 提取无后处理预测
            preds_raw = torch.argmax(probs, dim=1, keepdim=True)
            
            # 加入连通域过滤后处理
            preds_post = []
            for b_idx in range(preds_raw.shape[0]):
                # MONAI的后处理操作默认是就地修改(in-place)！必须添加 clone()！！！
                cleaned_p = post_transform(preds_raw[b_idx].clone())
                preds_post.append(cleaned_p)
            preds_post = torch.stack(preds_post, dim=0)

            # 转换成 One-hot 给指标计算器
            labels_onehot = AsDiscrete(to_onehot=3, dim=1)(labels)
            preds_raw_onehot = AsDiscrete(to_onehot=3, dim=1)(preds_raw)
            preds_post_onehot = AsDiscrete(to_onehot=3, dim=1)(preds_post)

            # 更新各类缓存
            dice_metric_raw(y_pred=preds_raw_onehot, y=labels_onehot)
            dice_metric_post(y_pred=preds_post_onehot, y=labels_onehot)
            hd95_metric(y_pred=preds_post_onehot, y=labels_onehot)
            conf_metric(y_pred=preds_post_onehot, y=labels_onehot)

            # clDice (基于Softmax的平滑矩阵)
            loss_cldice = cldice_metric(labels_onehot.float(), probs)
            cldice_sum += (1.0 - loss_cldice.item())
            val_steps += 1

    # ==========================
    # 最终指标大汇总提取
    # ==========================
    res_raw_dice = dice_metric_raw.aggregate()
    res_post_dice = dice_metric_post.aggregate()
    
    mean_raw_dice = torch.nanmean(res_raw_dice).item()
    mean_post_dice = torch.nanmean(res_post_dice).item()
    
    hv_dice = res_post_dice[0].item()
    pv_dice = res_post_dice[1].item()
    
    res_hd95 = hd95_metric.aggregate()
    mean_hd95 = torch.nanmean(res_hd95).item()
    
    res_conf = conf_metric.aggregate()
    if isinstance(res_conf, (tuple, list)):
        precision = torch.nanmean(res_conf[0]).item()
        sensitivity = torch.nanmean(res_conf[1]).item()
    else:
        precision = torch.nanmean(res_conf).item()
        sensitivity = 0.0

    mean_cldice = cldice_sum / val_steps if val_steps > 0 else 0.0

    from pathlib import Path
    train_name = Path(args.weights).parent.parent.name
    save_dir = os.path.join("runs", "val", train_name)
    os.makedirs(save_dir, exist_ok=True)
    report_file = os.path.join(save_dir, "validation_report.txt")

    report_str = (
        f"\n{'='*55}\n"
        f" 📊 FINAL VALIDATION REPORT\n"
        f"{'='*55}\n"
        f"【 综合核心指标 (Overall Metrics) 】\n"
        f" 🔹 Raw Dice (生推断)             : {mean_raw_dice:.4f}\n"
        f" 🔹 Post-Processed Dice (去噪后)  : {mean_post_dice:.4f}\n"
        f" 🔹 clDice (全局拓扑连通性能)     : {mean_cldice:.4f}\n"
        f" 🔹 95HD (豪斯多夫距离)           : {mean_hd95:.4f}\n"
        f" 🔹 Precision (精确率/查准)       : {precision:.4f}\n"
        f" 🔹 Sensitivity (敏感度/查全)     : {sensitivity:.4f}\n"
        f"{'-' * 55}\n"
        f"【 细分类别解剖 (Class Decomposition After Post-Proc) 】\n"
        f" 🩸 HV (Hepatic Vein) Dice    : {hv_dice:.4f}\n"
        f" 🩸 PV (Portal Vein) Dice     : {pv_dice:.4f}\n"
        f"{'='*55}\n"
    )
    
    print(report_str)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_str)
        
    print(f"📝 验证报告已经成功保存至: {report_file}\n")

if __name__ == "__main__":
    main()
