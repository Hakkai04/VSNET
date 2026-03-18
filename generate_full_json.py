import os
import json
import nibabel as nib
from tqdm import tqdm # 推荐安装 tqdm 显示进度: pip install tqdm

def generate_thin_dataset_json(
    data_root="./dataset",
    images_dir="imagesTr",
    labels_dir="labelsTr",
    edges_dir="preprocessed/edge",
    centerlines_dir="preprocessed/reg",
    output_file="dataset_full_thin.json", # 输出为薄层专用json
    thickness_threshold=2.0 # 论文 Table 1 定义 Thin 为 [0.8, 2]
):
    """
    1. 遍历所有影像
    2. 检查辅助文件是否存在
    3. 读取 NIfTI Header 筛查层厚 (Slice Thickness <= 2.0mm)
    4. 生成 dataset_thin.json
    """
    
    # 1. 检查目录
    dirs = [images_dir, labels_dir, edges_dir, centerlines_dir]
    for d in dirs:
        if not os.path.exists(os.path.join(data_root, d)):
            print(f"❌ 错误: 找不到目录 {os.path.join(data_root, d)}")
            return

    abs_img_path = os.path.join(data_root, images_dir)
    image_files = sorted([f for f in os.listdir(abs_img_path) if f.endswith(".nii.gz")])
    
    training_data = []
    skipped_thick = []
    skipped_missing = []

    print(f"🔍 开始扫描 {len(image_files)} 个文件，正在筛选层厚 <= {thickness_threshold}mm 的数据...")
    
    # 使用 tqdm 显示进度条，因为读取 header 需要一点时间
    for img_name in tqdm(image_files):
        # --- A. 路径构建 ---
        p_img = os.path.join(data_root, images_dir, img_name)
        p_lbl = os.path.join(data_root, labels_dir, img_name)
        p_edge = os.path.join(data_root, edges_dir, img_name)
        p_line = os.path.join(data_root, centerlines_dir, img_name)

        # --- B. 检查文件完整性 ---
        if not (os.path.exists(p_lbl) and os.path.exists(p_edge) and os.path.exists(p_line)):
            skipped_missing.append(img_name)
            continue

        # --- C. 检查层厚 (核心逻辑) ---
        try:
            # nibabel 只读取 header，不加载整个图像数据，速度很快
            img_obj = nib.load(p_img)
            header = img_obj.header
            # get_zooms() 返回 (x_spacing, y_spacing, z_spacing)
            # CT 图像通常第3维 (index 2) 是层厚
            # 注意：有些 NIfTI 可能是 4D，需要防御性编程
            zooms = header.get_zooms()
            slice_thickness = zooms[2] 
            
            # 论文 Table 1: Thin-slice range is [0.8, 2]
            # 所以这里使用 <= 2.0。如果想要严格 < 2.0，请修改此处。
            if slice_thickness <= thickness_threshold:
                training_data.append({
                    "image": f"./{images_dir}/{img_name}",
                    "label": f"./{labels_dir}/{img_name}",
                    "edge": f"./{edges_dir}/{img_name}",
                    "centerline": f"./{centerlines_dir}/{img_name}",
                    "meta": {
                        "thickness": float(slice_thickness) # 记录下来备查
                    }
                })
            else:
                skipped_thick.append(f"{img_name} ({slice_thickness:.2f}mm)")
                
        except Exception as e:
            print(f"\n⚠️ 读取文件出错 {img_name}: {e}")
            skipped_missing.append(img_name)

    # 4. 构建 JSON
    json_dict = {
        "name": "VSNet_ThinSlice_Dataset",
        "description": f"Filtered dataset with slice thickness <= {thickness_threshold}mm",
        "reference": "VSNet Paper (Table 1)",
        "labels": {"0": "background", "1": "hepatic_vein", "2": "portal_vein"},
        "numTraining": len(training_data),
        "training": training_data,
        "validation": [] 
    }

    # 5. 写入
    output_path = os.path.join(data_root, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, indent=4)

    # 6. 统计报告
    print("\n" + "="*40)
    print(f"✅ 处理完成！JSON 已保存: {output_path}")
    print(f"📄 原始文件总数: {len(image_files)}")
    print(f"🎯 符合条件(Thin)入库: {len(training_data)}")
    print(f"🤏 排除(Thick > {thickness_threshold}mm): {len(skipped_thick)}")
    print(f"❌ 排除(文件缺失/损坏): {len(skipped_missing)}")
    
    if len(skipped_thick) > 0:
        print("\n--- 被排除的厚层样本示例 (前5个) ---")
        for s in skipped_thick[:5]:
            print(f"   - {s}")

if __name__ == "__main__":
    generate_thin_dataset_json(
        data_root="./dataset",
        output_file="dataset_full_thin.json" # 对应训练脚本中的 --dataset_json 参数
    )