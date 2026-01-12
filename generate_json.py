import os
import json
import glob

def generate_dataset_json(
    data_root="./dataset",
    images_dir="imagesTr",
    labels_dir="labelsTr",
    output_file="dataset.json",
):
    """
    遍历 imagesTr 和 labelsTr，生成 dataset.json
    """
    
    # 1. 确定绝对路径以确保文件存在检查无误
    abs_img_path = os.path.join(data_root, images_dir)
    abs_lbl_path = os.path.join(data_root, labels_dir)
    
    if not os.path.exists(abs_img_path) or not os.path.exists(abs_lbl_path):
        print(f"❌ 错误: 找不到路径。请检查 {abs_img_path} 和 {abs_lbl_path} 是否存在。")
        return

    # 2. 获取所有 .nii.gz 文件并排序
    # 假设你已经重命名为 hpXXX.nii.gz，这里会列出所有文件
    image_files = sorted([f for f in os.listdir(abs_img_path) if f.endswith(".nii.gz")])
    
    training_data = []
    
    print(f"🔍 正在扫描 {len(image_files)} 个影像文件...")

    # 3. 遍历影像，寻找对应的标签
    matched_count = 0
    missing_labels = []

    for img_name in image_files:
        label_path = os.path.join(abs_lbl_path, img_name)
        
        # 检查对应的标签文件是否存在
        if os.path.exists(label_path):
            # 构建相对路径（MONAI 读取时通常需要相对于 JSON 文件的路径）
            # 格式: ./imagesTr/hp001.nii.gz
            item = {
                "image": f"./{images_dir}/{img_name}",
                "label": f"./{labels_dir}/{img_name}"
            }
            training_data.append(item)
            matched_count += 1
        else:
            missing_labels.append(img_name)

    # 4. 构建 JSON 结构
    # 注意：VSNet 区分肝静脉(HV)和门静脉(PV)，通常背景是0，其他是1和2
    json_dict = {
        "name": "HepaticVessel_VSNet",
        "description": "Hepatic and Portal Vein Segmentation",
        "reference": "VSNet Paper Re-annotation",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": {
            "0": "CT"
        },
        "labels": {
            "0": "background",
            "1": "hepatic_vein", 
            "2": "portal_vein"
        },
        "numTraining": len(training_data),
        "numTest": 0,
        "training": training_data,
        "test": [] # 这里的测试集通常留空，因为你没有 Ground Truth
    }

    # 5. 写入文件
    output_path = os.path.join(data_root, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, indent=4)

    print("-" * 30)
    print(f"✅ JSON 生成完成！已保存至: {output_path}")
    print(f"📊 总影像数: {len(image_files)}")
    print(f"🔗 成功匹配标签: {matched_count}")
    
    if missing_labels:
        print(f"⚠️ 警告: 有 {len(missing_labels)} 个影像没有找到对应的标签 (不会被写入 JSON):")
        print(missing_labels[:5]) # 只打印前5个

if __name__ == "__main__":
    # 请确保这里的路径与你的实际文件夹结构一致
    # 你的结构看起来是:
    # VSNet-main/
    #   dataset/
    #      imagesTr/
    #      labelsTr/
    generate_dataset_json()