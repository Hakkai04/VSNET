import os
import glob
import numpy as np
import nibabel as nib
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion, generate_binary_structure
from tqdm import tqdm

# ================= 配置路径 =================
# 输入路径
IMAGE_DIR = "./dataset/imagesTr"
LABEL_DIR = "./dataset/labelsTr"

# 输出路径 (自动创建)
OUT_ROOT = "./dataset/preprocessed"
OUT_REG_DIR = os.path.join(OUT_ROOT, "reg")
OUT_EDGE_DIR = os.path.join(OUT_ROOT, "edge")

def setup_dirs():
    os.makedirs(OUT_REG_DIR, exist_ok=True)
    os.makedirs(OUT_EDGE_DIR, exist_ok=True)

# ================= 核心算法实现 =================

def generate_centerline_map(mask_data, affine):
    """
    实现论文公式 (8) 和 (9): Centerline Regression Task
    M_d(x) = 1 / log(e + d)
    """
    # 将 HV(1) 和 PV(2) 合并为二值掩码，因为辅助任务关注整体血管结构
    binary_mask = (mask_data > 0).astype(bool)
    
    if not np.any(binary_mask):
        return np.zeros_like(mask_data, dtype=np.float32)

    # 1. 骨架化 (Skeletonization)
    # 注意: skimage 的 skeletonize 3D 计算较慢，请耐心等待
    skeleton = skeletonize(binary_mask)
    
    # 2. 距离变换
    # 计算二值掩码内每个点到最近骨架点的欧式距离
    # distance_transform_edt 计算的是到“背景(0)”的距离
    # 所以我们反转骨架：骨架为0(目标)，其他为1
    dist_map = distance_transform_edt(np.logical_not(skeleton))
    
    # 3. 对数映射
    # 仅保留血管掩码内部的值
    final_map = np.zeros_like(dist_map, dtype=np.float32)
    mask_indices = binary_mask
    
    # 获取血管内的距离 d
    d = dist_map[mask_indices]
    
    # 应用公式: 1 / log(e + d)
    # np.e 是自然常数
    epsilon = np.e 
    values = 1.0 / np.log(epsilon + d)
    
    final_map[mask_indices] = values
    
    return final_map

def get_boundary(mask, struct):
    """提取单个掩码的边界: Mask XOR Erode(Mask)"""
    eroded = binary_erosion(mask, structure=struct)
    return np.logical_xor(mask, eroded)

def generate_expanded_edge_map(mask_data, affine):
    """
    实现论文公式 (10): Edge Segmentation Task
    C_exp = C_o U C_d U C_e
    """
    binary_mask = (mask_data > 0).astype(bool)
    
    if not np.any(binary_mask):
        return np.zeros_like(mask_data, dtype=np.uint8)

    # 定义 3D 连通性结构
    struct = generate_binary_structure(3, 1) # 3x3x3, 6-connectivity
    
    # 1. 生成三种状态的掩码
    M_o = binary_mask                       # 原始
    M_d = binary_dilation(M_o, structure=struct) # 膨胀
    M_e = binary_erosion(M_o, structure=struct)  # 腐蚀
    
    # 2. 提取三种状态的边缘
    C_o = get_boundary(M_o, struct)
    C_d = get_boundary(M_d, struct)
    C_e = get_boundary(M_e, struct)
    
    # 3. 求并集 (Union)
    C_exp = np.logical_or(C_o, np.logical_or(C_d, C_e))
    
    return C_exp.astype(np.uint8) # 边缘是二分类任务 (0, 1)

# ================= 主流程 =================

def run_preprocessing():
    setup_dirs()
    
    # 获取所有标签文件
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.nii.gz")))
    print(f"发现 {len(label_files)} 个标签文件，准备处理...")
    
    for label_path in tqdm(label_files):
        file_name = os.path.basename(label_path)
        
        # 读取标签
        lbl_obj = nib.load(label_path)
        lbl_data = lbl_obj.get_fdata()
        affine = lbl_obj.affine
        
        # --- 任务 1: 生成中心线回归图 ---
        reg_data = generate_centerline_map(lbl_data, affine)
        reg_save_path = os.path.join(OUT_REG_DIR, file_name) # 保持同名，方便对应
        nib.save(nib.Nifti1Image(reg_data, affine), reg_save_path)
        
        # --- 任务 2: 生成扩展边缘图 ---
        edge_data = generate_expanded_edge_map(lbl_data, affine)
        edge_save_path = os.path.join(OUT_EDGE_DIR, file_name)
        nib.save(nib.Nifti1Image(edge_data, affine), edge_save_path)

    print("\n预处理完成！")
    print(f"中心线图保存在: {OUT_REG_DIR}")
    print(f"边缘图保存在: {OUT_EDGE_DIR}")

if __name__ == "__main__":
    run_preprocessing()