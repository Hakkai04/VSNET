import os
import glob
import numpy as np
import nibabel as nib
import concurrent.futures
import multiprocessing
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
    其中 d = min ||x-y||^2 是点云到最近中心线距离的平方。
    """
    # 将 HV(1) 和 PV(2) 合并为二值掩码，因为辅助任务关注整体血管结构
    binary_mask = (mask_data > 0).astype(bool)
    
    if not np.any(binary_mask):
        return np.zeros_like(mask_data, dtype=np.float32)

    # 1. 骨架化 (Skeletonization)
    # 注意: skimage 的 skeletonize 3D 计算较慢，请耐心等待
    skeleton = skeletonize(binary_mask)
    
    # 2. 距离变换
    # 计算二值掩码内每个点到最近骨架点的欧式距离 (Euclidean Distance Transform)
    # distance_transform_edt 计算的是到“背景(0)”的欧式距离
    # 所以我们反转骨架：骨架为0(目标)，其他为1
    dist_euclidean = distance_transform_edt(np.logical_not(skeleton))
    
    # 3. 对数映射
    # 仅保留血管掩码内部的值
    final_map = np.zeros_like(dist_euclidean, dtype=np.float32)
    
    # 获取血管内的欧氏距离
    euclidean_dist = dist_euclidean[binary_mask]
    
    # 根据论文公式，令 d 为欧氏距离的平方：d = min ||x-y||^2
    d = euclidean_dist ** 2
    
    # 应用对数映射函数: M_d(x) = 1 / log(e + d)
    # np.e 是自然对数底数 e
    epsilon = np.e 
    values = 1.0 / np.log(epsilon + d)
    
    final_map[binary_mask] = values
    
    return final_map

def get_boundary(mask, struct):
    """提取单个掩码的边界: Mask XOR Erode(Mask)"""
    eroded = binary_erosion(mask, structure=struct)
    return np.logical_xor(mask, eroded)

def generate_expanded_edge_map(mask_data, affine):
    """
    实现论文公式 (10): Edge Segmentation Task
    C_exp = C_o U C_d U C_e
    该段逻辑完全符合原有要求，保留工程优化。
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


# ================= 多进程处理模块 =================

def process_single_file(label_path):
    """工作进程：处理单个 NIfTI 标签的任务函数"""
    file_name = os.path.basename(label_path)
    
    # 读取标签
    lbl_obj = nib.load(label_path)
    lbl_data = lbl_obj.get_fdata()
    affine = lbl_obj.affine
    
    # --- 任务 1: 生成中心线回归图 ---
    reg_data = generate_centerline_map(lbl_data, affine)
    reg_save_path = os.path.join(OUT_REG_DIR, file_name) # 保持同名，方便对应
    nib.save(nib.Nifti1Image(reg_data.astype(np.float32), affine), reg_save_path)
    
    # --- 任务 2: 生成扩展边缘图 ---
    edge_data = generate_expanded_edge_map(lbl_data, affine)
    edge_save_path = os.path.join(OUT_EDGE_DIR, file_name)
    nib.save(nib.Nifti1Image(edge_data.astype(np.uint8), affine), edge_save_path)
    
    return file_name

# ================= 主流程 =================

def run_preprocessing():
    setup_dirs()
    
    # 获取所有标签文件
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.nii.gz")))
    print(f"发现 {len(label_files)} 个标签文件，准备处理...")
    
    if not label_files:
        print("未找到任何标签文件，请检查输入路径。")
        return

    # 【多进程工程优化】
    # 3D 医学影像数据体积庞大，限制最大并行工作进程数，防止 OOM (内存溢出) 崩溃
    # 这里取 CPU 逻辑核心数的 75%，并且最大不超过 8 个进程
    cpu_count = os.cpu_count() or 4
    max_workers = min(max(1, int(cpu_count * 0.75)), 8)
    
    print(f"-> 启动多进程并发处理，分配工作进程数: {max_workers}")
    
    # 使用 ProcessPoolExecutor 绕过 GIL 限制，真正的并行执行 CPU 密集型任务
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # submit 将任务分发给进程池并返回 Future 对象列表
        futures = {executor.submit(process_single_file, path): path for path in label_files}
        
        # as_completed 迭代器使得任何一个任务完成后，立刻返回该任务
        # 使用 tqdm 包装这个迭代器即可正确、无阻塞地追踪整体进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="3D 预处理进度"):
            path = futures[future]
            try:
                # 获取结果可以抛出内部发生的异常，以便我们准确捕捉由于数据损坏或内存不足引起的崩溃
                result_name = future.result() 
            except Exception as exc:
                print(f"\n[错误] 处理文件 {os.path.basename(path)} 时发生异常: {exc}")

    print("\n预处理完成！")
    print(f"中心线图保存在: {OUT_REG_DIR}")
    print(f"边缘图保存在: {OUT_EDGE_DIR}")

if __name__ == "__main__":
    # OS 平台兼容保障: 尤其是 Windows 系统在使用 multiprocessing 时，必须在调用入口启动支持
    multiprocessing.freeze_support()
    run_preprocessing()