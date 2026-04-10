import os
import glob
import numpy as np
import nibabel as nib
import multiprocessing
from skimage.morphology import skeletonize_3d
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

def process_single_file(args):
    img_path, output_path, D_max = args
    if os.path.exists(output_path):
        return  # Skip if already processed
        
    try:
        # Load NIfTI file
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata()
        
        # 1. Unify all foreground labels (extract skeleton globally)
        binary_mask = (img_data > 0).astype(np.uint8)
        skeleton = skeletonize_3d(binary_mask)
        skeleton_binary = (skeleton > 0).astype(np.uint8)
        
        # 2. Distance Transform
        # EDT computes distance to zero-valued pixels. 
        # We invert the skeleton so skeleton=0, others=1
        skeleton_inv = 1 - skeleton_binary
        distance_map = distance_transform_edt(skeleton_inv)
        
        # 3. Truncated EDT (T-EDT): Clamp distance to D_max to suppress ghost centerlines in background
        distance_map = np.clip(distance_map, 0.0, D_max)
        
        # 4. Save as float32 since it is a continuous regression target
        new_nii = nib.Nifti1Image(distance_map.astype(np.float32), img_nii.affine, img_nii.header)
        nib.save(new_nii, output_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    base_dir = "./dataset"
    labels_dir = os.path.join(base_dir, "labelsTr")
    output_dir = os.path.join(base_dir, "preprocessed", "centerline_gt")
    D_max = 10.0 # T-EDT truncation maximum distance limit
    
    os.makedirs(output_dir, exist_ok=True)
    
    label_files = glob.glob(os.path.join(labels_dir, "*.nii.gz"))
    if not label_files:
        print(f"No NIfTI files found in {labels_dir}")
        return
        
    print(f"Found {len(label_files)} label files. Starting T-EDT Skeletonization...")
    
    # Prepare arguments for multiprocessing
    tasks = []
    for f in label_files:
        basename = os.path.basename(f)
        out_f = os.path.join(output_dir, basename)
        tasks.append((f, out_f, D_max))
        
    # Use multi-processing to speed up distance conversion
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {num_cores} cores...")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap(process_single_file, tasks), total=len(tasks)))
        
    print("Optimization completed! Truncated Distance Maps (T-EDT) saved to:", output_dir)

if __name__ == "__main__":
    main()
