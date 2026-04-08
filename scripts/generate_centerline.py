import os
import glob
import numpy as np
import nibabel as nib
import multiprocessing
from skimage.morphology import skeletonize_3d
from tqdm import tqdm

def process_single_file(args):
    img_path, output_path = args
    if os.path.exists(output_path):
        return  # Skip if already processed
        
    try:
        # Load NIfTI file
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata()
        
        # Initialize output skeleton array
        skeleton_combined = np.zeros_like(img_data, dtype=np.uint8)
        
        # We need to maintain categories for the skeleton.
        # Hepatic Vein is usually 1, Portal Vein is 2.
        # We find unique labels (excluding background 0)
        unique_labels = np.unique(img_data)
        
        for label_val in unique_labels:
            if label_val == 0:
                continue
                
            # Create boolean mask for the current class
            binary_mask = (img_data == label_val).astype(np.uint8)
            
            # Extract 3D skeleton for the class
            skeleton = skeletonize_3d(binary_mask)
            
            # Map the skeleton lines back to their original label value (e.g. 1 or 2)
            # skeletonize_3d returns 255 for foreground by default in some versions, or 1 in others.
            # Convert boolean/nonzero to logic:
            skeleton_binary = (skeleton > 0).astype(np.uint8)
            skeleton_combined[skeleton_binary == 1] = label_val
            
        # Save preprocessed nifti
        new_nii = nib.Nifti1Image(skeleton_combined, img_nii.affine, img_nii.header)
        nib.save(new_nii, output_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    base_dir = "./dataset"
    labels_dir = os.path.join(base_dir, "labelsTr")
    output_dir = os.path.join(base_dir, "preprocessed", "centerline_gt")
    
    os.makedirs(output_dir, exist_ok=True)
    
    label_files = glob.glob(os.path.join(labels_dir, "*.nii.gz"))
    if not label_files:
        print(f"No NIfTI files found in {labels_dir}")
        return
        
    print(f"Found {len(label_files)} label files. Starting skeletonization...")
    
    # Prepare arguments for multiprocessing
    tasks = []
    for f in label_files:
        basename = os.path.basename(f)
        out_f = os.path.join(output_dir, basename)
        tasks.append((f, out_f))
        
    # Use multi-processing to speed up skeletonization
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {num_cores} cores...")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap(process_single_file, tasks), total=len(tasks)))
        
    print("Optimization completed! Centerlines saved to:", output_dir)

if __name__ == "__main__":
    main()
