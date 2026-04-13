import os
import json
import random
import argparse
import numpy as np
import nibabel as nib
import multiprocessing
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from skimage.morphology import skeletonize_3d
from tqdm import tqdm

def find_topological_nodes(skeleton_map):
    """ Extract Endpoints and Bifurcations using 3D 26-connectivity summation """
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    neighbor_counts = ndimage.convolve(skeleton_map.astype(np.uint8), kernel, mode='constant', cval=0)
    
    valid_mask = skeleton_map > 0
    # Endpoints naturally have 2 in the 3x3x3 sum (the voxel itself + 1 neighbor)
    endpoints_mask = valid_mask & (neighbor_counts == 2)
    # Bifurcations have >= 4 (voxel itself + >= 3 branching neighbors)
    bifurcations_mask = valid_mask & (neighbor_counts >= 4)
    
    return endpoints_mask, bifurcations_mask

def filter_close_points(points, min_dist=12.0):
    """ Prune clustered topological points to avoid tearing intersecting splines """
    if len(points) == 0:
        return points
    keep = [points[0]]
    for p in points[1:]:
        dists = np.linalg.norm(np.array(keep) - p, axis=1)
        if np.min(dists) >= min_dist:
            keep.append(p)
    return np.array(keep)

def generate_anchors(shape, step=64):
    """ Generate 6-Face dense bounding anchors fixing the boundaries explicitly """
    D, H, W = shape
    anchors = []
    # Z faces
    for h in range(0, H, step):
        for w in range(0, W, step):
            anchors.extend([[0, h, w], [D-1, h, w]])
    # Y faces
    for d in range(0, D, step):
        for w in range(0, W, step):
            anchors.extend([[d, 0, w], [d, H-1, w]])
    # X faces
    for d in range(0, D, step):
        for h in range(0, H, step):
            anchors.extend([[d, h, 0], [d, h, W-1]])
            
    anchors = np.array(anchors)
    return np.unique(anchors, axis=0)

def generate_tps_mapping_fast(shape, src_pts, tgt_pts, downsample=4):
    """ 
    Solve 3D Thin-Plate Spline. 
    Inverse evaluation: Fit Target->Displacement, then map X -> X - Disp.
    Evaluated stochastically on a subgrid and linearly interpolated to prevent memory explosion.
    """
    D, H, W = shape
    displacements = tgt_pts - src_pts
    
    # Fit RBF: "Given spatial location at TGT, what was the displacement from SRC?"
    rbf = RBFInterpolator(tgt_pts, displacements, kernel='thin_plate_spline')
    
    grid_d = np.arange(0, D, downsample)
    grid_h = np.arange(0, H, downsample)
    grid_w = np.arange(0, W, downsample)
    
    d_mesh, h_mesh, w_mesh = np.meshgrid(grid_d, grid_h, grid_w, indexing='ij')
    eval_pts = np.vstack([d_mesh.ravel(), h_mesh.ravel(), w_mesh.ravel()]).T
    
    # RBF takes ~1-2 seconds on the downsampled lattice (vs. Out-Of-Memory on 30 Million dense grid)
    pred_disp = rbf(eval_pts)
    
    disp_d = pred_disp[:, 0].reshape(len(grid_d), len(grid_h), len(grid_w))
    disp_h = pred_disp[:, 1].reshape(len(grid_d), len(grid_h), len(grid_w))
    disp_w = pred_disp[:, 2].reshape(len(grid_d), len(grid_h), len(grid_w))
    
    # Fast GPU-style zoom back to literal physical 1x1x1 matrix scale
    zoom_factors = (D / len(grid_d), H / len(grid_h), W / len(grid_w))
    
    fd = ndimage.zoom(disp_d, zoom_factors, order=1, mode='nearest')[:D, :H, :W]
    fh = ndimage.zoom(disp_h, zoom_factors, order=1, mode='nearest')[:D, :H, :W]
    fw = ndimage.zoom(disp_w, zoom_factors, order=1, mode='nearest')[:D, :H, :W]
    
    full_d, full_h, full_w = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    
    map_d = full_d - fd
    map_h = full_h - fh
    map_w = full_w - fw
    
    return np.stack([map_d, map_h, map_w])

def process_single_patient(args):
    img_path, lbl_path, out_img, out_lbl, out_cen = args
    if os.path.exists(out_cen): return [] # Skip already fully generated
    
    try:
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path)
        img_data = img_nii.get_fdata()
        lbl_data = lbl_nii.get_fdata()
        
        # 1. Analyze topology
        binary_mask = (lbl_data > 0).astype(np.uint8)
        skeleton = skeletonize_3d(binary_mask)
        
        endpoints, bifurcations = find_topological_nodes(skeleton)
        ep_coords = filter_close_points(np.argwhere(endpoints), min_dist=10.0)
        bf_coords = filter_close_points(np.argwhere(bifurcations), min_dist=12.0)
        
        # Shuffle explicitly 
        rng = np.random.default_rng() # Local safe rng to prevent multiprocess collisions
        rng.shuffle(ep_coords)
        rng.shuffle(bf_coords)
        
        sel_ep = ep_coords[:min(len(ep_coords), 15)]
        sel_bf = bf_coords[:min(len(bf_coords), 25)]
        
        # 2. Build Structural Maps
        src_pts, tgt_pts = [], []
        
        for p in sel_bf:
            src_pts.append(p)
            tgt_pts.append(p + rng.uniform(-6, 8, size=(3,)))   # Aggressive bifurcation twist
            
        for p in sel_ep:
            src_pts.append(p)
            tgt_pts.append(p + rng.uniform(-4, 6, size=(3,)))   # Subdued Endpoint elongation (keep within liver)
            
        anchors = generate_anchors(skeleton.shape, step=24)
        for p in anchors:
            src_pts.append(p)
            tgt_pts.append(p)
            
        src_pts, tgt_pts = np.array(src_pts), np.array(tgt_pts)
        
        # 3. Geometric Space Warping
        mapping_field = generate_tps_mapping_fast(skeleton.shape, src_pts, tgt_pts, downsample=4)
        
        # Bicubic (smooth) for CT array, Nearest (discrete) for labels
        warped_img = ndimage.map_coordinates(img_data, mapping_field, order=3, mode='nearest')
        warped_lbl = ndimage.map_coordinates(lbl_data, mapping_field, order=0, mode='nearest')
        
        # 4. Generate the corresponding Centerline (T-EDT Regression Map) dynamically
        new_skel = skeletonize_3d((warped_lbl > 0).astype(np.uint8))
        skel_inv = 1 - (new_skel > 0).astype(np.uint8)
        distance_map = np.clip(ndimage.distance_transform_edt(skel_inv), 0.0, 10.0)
        
        # 5. Flush to Disk
        nib.save(nib.Nifti1Image(warped_img.astype(np.float32), img_nii.affine), out_img)
        nib.save(nib.Nifti1Image(warped_lbl.astype(np.uint8), lbl_nii.affine), out_lbl)
        nib.save(nib.Nifti1Image(distance_map.astype(np.float32), lbl_nii.affine), out_cen)
        
        # Return logical object for the appended JSON list
        return {
            "image": os.path.relpath(out_img, "dataset"),
            "label": os.path.relpath(out_lbl, "dataset"),
            "centerline": os.path.relpath(out_cen, "dataset")
        }
        
    except Exception as e:
        print(f"Error on {img_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="dataset/dataset_full_thin.json")
    parser.add_argument("--out_json", type=str, default="dataset/dataset_full_aug.json")
    parser.add_argument("--clones", type=int, default=2, help="Number of clones per training image")
    args = parser.parse_args()
    
    with open(args.json, 'r') as f:
        data_info = json.load(f)
        
    all_files = data_info.get("training", [])
    
    # REPLICATE EXACT SEED LOCK OF THE ORIGINAL data_utils.py VALIDATION HOLD-OUT
    random.seed(42)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.8)
    
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"🔏 Seed-Lock Established: {len(train_files)} files safe to Augment. {len(val_files)} isolated as Validation.")
    
    tasks = []
    for item in train_files:
        for i in range(args.clones):
            in_img = os.path.join("dataset", item["image"])
            in_lbl = os.path.join("dataset", item["label"])
            
            # Formulate outbound naming templates natively into the same directories
            base_dir = os.path.dirname(in_img)
            name, ext = os.path.basename(in_img).split('.', 1)
            
            out_img = os.path.join(base_dir, f"{name}_tps_{i}.{ext}")
            out_lbl = os.path.join(os.path.dirname(in_lbl), f"{name}_tps_{i}.{ext}")
            out_cen = os.path.join("dataset", "preprocessed", "centerline_gt", f"{name}_tps_{i}.{ext}")
            
            tasks.append((in_img, in_lbl, out_img, out_lbl, out_cen))
            
    print(f"⏳ Generating {len(tasks)} highly distorted Topology-Safe clones...")
    
    cores = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(processes=cores) as pool:
        results = list(tqdm(pool.imap(process_single_patient, tasks), total=len(tasks)))
        
    flat_results = [r for r in results if r]
    
    print(f"✅ Generated {len(flat_results)} viable TPS clones.")
    
    # 6. Compose strict explicit isolated dataset JSON
    new_json = {
        "training": train_files + flat_results,
        "validation": val_files
    }
    
    with open(args.out_json, "w") as f:
        json.dump(new_json, f, indent=4)
        
    print(f"🎉 Architecture Complete. Clones indexed permanently in '{args.out_json}'")
    print(f"Update data_utils.py or config_cga_unet.yaml to load '{args.out_json}' to use the explicit Split.")

if __name__ == "__main__":
    main()
