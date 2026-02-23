# test_multimodal.py
import argparse
import cv2
import numpy as np
import os
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from pointpillars.utils import setup_seed, read_points
from pointpillars.model import PointPillars, MultimodalPointPillars
from pointpillars.model.pointpillars import get_model_config


# V2X specific classes
V2X_CLASSES = {
    'Pedestrian': 0,
    'Cyclist': 1,
    'Car': 2,
    'Motorcyclist': 3,
    'Trafficcone': 4
}
V2X_LABEL2CLASS = {v: k for k, v in V2X_CLASSES.items()}

# Colors for visualization (RGB format for matplotlib, BGR for OpenCV)
COLORS_RGB = {
    0: (0, 1, 0),        # Pedestrian - Green
    1: (0, 0, 1),        # Cyclist - Blue
    2: (1, 0, 0),        # Car - Red
    3: (0, 1, 1),        # Motorcyclist - Cyan
    4: (1, 1, 0),        # Trafficcone - Yellow
    -1: (0.5, 0.5, 0.5)  # GT - Gray
}

COLORS_BGR = {
    0: (0, 255, 0),      # Pedestrian - Green
    1: (255, 0, 0),      # Cyclist - Blue
    2: (0, 0, 255),      # Car - Red
    3: (255, 255, 0),    # Motorcyclist - Cyan
    4: (0, 255, 255),    # Trafficcone - Yellow
    -1: (128, 128, 128)  # GT - Gray
}


def point_range_filter(pts, point_range):
    """Filter points within specified range"""
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    return pts[keep_mask]


def read_pcd(path):
    """Read PCD file"""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] > 0:
        intensity = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts = np.hstack([pts, intensity])
    else:
        pts = np.zeros((0, 4), dtype=np.float32)
    return pts


def read_v2x_calib(calib_dir, sample_id):
    """Read V2X calibration files"""
    id_str = str(sample_id).zfill(6)
    
    cam_intrinsic_path = os.path.join(calib_dir, 'camera_intrinsic', f'{id_str}.json')
    with open(cam_intrinsic_path, 'r') as f:
        cam_data = json.load(f)
    
    lidar2cam_path = os.path.join(calib_dir, 'virtuallidar_to_camera', f'{id_str}.json')
    with open(lidar2cam_path, 'r') as f:
        lidar2cam_data = json.load(f)
    
    cam_K = np.array(cam_data['cam_K']).reshape(3, 3)
    P2 = np.zeros((4, 4), dtype=np.float32)
    P2[:3, :3] = cam_K
    P2[3, 3] = 1.0
    
    rotation = np.array(lidar2cam_data['rotation'], dtype=np.float32)
    translation = np.array(lidar2cam_data['translation'], dtype=np.float32).flatten()
    
    Tr_velo_to_cam = np.eye(4, dtype=np.float32)
    Tr_velo_to_cam[:3, :3] = rotation
    Tr_velo_to_cam[:3, 3] = translation
    
    R0_rect = np.eye(4, dtype=np.float32)
    
    return {
        'P2': P2,
        'R0_rect': R0_rect,
        'Tr_velo_to_cam': Tr_velo_to_cam,
    }


def read_v2x_label(label_path):
    """Read V2X label file"""
    with open(label_path, 'r') as f:
        labels = json.load(f)
    
    if len(labels) == 0:
        return None
    
    names = []
    locations = []
    dimensions = []
    rotations = []
    bbox2d = []
    
    for obj in labels:
        names.append(obj['type'])
        
        loc = obj['3d_location']
        locations.append([float(loc['x']), float(loc['y']), float(loc['z'])])
        
        dims = obj['3d_dimensions']
        dimensions.append([float(dims['h']), float(dims['w']), float(dims['l'])])
        
        rotations.append(float(obj['rotation']))
        
        box2d = obj['2d_box']
        bbox2d.append([float(box2d['xmin']), float(box2d['ymin']), 
                       float(box2d['xmax']), float(box2d['ymax'])])
    
    return {
        'name': np.array(names),
        'location': np.array(locations, dtype=np.float32),
        'dimensions': np.array(dimensions, dtype=np.float32),
        'rotation_y': np.array(rotations, dtype=np.float32),
        'bbox2d': np.array(bbox2d, dtype=np.float32)
    }


def lidar_bbox_to_corners(bbox):
    """Convert lidar bbox [x, y, z, w, l, h, yaw] to 8 corners"""
    x, y, z, w, l, h, yaw = bbox
    
    corners = np.array([
        [-l/2, -w/2, 0],
        [l/2, -w/2, 0],
        [l/2, w/2, 0],
        [-l/2, w/2, 0],
        [-l/2, -w/2, h],
        [l/2, -w/2, h],
        [l/2, w/2, h],
        [-l/2, w/2, h]
    ], dtype=np.float32)
    
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    corners = corners @ R.T
    corners += np.array([x, y, z])
    
    return corners


def lidar_bbox_to_bev_corners(bbox):
    """Get BEV (Bird's Eye View) corners of bbox"""
    x, y, z, w, l, h, yaw = bbox
    
    corners = np.array([
        [-l/2, -w/2],
        [l/2, -w/2],
        [l/2, w/2],
        [-l/2, w/2],
    ], dtype=np.float32)
    
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ], dtype=np.float32)
    
    corners = corners @ R.T
    corners += np.array([x, y])
    
    return corners


def draw_3d_box_on_image(img, corners_2d, color, thickness=2):
    """Draw 3D bounding box on image"""
    corners_2d = corners_2d.astype(np.int32)
    
    # Draw bottom face
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, thickness)
    
    # Draw top face
    for i in range(4, 8):
        j = 4 + (i - 4 + 1) % 4
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
    
    # Draw vertical edges
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)
    
    return img


def visualize_on_image(img, pred_bboxes, pred_labels, pred_scores, 
                       calib_dict, gt_bboxes=None, gt_labels=None,
                       score_thresh=0.3):
    """Visualize predictions and GT on image"""
    img_vis = img.copy()
    h, w = img.shape[:2]
    
    P2 = calib_dict['P2']
    R0_rect = calib_dict['R0_rect']
    Tr_velo_to_cam = calib_dict['Tr_velo_to_cam']
    
    # Draw GT first (so predictions are on top)
    if gt_bboxes is not None and gt_labels is not None:
        for i, (bbox, label) in enumerate(zip(gt_bboxes, gt_labels)):
            corners = lidar_bbox_to_corners(bbox)
            corners_hom = np.hstack([corners, np.ones((8, 1))])
            
            corners_cam = corners_hom @ Tr_velo_to_cam.T
            corners_rect = corners_cam @ R0_rect.T
            corners_img = corners_rect @ P2.T
            
            depth = corners_img[:, 2:3]
            depth[depth <= 0] = 1e-6
            corners_2d = corners_img[:, :2] / depth
            
            if np.any(corners_2d[:, 0] < -100) or np.any(corners_2d[:, 0] >= w + 100):
                continue
            if np.any(corners_2d[:, 1] < -100) or np.any(corners_2d[:, 1] >= h + 100):
                continue
            if np.any(corners_rect[:, 2] <= 0):
                continue
            
            # Clip corners to image
            corners_2d[:, 0] = np.clip(corners_2d[:, 0], 0, w - 1)
            corners_2d[:, 1] = np.clip(corners_2d[:, 1], 0, h - 1)
            
            img_vis = draw_3d_box_on_image(img_vis, corners_2d, (128, 128, 128), thickness=1)
    
    # Draw predictions
    for i, (bbox, label, score) in enumerate(zip(pred_bboxes, pred_labels, pred_scores)):
        if score < score_thresh:
            continue
        
        corners = lidar_bbox_to_corners(bbox)
        corners_hom = np.hstack([corners, np.ones((8, 1))])
        
        corners_cam = corners_hom @ Tr_velo_to_cam.T
        corners_rect = corners_cam @ R0_rect.T
        corners_img = corners_rect @ P2.T
        
        depth = corners_img[:, 2:3]
        depth[depth <= 0] = 1e-6
        corners_2d = corners_img[:, :2] / depth
        
        if np.any(corners_2d[:, 0] < -100) or np.any(corners_2d[:, 0] >= w + 100):
            continue
        if np.any(corners_2d[:, 1] < -100) or np.any(corners_2d[:, 1] >= h + 100):
            continue
        if np.any(corners_rect[:, 2] <= 0):
            continue
        
        corners_2d[:, 0] = np.clip(corners_2d[:, 0], 0, w - 1)
        corners_2d[:, 1] = np.clip(corners_2d[:, 1], 0, h - 1)
        
        color = COLORS_BGR.get(label, (255, 255, 255))
        img_vis = draw_3d_box_on_image(img_vis, corners_2d, color, thickness=2)
        
        # Add label text
        class_name = V2X_LABEL2CLASS.get(label, 'Unknown')
        text = f'{class_name}: {score:.2f}'
        text_pos = (int(corners_2d[0, 0]), max(int(corners_2d[0, 1]) - 10, 20))
        cv2.putText(img_vis, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
    
    return img_vis


def visualize_bev(points, pred_bboxes, pred_labels, pred_scores,
                  gt_bboxes=None, gt_labels=None, score_thresh=0.3,
                  point_range=[0, -40, -3, 100, 40, 1]):
    """
    Create Bird's Eye View visualization and return as figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], s=0.1, c='gray', alpha=0.3)
    
    # Draw GT boxes
    if gt_bboxes is not None:
        for bbox in gt_bboxes:
            corners = lidar_bbox_to_bev_corners(bbox)
            polygon = Polygon(corners, fill=False, edgecolor='gray', 
                            linewidth=1, linestyle='--')
            ax.add_patch(polygon)
    
    # Draw prediction boxes
    for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
        if score < score_thresh:
            continue
        
        corners = lidar_bbox_to_bev_corners(bbox)
        color = COLORS_RGB.get(label, (1, 1, 1))
        polygon = Polygon(corners, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(polygon)
        
        # Add label
        class_name = V2X_LABEL2CLASS.get(label, 'Unk')
        ax.text(bbox[0], bbox[1] + 2, f'{class_name}:{score:.2f}', 
                fontsize=8, color=color)
    
    # Set axis
    ax.set_xlim(point_range[0], point_range[3])
    ax.set_ylim(point_range[1], point_range[4])
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Bird\'s Eye View Detection Result')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = []
    for label, name in V2X_LABEL2CLASS.items():
        color = COLORS_RGB.get(label, (1, 1, 1))
        legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, label=name))
    legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=1, 
                                       linestyle='--', label='Ground Truth'))
    ax.legend(handles=legend_elements, loc='upper right')
    
    return fig


def visualize_3d_pointcloud(points, pred_bboxes, pred_labels, pred_scores,
                            gt_bboxes=None, gt_labels=None, score_thresh=0.3,
                            save_path=None):
    """
    Create 3D point cloud visualization and save to file
    """
    import open3d as o3d
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Color points by height
    heights = points[:, 2]
    heights_normalized = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
    colors = plt.cm.viridis(heights_normalized)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = [pcd]
    
    # Add prediction bboxes
    for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
        if score < score_thresh:
            continue
        
        corners = lidar_bbox_to_corners(bbox)
        
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        color = COLORS_RGB.get(label, (1, 1, 1))
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
        
        geometries.append(line_set)
    
    # Add GT bboxes
    if gt_bboxes is not None:
        for bbox in gt_bboxes:
            corners = lidar_bbox_to_corners(bbox)
            
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(lines))
            
            geometries.append(line_set)
    
    # Save to PLY file (can be viewed in MeshLab, CloudCompare, etc.)
    if save_path:
        # Merge all geometries for saving
        merged_pcd = o3d.geometry.PointCloud()
        all_points = [np.asarray(pcd.points)]
        all_colors = [np.asarray(pcd.colors)]
        
        # Add bbox corners as points with colors
        for geom in geometries[1:]:
            if isinstance(geom, o3d.geometry.LineSet):
                pts = np.asarray(geom.points)
                cols = np.asarray(geom.colors)
                if len(cols) > 0:
                    # Repeat color for each point
                    cols_expanded = np.tile(cols[0], (len(pts), 1))
                    all_points.append(pts)
                    all_colors.append(cols_expanded)
        
        merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        
        ply_path = save_path.replace('.png', '.ply')
        o3d.io.write_point_cloud(ply_path, merged_pcd)
        print(f"Saved 3D point cloud to {ply_path}")
    
    return geometries


def create_combined_visualization(img_vis, bev_fig, save_path):
    """Combine image and BEV visualization into one figure"""
    # Convert BEV figure to image
    bev_fig.canvas.draw()
    bev_img = np.frombuffer(bev_fig.canvas.tostring_rgb(), dtype=np.uint8)
    bev_img = bev_img.reshape(bev_fig.canvas.get_width_height()[::-1] + (3,))
    
    # Resize to match heights
    h_img = img_vis.shape[0]
    h_bev = bev_img.shape[0]
    
    if h_img != h_bev:
        scale = h_img / h_bev
        new_w = int(bev_img.shape[1] * scale)
        bev_img = cv2.resize(bev_img, (new_w, h_img))
    
    # Convert BEV from RGB to BGR
    bev_img_bgr = cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR)
    
    # Combine horizontally
    combined = np.hstack([img_vis, bev_img_bgr])
    
    cv2.imwrite(save_path, combined)
    print(f"Saved combined visualization to {save_path}")
    
    return combined


def main(args):
    # Create save directory
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
    
    print(f"Loading model from {args.ckpt}")
    
    # Point cloud range for V2X
    point_range = [0, -40, -3, 100, 40, 1]
    
    # Create model
    nclasses = len(V2X_CLASSES)
    
    if args.multimodal:
        model = MultimodalPointPillars(
            nclasses=nclasses,
            dataset='v2x',
            image_backbone=args.image_backbone,
            image_dim=64,
            pretrained=False
        )
    else:
        model = PointPillars(nclasses=nclasses, dataset='v2x')
    
    # Load weights
    if not args.no_cuda:
        model = model.cuda()
        state_dict = torch.load(args.ckpt)
    else:
        state_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Read data
    print(f"Loading point cloud from {args.pc_path}")
    if args.pc_path.endswith('.pcd'):
        pc = read_pcd(args.pc_path)
    else:
        pc = read_points(args.pc_path)
    
    pc_original = pc.copy()
    pc = point_range_filter(pc, point_range)
    print(f"Points after filtering: {pc.shape[0]}")
    
    # Read calibration
    calib_dict = None
    if args.calib_dir and args.sample_id is not None:
        print(f"Loading calibration for sample {args.sample_id}")
        calib_dict = read_v2x_calib(args.calib_dir, args.sample_id)
    
    # Read image
    img = None
    if args.img_path and os.path.exists(args.img_path):
        print(f"Loading image from {args.img_path}")
        img = cv2.imread(args.img_path)
    
    # Read GT label
    gt_label = None
    if args.gt_path and os.path.exists(args.gt_path):
        print(f"Loading GT from {args.gt_path}")
        gt_label = read_v2x_label(args.gt_path)
    
    # Prepare input
    pc_torch = torch.from_numpy(pc).float()
    if not args.no_cuda:
        pc_torch = pc_torch.cuda()
    
    # Prepare image for multimodal
    img_tensor = None
    if args.multimodal and img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        if not args.no_cuda:
            img_tensor = img_tensor.cuda()
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        if args.multimodal and img_tensor is not None and calib_dict is not None:
            results = model(
                batched_pts=[pc_torch],
                mode='test',
                batched_images=img_tensor,
                batched_calib=[calib_dict]
            )
        else:
            results = model(batched_pts=[pc_torch], mode='test')
    
    result = results[0]
    
    # Get predictions
    pred_bboxes = result['lidar_bboxes']
    pred_labels = result['labels']
    pred_scores = result['scores']
    
    # Print predictions
    print(f"\n{'='*50}")
    print("PREDICTIONS")
    print(f"{'='*50}")
    print(f"Number of detections: {len(pred_bboxes)}")
    
    filtered_count = 0
    for i, (bbox, label, score) in enumerate(zip(pred_bboxes, pred_labels, pred_scores)):
        if score >= args.score_thresh:
            class_name = V2X_LABEL2CLASS.get(label, 'Unknown')
            print(f"  [{filtered_count}] {class_name}: score={score:.3f}, "
                  f"pos=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}), "
                  f"size=({bbox[3]:.1f}, {bbox[4]:.1f}, {bbox[5]:.1f})")
            filtered_count += 1
    print(f"Detections above threshold ({args.score_thresh}): {filtered_count}")
    
    # Prepare GT bboxes
    gt_bboxes_lidar = None
    gt_labels_np = None
    if gt_label is not None:
        print(f"\n{'='*50}")
        print("GROUND TRUTH")
        print(f"{'='*50}")
        print(f"Number of GT objects: {len(gt_label['name'])}")
        
        gt_bboxes_lidar = []
        gt_labels_np = []
        for i in range(len(gt_label['name'])):
            name = gt_label['name'][i]
            loc = gt_label['location'][i]
            dims = gt_label['dimensions'][i]
            rot = gt_label['rotation_y'][i]
            
            bbox = [loc[0], loc[1], loc[2], dims[1], dims[2], dims[0], rot]
            gt_bboxes_lidar.append(bbox)
            gt_labels_np.append(V2X_CLASSES.get(name, -1))
            print(f"  [{i}] {name}: pos=({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f})")
        
        gt_bboxes_lidar = np.array(gt_bboxes_lidar, dtype=np.float32)
        gt_labels_np = np.array(gt_labels_np)
    
    # === SAVE VISUALIZATIONS ===
    print(f"\n{'='*50}")
    print("SAVING VISUALIZATIONS")
    print(f"{'='*50}")
    
    save_dir = args.save_path if args.save_path else './test_results'
    os.makedirs(save_dir, exist_ok=True)
    
    sample_name = os.path.splitext(os.path.basename(args.pc_path))[0]
    
    # 1. Save BEV visualization
    print("Creating BEV visualization...")
    bev_fig = visualize_bev(
        pc, pred_bboxes, pred_labels, pred_scores,
        gt_bboxes_lidar, gt_labels_np,
        score_thresh=args.score_thresh,
        point_range=point_range
    )
    bev_path = os.path.join(save_dir, f'{sample_name}_bev.png')
    bev_fig.savefig(bev_path, dpi=150, bbox_inches='tight')
    print(f"  Saved BEV to: {bev_path}")
    plt.close(bev_fig)
    
    # 2. Save image visualization (if image and calib available)
    if img is not None and calib_dict is not None:
        print("Creating image visualization...")
        img_vis = visualize_on_image(
            img, pred_bboxes, pred_labels, pred_scores,
            calib_dict, gt_bboxes_lidar, gt_labels_np,
            score_thresh=args.score_thresh
        )
        img_path = os.path.join(save_dir, f'{sample_name}_image.png')
        cv2.imwrite(img_path, img_vis)
        print(f"  Saved image to: {img_path}")
        
        # 3. Save combined visualization
        print("Creating combined visualization...")
        bev_fig2 = visualize_bev(
            pc, pred_bboxes, pred_labels, pred_scores,
            gt_bboxes_lidar, gt_labels_np,
            score_thresh=args.score_thresh,
            point_range=point_range
        )
        combined_path = os.path.join(save_dir, f'{sample_name}_combined.png')
        create_combined_visualization(img_vis, bev_fig2, combined_path)
        plt.close(bev_fig2)
    
    # 4. Save 3D point cloud (PLY format)
    print("Creating 3D point cloud file...")
    ply_path = os.path.join(save_dir, f'{sample_name}_3d.ply')
    visualize_3d_pointcloud(
        pc, pred_bboxes, pred_labels, pred_scores,
        gt_bboxes_lidar, gt_labels_np,
        score_thresh=args.score_thresh,
        save_path=ply_path
    )
    
    # 5. Save detection results to JSON
    results_json = {
        'sample': sample_name,
        'num_points': int(pc.shape[0]),
        'predictions': [],
        'ground_truth': []
    }
    
    for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
        if score >= args.score_thresh:
            results_json['predictions'].append({
                'class': V2X_LABEL2CLASS.get(int(label), 'Unknown'),
                'score': float(score),
                'bbox': bbox.tolist()
            })
    
    if gt_label is not None:
        for i in range(len(gt_label['name'])):
            results_json['ground_truth'].append({
                'class': gt_label['name'][i],
                'location': gt_label['location'][i].tolist(),
                'dimensions': gt_label['dimensions'][i].tolist(),
                'rotation': float(gt_label['rotation_y'][i])
            })
    
    json_path = os.path.join(save_dir, f'{sample_name}_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved results JSON to: {json_path}")
    
    print(f"\n{'='*50}")
    print(f"All visualizations saved to: {save_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Multimodal PointPillars')
    parser.add_argument('--ckpt', required=True, help='Checkpoint path')
    parser.add_argument('--pc_path', required=True, help='Point cloud path (.bin or .pcd)')
    parser.add_argument('--img_path', default='', help='Image path')
    parser.add_argument('--calib_dir', default='', help='Calibration directory')
    parser.add_argument('--gt_path', default='', help='Ground truth label path')
    parser.add_argument('--sample_id', type=int, default=None, help='Sample ID for calibration')
    parser.add_argument('--save_path', default='test_results', help='Path to save visualizations')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--multimodal', action='store_true', help='Use multimodal model')
    parser.add_argument('--image_backbone', default='resnet18', help='Image backbone')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    main(args)