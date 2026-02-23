import argparse
import os
import numpy as np
import torch
import open3d as o3d
from pointpillars.model import PointPillars
from pointpillars.utils import keep_bbox_from_lidar_range

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    """Filter point cloud within range"""
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    return pts[keep_mask]

def save_boxes_to_txt(lidar_bboxes, labels, scores, save_path):
    """Save predicted boxes, labels, and scores to a text file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("# x y z dx dy dz yaw label score\n")
        for i in range(len(lidar_bboxes)):
            box = lidar_bboxes[i]
            label = int(labels[i])
            score = float(scores[i])
            f.write(" ".join(map(str, box.tolist())) + f" {label} {score:.4f}\n")
    print(f"[INFO] Saved predicted boxes to: {save_path}")

def main(args):
    CLASSES = {
        "Pedestrian": 0,
        "Cyclist": 1,
        "Car": 2,
        "Motorcyclist": 3,
        "Trafficcone": 4
    }

    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    # Load model
    print(f"[INFO] Loading model from {args.ckpt}")
    model = PointPillars(nclasses=len(CLASSES))
    if not args.no_cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    # Load point cloud
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError(f"Point cloud not found: {args.pc_path}")

    pc = o3d.io.read_point_cloud(args.pc_path)
    pc = np.asarray(pc.points, dtype=np.float32)
    if pc.shape[1] == 3:
        pc = np.hstack([pc, np.zeros((pc.shape[0], 1), dtype=np.float32)])

    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc)
    if not args.no_cuda:
        pc_torch = pc_torch.cuda()

    # Inference
    print("[INFO] Running inference...")
    with torch.no_grad():
        result = model(batched_pts=[pc_torch], mode="test")[0]
        print("[DEBUG] Model output keys:", result.keys())

    # Filter predictions
    result = keep_bbox_from_lidar_range(result, pcd_limit_range)
    lidar_bboxes = result["lidar_bboxes"]
    labels = result["labels"]
    scores = result["scores"]

    # Save predictions
    os.makedirs("predictions", exist_ok=True)
    txt_path = os.path.join("predictions", os.path.basename(args.pc_path).replace(".pcd", "_boxes.txt"))
    save_boxes_to_txt(lidar_bboxes, labels, scores, txt_path)

    print("[INFO] Inference complete. Bounding boxes saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2X PointPillars Test Script (Headless Mode)")
    parser.add_argument("--ckpt", default="pillar_logs/checkpoints/epoch_40.pth", help="Checkpoint path")
    parser.add_argument("--pc_path", required=True, help="Path to .pcd point cloud file")
    parser.add_argument("--no_cuda", action="store_true", help="Run without CUDA")
    args = parser.parse_args()
    main(args)
