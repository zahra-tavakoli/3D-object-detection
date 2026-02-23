# evaluate_v2x.py
import argparse
import numpy as np
import os
import torch
from tqdm import tqdm
import json

from pointpillars.utils import setup_seed, read_pickle, write_pickle
from pointpillars.dataset.V2X import V2X
from pointpillars.dataset.dataloader import get_dataloader
from pointpillars.model import PointPillars, MultimodalPointPillars


def compute_iou_3d(box1, box2):
    """Compute 3D IoU between two boxes (simplified BEV IoU)"""
    # box format: [x, y, z, w, l, h, yaw]
    # Use BEV IoU for simplicity
    
    x1, y1, w1, l1 = box1[0], box1[1], box1[3], box1[4]
    x2, y2, w2, l2 = box2[0], box2[1], box2[3], box2[4]
    
    # Compute axis-aligned bounding box (ignoring rotation for simplicity)
    x1_min, x1_max = x1 - l1/2, x1 + l1/2
    y1_min, y1_max = y1 - w1/2, y1 + w1/2
    
    x2_min, x2_max = x2 - l2/2, x2 + l2/2
    y2_min, y2_max = y2 - w2/2, y2 + w2/2
    
    # Intersection
    xi_min = max(x1_min, x2_min)
    xi_max = min(x1_max, x2_max)
    yi_min = max(y1_min, y2_min)
    yi_max = min(y1_max, y2_max)
    
    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0
    
    inter_area = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Union
    area1 = l1 * w1
    area2 = l2 * w2
    union_area = area1 + area2 - inter_area
    
    return inter_area / (union_area + 1e-6)


def evaluate_detections(all_predictions, all_gt, classes, iou_thresholds=[0.5, 0.7]):
    """
    Compute AP for each class
    """
    results = {}
    
    for class_name, class_id in classes.items():
        for iou_thresh in iou_thresholds:
            # Collect all predictions and GT for this class
            all_scores = []
            all_tp = []
            total_gt = 0
            
            for sample_id in all_gt.keys():
                gt = all_gt[sample_id]
                pred = all_predictions.get(sample_id, {'lidar_bboxes': [], 'labels': [], 'scores': []})
                
                # Get GT for this class
                gt_mask = gt['labels'] == class_id
                gt_boxes = gt['bboxes'][gt_mask]
                total_gt += len(gt_boxes)
                
                # Get predictions for this class
                pred_mask = np.array(pred['labels']) == class_id
                if len(pred['labels']) == 0 or pred_mask.sum() == 0:
                    continue
                
                pred_boxes = np.array(pred['lidar_bboxes'])[pred_mask]
                pred_scores = np.array(pred['scores'])[pred_mask]
                
                # Sort by score
                sorted_idx = np.argsort(-pred_scores)
                pred_boxes = pred_boxes[sorted_idx]
                pred_scores = pred_scores[sorted_idx]
                
                # Match predictions to GT
                gt_matched = np.zeros(len(gt_boxes), dtype=bool)
                
                for pred_box, pred_score in zip(pred_boxes, pred_scores):
                    all_scores.append(pred_score)
                    
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_matched[gt_idx]:
                            continue
                        iou = compute_iou_3d(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_thresh and best_gt_idx >= 0:
                        all_tp.append(1)
                        gt_matched[best_gt_idx] = True
                    else:
                        all_tp.append(0)
            
            # Compute AP
            if total_gt == 0:
                ap = 0.0
            else:
                all_scores = np.array(all_scores)
                all_tp = np.array(all_tp)
                
                # Sort by score
                sorted_idx = np.argsort(-all_scores)
                all_tp = all_tp[sorted_idx]
                
                # Compute precision/recall
                tp_cumsum = np.cumsum(all_tp)
                fp_cumsum = np.cumsum(1 - all_tp)
                
                recalls = tp_cumsum / total_gt
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
                
                # Compute AP using 11-point interpolation
                ap = 0.0
                for t in np.arange(0, 1.1, 0.1):
                    if np.sum(recalls >= t) == 0:
                        p = 0
                    else:
                        p = np.max(precisions[recalls >= t])
                    ap += p / 11
            
            results[f'{class_name}_AP@{iou_thresh}'] = ap * 100
    
    # Compute mAP
    for iou_thresh in iou_thresholds:
        aps = [results[f'{cls}_AP@{iou_thresh}'] for cls in classes.keys() 
               if f'{cls}_AP@{iou_thresh}' in results]
        results[f'mAP@{iou_thresh}'] = np.mean(aps) if aps else 0.0
    
    return results


def main(args):
    setup_seed()
    
    # Dataset
    val_dataset = V2X(
        data_root=args.data_root,
        split='val',
        load_image=args.multimodal
    )
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    CLASSES = V2X.CLASSES
    nclasses = len(CLASSES)
    
    print(f"Evaluating on V2X val set: {len(val_dataset)} samples")
    
    # Model
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
    
    if not args.no_cuda:
        model = model.cuda()
        state_dict = torch.load(args.ckpt)
    else:
        state_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Collect predictions and GT
    all_predictions = {}
    all_gt = {}
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(val_dataloader)):
            # Move to GPU
            if not args.no_cuda:
                for key in data_dict:
                    if key in ['batched_pts', 'batched_gt_bboxes', 'batched_labels']:
                        for j in range(len(data_dict[key])):
                            if torch.is_tensor(data_dict[key][j]):
                                data_dict[key][j] = data_dict[key][j].cuda()
                    elif key == 'batched_images' and data_dict[key] is not None:
                        data_dict[key] = data_dict[key].cuda()
            
            batched_pts = data_dict['batched_pts']
            
            # Inference
            if args.multimodal and 'batched_images' in data_dict:
                results = model(
                    batched_pts=batched_pts,
                    mode='test',
                    batched_images=data_dict.get('batched_images'),
                    batched_calib=data_dict.get('batched_calib_info')
                )
            else:
                results = model(batched_pts=batched_pts, mode='test')
            
            # Store predictions and GT
            for j, result in enumerate(results):
                sample_idx = batch_idx * args.batch_size + j
                if sample_idx >= len(val_dataset):
                    break
                
                sample_id = val_dataset.sorted_ids[sample_idx]
                
                all_predictions[sample_id] = {
                    'lidar_bboxes': result['lidar_bboxes'],
                    'labels': result['labels'],
                    'scores': result['scores']
                }
                
                # Get GT
                gt_bboxes = data_dict['batched_gt_bboxes'][j].cpu().numpy()
                gt_labels = data_dict['batched_labels'][j].cpu().numpy()
                
                all_gt[sample_id] = {
                    'bboxes': gt_bboxes,
                    'labels': gt_labels
                }
    
    # Evaluate
    print("\nComputing metrics...")
    results = evaluate_detections(all_predictions, all_gt, CLASSES, 
                                   iou_thresholds=[0.5, 0.7])
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    
    for class_name in CLASSES.keys():
        ap50 = results.get(f'{class_name}_AP@0.5', 0)
        ap70 = results.get(f'{class_name}_AP@0.7', 0)
        print(f"{class_name:15s}: AP@0.5 = {ap50:6.2f}%, AP@0.7 = {ap70:6.2f}%")
    
    print("-" * 50)
    print(f"{'mAP':15s}: AP@0.5 = {results['mAP@0.5']:6.2f}%, AP@0.7 = {results['mAP@0.7']:6.2f}%")
    print("=" * 50)
    
    # Save results
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(args.save_path, 'eval_results.txt'), 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value:.2f}\n")
        
        # Save predictions
        write_pickle(all_predictions, os.path.join(args.save_path, 'predictions.pkl'))
        
        print(f"\nResults saved to {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate V2X Detection')
    parser.add_argument('--data_root', required=True, help='V2X dataset root')
    parser.add_argument('--ckpt', required=True, help='Checkpoint path')
    parser.add_argument('--save_path', default='eval_results', help='Save path')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--multimodal', action='store_true', help='Use multimodal model')
    parser.add_argument('--image_backbone', default='resnet18')
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
    main(args)