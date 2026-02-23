import numpy as np
import torch
from torch.utils.data import DataLoader

def collate_fn(list_data):
    """Collate function for both LiDAR-only and multimodal data"""
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list, batched_names_list = [], []
    batched_difficulty_list = []
    batched_img_list, batched_calib_list = [], []
    batched_images = []  # For multimodal
    
    for data_dict in list_data:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']
        difficulty = data_dict['difficulty']
        image_info, calib_info = data_dict['image_info'], data_dict['calib_info']

        batched_pts_list.append(torch.from_numpy(pts))
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))
        batched_names_list.append(gt_names)
        batched_difficulty_list.append(torch.from_numpy(difficulty))
        batched_img_list.append(image_info)
        batched_calib_list.append(calib_info)
        
        # Handle images for multimodal
        if 'image' in data_dict:
            img = data_dict['image']
            # Convert to tensor: (H, W, C) -> (C, H, W), normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            batched_images.append(img)
    
    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_difficulty=batched_difficulty_list,
        batched_img_info=batched_img_list,
        batched_calib_info=batched_calib_list
    )
    
    # Add batched images if available
    if batched_images:
        rt_data_dict['batched_images'] = pad_images(batched_images)
    
    return rt_data_dict


def pad_images(images):
    """Pad images to the same size within a batch"""
    if len(images) == 0:
        return None
    
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    padded = []
    for img in images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
        padded.append(img)
    
    return torch.stack(padded, dim=0)


def get_dataloader(dataset, batch_size, num_workers, shuffle=True, drop_last=False):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    return dataloader