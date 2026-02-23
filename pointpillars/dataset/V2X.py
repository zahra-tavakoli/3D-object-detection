import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset
import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from pointpillars.utils import read_pickle, read_points
from .data_aug import point_range_filter, data_augment
from .kitti import BaseSampler


class V2X(Dataset):
    """V2X Dataset for PointPillars"""

    CLASSES = {
        'Pedestrian': 0,
        'Cyclist': 1,
        'Car': 2,
        'Motorcyclist': 3,
        'Trafficcone': 4
    }

    def __init__(self, data_root, split, pts_prefix='velodyne_reduced', 
                 use_reduced=True, load_image=False):
        """
        Args:
            data_root: Path to V2X dataset root
            split: 'train', 'val', or 'trainval'
            pts_prefix: Prefix for point cloud folder
            use_reduced: Whether to use reduced point clouds (within camera FOV)
            load_image: Whether to load images (for multimodal fusion)
        """
        assert split in ['train', 'val', 'trainval']
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix
        self.use_reduced = use_reduced
        self.load_image = load_image
        self.single_side = 'single-infrastructure-side'
        
        # Load data info
        info_path = os.path.join(data_root, f'v2x_infos_{split}.pkl')
        self.data_infos = read_pickle(info_path)
        self.sorted_ids = list(self.data_infos.keys())
        
        # V2X point cloud range (longer range than KITTI)
        self.point_cloud_range = [0, -40, -3, 100, 40, 1]
        
        # Load database info for augmentation
        db_infos_path = os.path.join(data_root, 'v2x_dbinfos_train.pkl')
        if os.path.exists(db_infos_path) and split in ['train', 'trainval']:
            db_infos = read_pickle(db_infos_path)
            db_infos = self.filter_db(db_infos)
        else:
            db_infos = {}
        
        # Create samplers for data augmentation
        db_sampler = {}
        for cat_name in self.CLASSES:
            if cat_name in db_infos and len(db_infos[cat_name]) > 0:
                db_sampler[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
        
        self.data_aug_config = dict(
            db_sampler=dict(
                db_sampler=db_sampler,
                sample_groups=dict(
                    Car=5, 
                    Pedestrian=3, 
                    Cyclist=3, 
                    Motorcyclist=2, 
                    Trafficcone=2
                )
            ),
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267]
            ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
            ),
            point_range_filter=self.point_cloud_range,
            object_range_filter=self.point_cloud_range
        )

    def filter_db(self, db_infos):
        """Filter database by difficulty and minimum points"""
        for k, v in list(db_infos.items()):
            db_infos[k] = [item for item in v if item['difficulty'] != -1]
        
        filter_thrs = dict(
            Car=5, 
            Pedestrian=5, 
            Cyclist=5, 
            Motorcyclist=5, 
            Trafficcone=3
        )
        for cat in self.CLASSES:
            if cat in db_infos:
                filter_thr = filter_thrs.get(cat, 5)
                db_infos[cat] = [
                    item for item in db_infos[cat] 
                    if item['num_points_in_gt'] >= filter_thr
                ]
        
        return db_infos

    def __len__(self):
        return len(self.sorted_ids)

    def __getitem__(self, index):
        sample_id = self.sorted_ids[index]
        data_info = self.data_infos[sample_id]
        
        image_info = data_info['image']
        calib_info = data_info['calib']
        annos_info = data_info.get('annos', None)
        
        # Load point cloud
        if self.use_reduced:
            id_str = str(sample_id).zfill(6)
            pts_path = os.path.join(
                self.data_root, 
                f'{self.single_side}-{self.pts_prefix}',
                f'{id_str}.bin'
            )
            pts = read_points(pts_path)
        else:
            velodyne_path = data_info['velodyne_path']
            pts_path = os.path.join(self.data_root, velodyne_path)
            pts = self._read_pcd(pts_path)
        
        # Prepare calibration info (ensure float32)
        calib_dict = {
            'P2': calib_info['P2'].astype(np.float32),
            'R0_rect': calib_info['R0_rect'].astype(np.float32),
            'Tr_velo_to_cam': calib_info['Tr_velo_to_cam'].astype(np.float32),
        }
        
        # Handle annotations
        if annos_info is not None and len(annos_info['name']) > 0:
            annos_name = np.array(annos_info['name'])
            location = np.array(annos_info['location'], dtype=np.float32)
            dimensions = np.array(annos_info['dimensions'], dtype=np.float32)  # [h, w, l]
            rotation_y = np.array(annos_info['rotation_y'], dtype=np.float32)
            difficulty = np.array(annos_info['difficulty'], dtype=np.int32)
            
            # V2X dimensions are [h, w, l], convert to [w, l, h] for model
            dims_wlh = np.stack([
                dimensions[:, 1],  # w
                dimensions[:, 2],  # l
                dimensions[:, 0],  # h
            ], axis=1)
            
            # V2X labels are already in LiDAR coordinates
            # Format: [x, y, z, w, l, h, yaw]
            gt_bboxes_3d = np.concatenate([
                location,
                dims_wlh,
                rotation_y[:, None]
            ], axis=1).astype(np.float32)
            
            gt_labels = np.array([
                self.CLASSES.get(name, -1) for name in annos_name
            ], dtype=np.int64)
            gt_names = annos_name
            
            # Filter valid objects
            valid_mask = gt_labels >= 0
            gt_bboxes_3d = gt_bboxes_3d[valid_mask]
            gt_labels = gt_labels[valid_mask]
            gt_names = gt_names[valid_mask]
            difficulty = difficulty[valid_mask]
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_names = np.array([])
            difficulty = np.array([], dtype=np.int32)
        
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': gt_labels,
            'gt_names': gt_names,
            'difficulty': difficulty,
            'image_info': image_info,
            'calib_info': calib_dict,
        }
        
        # Load image if needed
        if self.load_image:
            img = self._load_image(sample_id)
            data_dict['image'] = img
        
        # Apply data augmentation for training
        if self.split in ['train', 'trainval'] and len(gt_bboxes_3d) > 0:
            data_dict = data_augment(
                self.CLASSES, self.data_root, data_dict, self.data_aug_config
            )
        else:
            data_dict = point_range_filter(
                data_dict, point_range=self.data_aug_config['point_range_filter']
            )
        
        return data_dict

    def _read_pcd(self, path):
        """Read PCD file and return points with intensity"""
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.shape[0] > 0:
            intensity = np.ones((pts.shape[0], 1), dtype=np.float32)
            pts = np.hstack([pts, intensity])
        else:
            pts = np.zeros((0, 4), dtype=np.float32)
        return pts

    def _load_image(self, sample_id):
        """Load RGB image"""
        id_str = str(sample_id).zfill(6)
        img_path = os.path.join(
            self.data_root,
            f'{self.single_side}-image',
            f'{id_str}.jpg'
        )
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


if __name__ == '__main__':
    dataset = V2X(data_root='/data/zahra/v2x', split='train', load_image=True)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Points shape: {sample['pts'].shape}")
    print(f"GT bboxes shape: {sample['gt_bboxes_3d'].shape}")
    print(f"GT labels: {sample['gt_labels']}")
    if 'image' in sample:
        print(f"Image shape: {sample['image'].shape}")