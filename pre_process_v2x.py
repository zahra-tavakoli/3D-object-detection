import os
import json
import numpy as np
import sys

CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR)

from pre_process import BasePreprocessor
from pointpillars.utils import (
    points_in_bboxes_v2_lidar, 
    get_points_num_in_bbox_lidar, 
    write_points, 
    remove_outside_points
)

try:
    import open3d as o3d
except ImportError:
    print("Warning: open3d not installed. PCD reading will fail.")
    o3d = None


class V2XPreprocessor(BasePreprocessor):
    """V2X (DAIR-V2X) dataset preprocessor for single-infrastructure-side"""
    
    # === IMPORTANT: V2X uses z as CENTER, we convert to BOTTOM ===
    Z_CENTER_TO_BOTTOM = True
    
    def __init__(self, data_root, prefix='v2x'):
        super().__init__(data_root, prefix)
        self.single_side = 'single-infrastructure-side'

    def get_ids(self, data_type):
        """Get list of sample IDs from split_data.json"""
        split_file = os.path.join(self.data_root, self.single_side, 'split_data.json')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        ids = split_data.get(data_type, [])
        return ids
    
    def get_split_folder(self, data_type):
        """V2X doesn't have separate training/testing folders"""
        return self.single_side
    
    def format_sample_id(self, sample_id):
        """Format sample ID to 6-digit string"""
        return str(sample_id).zfill(6)
    
    def get_image_path(self, sample_id, split):
        id_str = self.format_sample_id(sample_id)
        return os.path.join(self.data_root, f'{self.single_side}-image', f'{id_str}.jpg')
    
    def get_lidar_path(self, sample_id, split):
        id_str = self.format_sample_id(sample_id)
        return os.path.join(self.data_root, f'{self.single_side}-velodyne', f'{id_str}.pcd')
    
    def get_calib_paths(self, sample_id, split):
        id_str = self.format_sample_id(sample_id)
        calib_dir = os.path.join(self.data_root, self.single_side, 'calib')
        return {
            'camera_intrinsic': os.path.join(calib_dir, 'camera_intrinsic', f'{id_str}.json'),
            'virtuallidar_to_camera': os.path.join(calib_dir, 'virtuallidar_to_camera', f'{id_str}.json')
        }
    
    def get_label_path(self, sample_id, split):
        id_str = self.format_sample_id(sample_id)
        return os.path.join(self.data_root, self.single_side, 'label', 'virtuallidar', f'{id_str}.json')
    
    def get_reduced_points_folder(self, split):
        return os.path.join(self.data_root, f'{self.single_side}-velodyne_reduced')
    
    def get_relative_velodyne_path(self, lidar_path):
        return self.sep.join(lidar_path.split(self.sep)[-2:])
    
    def get_relative_image_path(self, img_path):
        return self.sep.join(img_path.split(self.sep)[-2:])
    
    def read_points(self, path):
        """Read PCD point cloud file"""
        if o3d is None:
            raise ImportError("open3d is required to read PCD files")
        
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points, dtype=np.float32)
        
        if len(points) > 0:
            if pcd.has_colors():
                colors = np.asarray(pcd.colors, dtype=np.float32)
                intensity = np.mean(colors, axis=1, keepdims=True)
            else:
                intensity = np.ones((points.shape[0], 1), dtype=np.float32)
            points = np.hstack([points, intensity])
        else:
            points = np.zeros((0, 4), dtype=np.float32)
        
        return points
    
    def read_calib(self, paths):
        """Read V2X calibration files and build standard calibration dict"""
        with open(paths['camera_intrinsic'], 'r') as f:
            cam_data = json.load(f)
        
        with open(paths['virtuallidar_to_camera'], 'r') as f:
            lidar2cam_data = json.load(f)
        
        # Build P2 projection matrix (3x4 -> 4x4)
        cam_K = np.array(cam_data['cam_K']).reshape(3, 3)
        P2 = np.zeros((4, 4), dtype=np.float32)
        P2[:3, :3] = cam_K
        P2[2, 2] = 1.0
        P2[3, 3] = 1.0
        
        if 'P' in cam_data:
            P_data = np.array(cam_data['P']).reshape(3, 4)
            P2 = np.zeros((4, 4), dtype=np.float32)
            P2[:3, :] = P_data
            P2[3, 3] = 1.0
        
        # Build Tr_velo_to_cam (4x4)
        rotation = np.array(lidar2cam_data['rotation'], dtype=np.float32)
        translation = np.array(lidar2cam_data['translation'], dtype=np.float32).flatten()
        
        Tr_velo_to_cam = np.eye(4, dtype=np.float32)
        Tr_velo_to_cam[:3, :3] = rotation
        Tr_velo_to_cam[:3, 3] = translation
        
        # R0_rect is identity for V2X
        R0_rect = np.eye(4, dtype=np.float32)
        
        return {
            'P2': P2,
            'R0_rect': R0_rect,
            'Tr_velo_to_cam': Tr_velo_to_cam,
            'cam_K': cam_K,
            'cam_D': np.array(cam_data.get('cam_D', [0, 0, 0, 0, 0]), dtype=np.float32),
        }
    
    def read_label(self, path):
        """
        Read V2X JSON label file (virtuallidar format - already in LiDAR coords)
        
        IMPORTANT: V2X uses z as CENTER of box. We convert to BOTTOM (KITTI convention).
        z_bottom = z_center - h/2
        """
        with open(path, 'r') as f:
            labels = json.load(f)
        
        if len(labels) == 0:
            return {
                'name': np.array([]),
                'truncated': np.array([], dtype=np.float32),
                'occluded': np.array([], dtype=np.int32),
                'alpha': np.array([], dtype=np.float32),
                'bbox': np.zeros((0, 4), dtype=np.float32),
                'dimensions': np.zeros((0, 3), dtype=np.float32),
                'location': np.zeros((0, 3), dtype=np.float32),
                'rotation_y': np.array([], dtype=np.float32),
            }
        
        names = []
        truncated = []
        occluded = []
        alpha = []
        bbox = []
        dimensions = []
        location = []
        rotation_y = []
        
        for obj in labels:
            obj_type = obj['type']
            names.append(obj_type)
            truncated.append(float(obj.get('truncated_state', 0)))
            occluded.append(int(obj.get('occluded_state', 0)))
            alpha.append(float(obj.get('alpha', 0)))
            
            # 2D bounding box
            box2d = obj['2d_box']
            bbox.append([
                float(box2d['xmin']),
                float(box2d['ymin']),
                float(box2d['xmax']),
                float(box2d['ymax'])
            ])
            
            # 3D dimensions (h, w, l in V2X format)
            dims = obj['3d_dimensions']
            h = float(dims['h'])
            w = float(dims['w'])
            l = float(dims['l'])
            dimensions.append([h, w, l])
            
            # 3D location
            loc = obj['3d_location']
            x = float(loc['x'])
            y = float(loc['y'])
            z_center = float(loc['z'])
            
            # === KEY FIX: Convert z from CENTER to BOTTOM ===
            if self.Z_CENTER_TO_BOTTOM:
                z_bottom = z_center - h / 2.0
                location.append([x, y, z_bottom])
            else:
                location.append([x, y, z_center])
            
            rotation_y.append(float(obj['rotation']))
        
        return {
            'name': np.array(names),
            'truncated': np.array(truncated, dtype=np.float32),
            'occluded': np.array(occluded, dtype=np.int32),
            'alpha': np.array(alpha, dtype=np.float32),
            'bbox': np.array(bbox, dtype=np.float32),
            'dimensions': np.array(dimensions, dtype=np.float32),  # h, w, l
            'location': np.array(location, dtype=np.float32),  # x, y, z (BOTTOM)
            'rotation_y': np.array(rotation_y, dtype=np.float32),
        }
    
    def get_reduced_points(self, points, calib_dict, image_shape):
        """Remove points outside image FOV"""        
        return remove_outside_points(
            points=points,
            r0_rect=calib_dict['R0_rect'],
            tr_velo_to_cam=calib_dict['Tr_velo_to_cam'],
            P2=calib_dict['P2'],
            image_shape=image_shape
        )
    
    def get_points_num_in_bbox(self, points, calib_dict, annotation_dict):
        """Get number of points in each bbox (V2X format - LiDAR coords)"""
        if len(annotation_dict['name']) == 0:
            return np.array([], dtype=np.int32)
        
        return get_points_num_in_bbox_lidar(
            points=points,
            dimensions=annotation_dict['dimensions'],
            location=annotation_dict['location'],
            rotation_y=annotation_dict['rotation_y'],
            name=annotation_dict['name']
        )
    
    def create_db_info(self, lidar_points, calib_dict, annotation_dict,
                       sample_id, db_path, dbinfos):
        """Create database info for V2X data augmentation"""
        if len(annotation_dict['name']) == 0:
            return
        
        indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
            points_in_bboxes_v2_lidar(
                points=lidar_points,
                dimensions=annotation_dict['dimensions'].astype(np.float32),
                location=annotation_dict['location'].astype(np.float32),
                rotation_y=annotation_dict['rotation_y'].astype(np.float32),
                name=annotation_dict['name']
            )
        
        id_str = self.format_sample_id(sample_id)
        
        for j in range(n_valid_bbox):
            db_points = lidar_points[indices[:, j]]
            if len(db_points) == 0:
                continue
            
            db_points[:, :3] -= bboxes_lidar[j, :3]
            
            db_points_saved_name = os.path.join(
                db_path, f'{id_str}_{name[j]}_{j}.bin'
            )
            write_points(db_points, db_points_saved_name)
            
            db_info = {
                'name': name[j],
                'path': os.path.join(os.path.basename(db_path),
                                    f'{id_str}_{name[j]}_{j}.bin'),
                'box3d_lidar': bboxes_lidar[j],
                'difficulty': annotation_dict['difficulty'][j] if j < len(annotation_dict['difficulty']) else -1,
                'num_points_in_gt': len(db_points),
            }
            
            if name[j] not in dbinfos:
                dbinfos[name[j]] = [db_info]
            else:
                dbinfos[name[j]].append(db_info)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='V2X Dataset Preprocessing')
    parser.add_argument('--data_root', default='/data/zahra/v2x',
                        help='Path to V2X dataset root')
    parser.add_argument('--prefix', default='v2x',
                        help='Prefix for output files')
    parser.add_argument('--sample', action='store_true',
                        help='Only process first 10 samples for testing')
    args = parser.parse_args()
    
    print("="*60)
    print("V2X Preprocessing")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"Z convention: CENTER -> BOTTOM (fixed)")
    print("="*60)
    
    preprocessor = V2XPreprocessor(args.data_root, args.prefix)
    
    if args.sample:
        results = preprocessor.run_sample(n_samples=10)
        for sample_id, info in results:
            print(f"\nSample ID: {sample_id}")
            if info and 'annos' in info:
                print(f"  Num objects: {len(info['annos']['name'])}")
                if len(info['annos']['location']) > 0:
                    print(f"  First obj z (BOTTOM): {info['annos']['location'][0][2]:.2f}")
    else:
        preprocessor.run()
    
    print("\nPreprocessing complete!")