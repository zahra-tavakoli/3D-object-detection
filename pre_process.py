# base_preprocessor.py
import os
import numpy as np
import cv2
from abc import ABC, abstractmethod
from tqdm import tqdm
import pickle

class BasePreprocessor(ABC):
    """Abstract base class for dataset preprocessing"""
    
    def __init__(self, data_root, prefix):
        self.data_root = data_root
        self.prefix = prefix
        self.sep = os.path.sep
    
    # ==================== Abstract Methods ====================
    @abstractmethod
    def get_ids(self, data_type):
        """Get list of sample IDs for given data type (train/val/test)"""
        pass
    
    @abstractmethod
    def get_image_path(self, sample_id, split):
        """Get path to image file"""
        pass
    
    @abstractmethod
    def get_lidar_path(self, sample_id, split):
        """Get path to point cloud file"""
        pass
    
    @abstractmethod
    def get_calib_paths(self, sample_id, split):
        """Get path(s) to calibration file(s)"""
        pass
    
    @abstractmethod
    def get_label_path(self, sample_id, split):
        """Get path to label file"""
        pass
    
    @abstractmethod
    def read_points(self, path):
        """Read point cloud from file"""
        pass
    
    @abstractmethod
    def read_calib(self, paths):
        """Read calibration data and return standardized dict with P2, R0_rect, Tr_velo_to_cam"""
        pass
    
    @abstractmethod
    def read_label(self, path):
        """Read labels and return standardized annotation dict"""
        pass
    
    @abstractmethod
    def get_reduced_points(self, points, calib_dict, image_shape):
        """Remove points outside image FOV"""
        pass
    
    @abstractmethod
    def get_points_num_in_bbox(self, points, calib_dict, annotation_dict):
        """Get number of points in each bounding box"""
        pass
    
    @abstractmethod
    def get_split_folder(self, data_type):
        """Get folder name for split (e.g., 'training', 'testing')"""
        pass
    
    @abstractmethod
    def get_reduced_points_folder(self, split):
        """Get folder path for reduced points"""
        pass
    
    @abstractmethod
    def get_relative_velodyne_path(self, lidar_path):
        """Get relative path for velodyne"""
        pass
    
    @abstractmethod
    def get_relative_image_path(self, img_path):
        """Get relative path for image"""
        pass
    
    @abstractmethod
    def create_db_info(self, lidar_points, calib_dict, annotation_dict, 
                       sample_id, db_path, dbinfos):
        """Create database info for data augmentation"""
        pass
    
    # ==================== Common Methods ====================
    def read_image(self, path):
        """Read image and return shape (h, w)"""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return img.shape[:2]
    
    def judge_difficulty(self, annotation_dict):
        """Judge difficulty level based on truncation, occlusion, and bbox height"""
        truncated = annotation_dict['truncated']
        occluded = annotation_dict['occluded']
        bbox = annotation_dict['bbox']
        
        if len(bbox) == 0:
            return np.array([], dtype=np.int32)
        
        height = bbox[:, 3] - bbox[:, 1]

        MIN_HEIGHTS = [40, 25, 25]
        MAX_OCCLUSION = [0, 1, 2]
        MAX_TRUNCATION = [0.15, 0.30, 0.50]
        
        difficultys = []
        for h, o, t in zip(height, occluded, truncated):
            difficulty = -1
            for i in range(2, -1, -1):
                if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                    difficulty = i
            difficultys.append(difficulty)
        return np.array(difficultys, dtype=np.int32)
    
    def write_points(self, points, path):
        """Write points to binary file"""
        points.astype(np.float32).tofile(path)
    
    def write_pickle(self, data, path):
        """Write data to pickle file"""
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def format_sample_id(self, sample_id):
        """Format sample ID to string (override if needed)"""
        return str(sample_id)
    
    def create_data_info_pkl(self, data_type, label=True, db=False):
        """Create data information pickle file"""
        print(f"Processing {data_type} data..")
        ids = self.get_ids(data_type)
        split = self.get_split_folder(data_type)
        
        infos_dict = {}
        if db:
            dbinfos = {}
            db_points_saved_path = os.path.join(self.data_root, f'{self.prefix}_gt_database')
            os.makedirs(db_points_saved_path, exist_ok=True)
        
        for sample_id in tqdm(ids):
            try:
                cur_info_dict = self._process_single_sample(
                    sample_id, split, label, db, 
                    db_points_saved_path if db else None, 
                    dbinfos if db else None
                )
                if cur_info_dict is not None:
                    infos_dict[sample_id] = cur_info_dict
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                continue
        
        saved_path = os.path.join(self.data_root, f'{self.prefix}_infos_{data_type}.pkl')
        self.write_pickle(infos_dict, saved_path)
        print(f"Saved info to {saved_path}")
        
        if db:
            saved_db_path = os.path.join(self.data_root, f'{self.prefix}_dbinfos_train.pkl')
            self.write_pickle(dbinfos, saved_db_path)
            print(f"Saved db info to {saved_db_path}")
        
        return infos_dict
    
    def _process_single_sample(self, sample_id, split, label, db, db_path, dbinfos):
        """Process a single sample"""
        cur_info_dict = {}
        
        # Image info
        img_path = self.get_image_path(sample_id, split)
        image_shape = self.read_image(img_path)
        cur_info_dict['image'] = {
            'image_shape': image_shape,
            'image_path': self.get_relative_image_path(img_path),
            'image_idx': sample_id,
        }
        
        # Point cloud info
        lidar_path = self.get_lidar_path(sample_id, split)
        cur_info_dict['velodyne_path'] = self.get_relative_velodyne_path(lidar_path)
        
        # Calibration
        calib_paths = self.get_calib_paths(sample_id, split)
        calib_dict = self.read_calib(calib_paths)
        cur_info_dict['calib'] = calib_dict
        
        # Read points
        lidar_points = self.read_points(lidar_path)
        
        # Get reduced points (within image FOV)
        reduced_lidar_points = self.get_reduced_points(lidar_points, calib_dict, image_shape)
        
        # Save reduced points
        saved_reduced_path = self.get_reduced_points_folder(split)
        os.makedirs(saved_reduced_path, exist_ok=True)
        id_str = self.format_sample_id(sample_id)
        saved_reduced_points_name = os.path.join(saved_reduced_path, f'{id_str}.bin')
        self.write_points(reduced_lidar_points, saved_reduced_points_name)
        
        if label:
            label_path = self.get_label_path(sample_id, split)
            annotation_dict = self.read_label(label_path)
            annotation_dict['difficulty'] = self.judge_difficulty(annotation_dict)
            annotation_dict['num_points_in_gt'] = self.get_points_num_in_bbox(
                reduced_lidar_points, calib_dict, annotation_dict
            )
            cur_info_dict['annos'] = annotation_dict
            
            if db and dbinfos is not None:
                self.create_db_info(
                    lidar_points, calib_dict, annotation_dict,
                    sample_id, db_path, dbinfos
                )
        
        return cur_info_dict
    
    def run(self):
        """Run full preprocessing pipeline"""
        # Train
        train_infos = self.create_data_info_pkl('train', db=True)
        
        # Val
        val_infos = self.create_data_info_pkl('val')
        
        # TrainVal combined
        trainval_infos = {**train_infos, **val_infos}
        saved_path = os.path.join(self.data_root, f'{self.prefix}_infos_trainval.pkl')
        self.write_pickle(trainval_infos, saved_path)
        print(f"Saved trainval info to {saved_path}")
        
        # Test (no labels)
        test_infos = self.create_data_info_pkl('test', label=False)
        
        return train_infos, val_infos, test_infos
    
    def run_sample(self, n_samples=10, data_type='train'):
        """Run preprocessing on first n samples for testing"""
        print(f"Processing first {n_samples} {data_type} samples...")
        ids = self.get_ids(data_type)[:n_samples]
        split = self.get_split_folder(data_type)
        
        results = []
        for sample_id in tqdm(ids):
            try:
                cur_info_dict = self._process_single_sample(
                    sample_id, split, label=True, db=False, 
                    db_path=None, dbinfos=None
                )
                results.append((sample_id, cur_info_dict))
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                import traceback
                traceback.print_exc()
                results.append((sample_id, None))
        
        return results