import numpy as np
import os
import pickle
import json


def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)


def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError


def write_points(lidar_points, file_path):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        with open(file_path, 'w') as f:
            lidar_points.tofile(f)
    else:
        raise NotImplementedError


def read_calib(file_path, extend_matrix=True):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    P0 = np.array([item for item in lines[0].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P1 = np.array([item for item in lines[1].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P2 = np.array([item for item in lines[2].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    P3 = np.array([item for item in lines[3].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    R0_rect = np.array([item for item in lines[4].split(' ')[1:]], dtype=np.float32).reshape(3, 3)
    Tr_velo_to_cam = np.array([item for item in lines[5].split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    Tr_imu_to_velo = np.array([item for item in lines[6].split(' ')[1:]], dtype=np.float32).reshape(3, 4)

    if extend_matrix:
        P0 = np.concatenate([P0, np.array([[0, 0, 0, 1]])], axis=0)
        P1 = np.concatenate([P1, np.array([[0, 0, 0, 1]])], axis=0)
        P2 = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)
        P3 = np.concatenate([P3, np.array([[0, 0, 0, 1]])], axis=0)

        R0_rect_extend = np.eye(4, dtype=R0_rect.dtype)
        R0_rect_extend[:3, :3] = R0_rect
        R0_rect = R0_rect_extend

        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
        Tr_imu_to_velo = np.concatenate([Tr_imu_to_velo, np.array([[0, 0, 0, 1]])], axis=0)

    calib_dict=dict(
        P0=P0,
        P1=P1,
        P2=P2,
        P3=P3,
        R0_rect=R0_rect,
        Tr_velo_to_cam=Tr_velo_to_cam,
        Tr_imu_to_velo=Tr_imu_to_velo
    )
    return calib_dict


def read_label(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    annotation = {}
    annotation['name'] = np.array([line[0] for line in lines])
    annotation['truncated'] = np.array([line[1] for line in lines], dtype=np.float32)
    annotation['occluded'] = np.array([line[2] for line in lines], dtype=np.int32)
    annotation['alpha'] = np.array([line[3] for line in lines], dtype=np.float32)
    annotation['bbox'] = np.array([line[4:8] for line in lines], dtype=np.float32)
    annotation['dimensions'] = np.array([line[8:11] for line in lines], dtype=np.float32)[:, [2, 0, 1]] # hwl -> camera coordinates (lhw)
    annotation['location'] = np.array([line[11:14] for line in lines], dtype=np.float32)
    annotation['rotation_y'] = np.array([line[14] for line in lines], dtype=np.float32)
    
    return annotation


def write_label(result, file_path, suffix='.txt'):
    '''
    result: dict,
    file_path: str
    '''
    assert os.path.splitext(file_path)[1] == suffix
    name, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score = \
        result['name'], result['truncated'], result['occluded'], result['alpha'], \
        result['bbox'], result['dimensions'], result['location'], result['rotation_y'], \
        result['score']
    
    with open(file_path, 'w') as f:
        for i in range(len(name)):
            bbox_str = ' '.join(map(str, bbox[i]))
            hwl = ' '.join(map(str, dimensions[i]))
            xyz = ' '.join(map(str, location[i]))
            line = f'{name[i]} {truncated[i]} {occluded[i]} {alpha[i]} {bbox_str} {hwl} {xyz} {rotation_y[i]} {score[i]}\n'
            f.writelines(line)

def load_json_calib(cam_json_path, extr_json_path, extend_matrix=True):
    with open(cam_json_path, "r") as f:
        cam_calib = json.load(f)

    with open(extr_json_path, "r") as f:
        extr_calib = json.load(f)

    P = np.array(cam_calib["P"]).reshape(3, 4)
    R = np.array(cam_calib["R"]).reshape(3, 3) 
    cam_D = np.array(cam_calib["cam_D"], dtype=np.float32)
    cam_K = np.array(cam_calib["cam_K"], dtype=np.float32).reshape(3, 3)

    rotation = np.array(extr_calib["rotation"], dtype=np.float32)        
    translation = np.array(extr_calib["translation"], dtype=np.float32)
    Tr_velo_to_cam = np.hstack([rotation, translation])
    
    if extend_matrix:
        P = np.vstack([P, [0, 0, 0, 1]])

        R0_rect_extend = np.eye(4)
        R0_rect_extend[:3, :3] = R
        R = R0_rect_extend

        Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
        
    calib_dict=dict(
        cam_D=cam_D,
        cam_K=cam_K,
        R=R,
        P=P,
        Tr_velo_to_cam=Tr_velo_to_cam
    )
    return calib_dict              

def read_json_label(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)  # list of dicts

    annotation = {}
    annotation['name'] = np.array([obj["type"] for obj in data])
    annotation['truncated'] = np.array([int(obj["truncated_state"]) for obj in data], dtype=np.int32)
    annotation['occluded'] = np.array([int(obj["occluded_state"]) for obj in data], dtype=np.int32)
    annotation['alpha'] = np.array([float(obj["alpha"]) for obj in data], dtype=np.float32)
    annotation['bbox'] = np.array([[float(obj["2d_box"]["xmin"]), float(obj["2d_box"]["ymin"]),
                                    float(obj["2d_box"]["xmax"]), float(obj["2d_box"]["ymax"])]
                                   for obj in data], dtype=np.float32)
    annotation['dimensions'] = np.array([[float(obj["3d_dimensions"]["h"]),
                                          float(obj["3d_dimensions"]["w"]),
                                          float(obj["3d_dimensions"]["l"])]
                                         for obj in data], dtype=np.float32)
    annotation['location'] = np.array([[float(obj["3d_location"]["x"]),
                                        float(obj["3d_location"]["y"]),
                                        float(obj["3d_location"]["z"])]
                                       for obj in data], dtype=np.float32)
    annotation['rotation_y'] = np.array([float(obj["rotation"]) for obj in data], dtype=np.float32)

    return annotation