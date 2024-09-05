import json
import numpy as np
from scipy.spatial.transform import Rotation
import pickle

CLASSES = ('CAR', 'TRAILER', 'TRUCK', 'VAN', 'PEDESTRIAN', 'BUS', 'MOTORCYCLE', 'OTHER', 'BICYCLE', 'EMERGENCY_VEHICLE')
cls_range = {
        "CAR": 50,
        "TRUCK": 50,
        "BUS": 50,
        "TRAILER": 50,
        "VAN": 50,
        'EMERGENCY_VEHICLE': 50,
        "PEDESTRIAN": 40,
        "MOTORCYCLE": 40,
        "BICYCLE": 40,
        "OTHER": 30
    }


def load_annotations(ann_file):
    with open(ann_file, mode='rb') as f:
        data = pickle.load(f)
    data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
    return data_infos

def load_gt(eval_pkl):
    data_infos = load_annotations(eval_pkl)
    # filter out bbox containing no points
    all_filtered_gt_boxes = []
    for info in data_infos:
        mask = info["valid_flag"]

        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names = info["gt_names"][mask]
        gt_corner_points = info['gt_corner_points'][mask]

        ego_dist = np.sqrt(np.sum(gt_bboxes_3d[:, :2] ** 2, axis=1))

        mask = []
        for dist, name in zip(ego_dist, gt_names):
            if dist < cls_range[name]:
                mask.append(True)
            else:
                mask.append(False)
        mask = np.array(mask)
        gt_corner_points = gt_corner_points[mask]
        all_filtered_gt_boxes.append(gt_corner_points)        
    return all_filtered_gt_boxes


if __name__ == '__main__':
    load_gt("/hd_cache/datasets/tumtraf_processed/tumtraf_v2x_nusc_infos_val.pkl")


    
 
    
