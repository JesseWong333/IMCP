# Author: Yangheng Zhao <zhaoyangheng-sjtu@sjtu.edu.cn>
import os
import pickle
from collections import OrderedDict
from typing import Dict
from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset
from os import path as osp

from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from mmdet3d.core.bbox import LiDARInstance3DBoxes

class TumtrafBaseDataset(Dataset):

    CLASSES = ('CAR', 'TRAILER', 'TRUCK', 'VAN', 'PEDESTRIAN', 'BUS', 'MOTORCYCLE', 'OTHER', 'BICYCLE', 'EMERGENCY_VEHICLE')

    def __init__(self,
                 params: Dict,
                 visualize: bool = False,
                 train: bool = True):
        self.params = params
        self.visualize = visualize
        self.train = train
        
        if self.train:
            self.ann_file = params['root_dir']
        else:
            self.ann_file  = params['validate_dir']
        
        self.load_lidar_file = True if 'lidar' in params['input_source'] else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False

        self.data_infos = self.load_annotations(self.ann_file)

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.data_augmentor = DataAugmentor(params['data_augment'], train)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        assert self.label_type in ['lidar', 'camera']

        self.use_valid_flag = True
        self.with_velocity = False
 
    def __len__(self) -> int:
        return self.len_record

    def load_annotations(self, ann_file):
        with open(ann_file, mode='rb') as f:
            data = pickle.load(f)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results
    
    def get_data_info(self, index: int):
        info = self.data_infos[index]

        data = OrderedDict()

        data = dict(
            timestamp=info["timestamp"],
            location=info["location"],
            vehicle_lidar_path=info["vehicle_lidar_path"],
            vehicle_sweeps=info["vehicle_sweeps"],
            infrastructure_lidar_path=info["infrastructure_lidar_path"],
            infrastructure_sweeps=info["infrastructure_sweeps"],
            registered_lidar_path=info["registered_lidar_path"],
            registered_sweeps=info["registered_sweeps"],
            vehicle2infrastructure = info["vehicle2infrastructure"],
        )

        # 车辆有多个摄像头
        if self.load_camera_file:
            data["vehicle_image_paths"] = []
            data["vehicle_lidar2camera"] = []
            data["vehicle_lidar2image"] = []
            data["vehicle_camera_intrinsics"] = []
            data["vehicle_camera2lidar"] = []
            data["infrastructure_image_paths"] = []
            data["infrastructure_lidar2camera"] = []
            data["infrastructure_lidar2image"] = []
            data["infrastructure_camera_intrinsics"] = []
            data["infrastructure_camera2lidar"] = []

            for _, vehicle_camera_info in info["vehicle_cams"].items():
                data["vehicle_image_paths"].append(vehicle_camera_info["data_path"])

                # lidar to camera transform
                vehicle_camera2lidar = vehicle_camera_info["sensor2lidar"]
                vehicle_camera2lidar = np.vstack([vehicle_camera2lidar, [0.0, 0.0, 0.0, 1.0]])
                vehicle_lidar2camera = np.linalg.inv(vehicle_camera2lidar)
                vehicle_lidar2camera = vehicle_lidar2camera[:-1, :]
                data["vehicle_lidar2camera"].append(vehicle_lidar2camera)

                # camera intrinsics
                data["vehicle_camera_intrinsics"].append(vehicle_camera_info["camera_intrinsics"])

                # lidar to image transform
                data["vehicle_lidar2image"].append(vehicle_camera_info["lidar2image"])

                # camera to lidar transform
                data["vehicle_camera2lidar"].append(vehicle_camera_info["sensor2lidar"])
            
            for _, infrastructure_camera_info in info["infrastructure_cams"].items():
                data["infrastructure_image_paths"].append(infrastructure_camera_info["data_path"])

                # lidar to camera transform
                infrastructure_camera2lidar = infrastructure_camera_info["sensor2lidar"]
                infrastructure_camera2lidar = np.vstack([infrastructure_camera2lidar, [0.0, 0.0, 0.0, 1.0]])
                infrastructure_lidar2camera = np.linalg.inv(infrastructure_camera2lidar)
                infrastructure_lidar2camera = infrastructure_lidar2camera[:-1, :]
                data["infrastructure_lidar2camera"].append(infrastructure_lidar2camera)

                # camera intrinsics
                data["infrastructure_camera_intrinsics"].append(infrastructure_camera_info["camera_intrinsics"])

                # lidar to image transform
                data["infrastructure_lidar2image"].append(infrastructure_camera_info["lidar2image"])

                # camera to lidar transform
                data["infrastructure_camera2lidar"].append(infrastructure_camera_info["sensor2lidar"])

        
        annos = self.get_ann_info(index)
        data["ann_info"] = annos

        # 装载之后有一个可配置的pipiline
        return data

    
    @abstractmethod
    def __getitem__(self, index):
        pass

    def retrieve_base_data(self, idx):
        pass

    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Notice: it is a wrap of postprocessor function

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        return self.post_processor.generate_object_center_v2x(
            cav_contents, reference_lidar_pose)
    
        raise NotImplementedError()

    def generate_object_center_camera(self, cav_contents, reference_lidar_pose):
        raise NotImplementedError()


if __name__ == "__main__":
    # data/tumtraf_v2x_cooperative_perception_dataset_processed/tumtraf_v2x_nusc_infos_train.pkl
    params = dict()
    params['root_dir'] = '/hd_cache/datasets/tumtraf_processed/tumtraf_v2x_nusc_infos_train.pkl'
    xxx = TumtrafBaseDataset(params)
    pass
