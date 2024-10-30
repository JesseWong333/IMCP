# Author: Junjie Wang <Junjie.Wang@umu.se>
import os
import pickle
from collections import OrderedDict
from typing import Dict
from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset
from os import path as osp
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.box_utils import project_world_objects_tumtraf
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor

class TUMTRAFBaseDataset(Dataset):

    # CLASSES = ('CAR', 'TRAILER', 'TRUCK', 'VAN', 'PEDESTRIAN', 'BUS', 'MOTORCYCLE', 'OTHER', 'BICYCLE', 'EMERGENCY_VEHICLE')
    CLASSES = ['CAR', 'TRAILER', 'TRUCK', 'VAN', 'PEDESTRIAN', 'BUS', 'BICYCLE']

    def __init__(self,
                 params: Dict,
                 visualize: bool = False,
                 train: bool = True):
        self.params = params
        self.visualize = visualize
        self.train = train
        
        if self.train:
            self.ann_file = params['train_dir']
        else:
            self.ann_file  = params['validate_dir']
        
        self.load_lidar_file = True if 'lidar' in params['input_source'] else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False

        self.inf_lidar_range = [-72., -72., -8., 72., 72., 0]
        self.veh_lidar_range = [-72., -72., -8., 72., 72., 6]

        self.data_infos = self.load_annotations(self.ann_file)

        self.pre_processor_i = build_preprocessor(params[params['method_i']]["preprocess"], train)
        self.pre_processor_v = build_preprocessor(params[params['method_v']]["preprocess"], train)

        self.post_processor_i = build_postprocessor(params[params['method_i']]["postprocess"], train)
        self.post_processor_v = build_postprocessor(params[params['method_v']]["postprocess"], train)

        self.post_processor_i.generate_gt_bbx = self.post_processor_i.generate_gt_bbx_by_iou_tumtraf
        self.post_processor_v.generate_gt_bbx = self.post_processor_v.generate_gt_bbx_by_iou_tumtraf

        self.pre_processor = self.pre_processor_v
        self.post_processor = self.post_processor_v

        self.data_augmentor = DataAugmentor(params['data_augment'], train)

        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        assert self.label_type in ['lidar', 'camera']

        self.use_valid_flag = True
 
    def __len__(self) -> int:
        return len(self.data_infos)

    def reinitialize(self,):
        pass

    def load_annotations(self, ann_file):
        with open(ann_file, mode='rb') as f:
            data = pickle.load(f)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_ann_info(self, info):
        
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] != 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_corner_points = info['gt_corner_points'][mask]

        # object name filters
        name_mask = []
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
                name_mask.append(True)
            else:
                name_mask.append(False)
        gt_labels_3d = np.array(gt_labels_3d)
        name_mask = np.array(name_mask)

        gt_corner_points = gt_corner_points[name_mask]
        gt_bboxes_3d = gt_bboxes_3d[name_mask]
        gt_names_3d = gt_names_3d[name_mask]
    
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        # gt_bboxes_3d = LiDARInstance3DBoxes(
        #     gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        # ).convert_to(self.box_mode_3d)

        # 
        anns_results = dict(
            gt_corner_points=gt_corner_points,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results
    
    def get_vehicle_cams_info(self, info):
        vehicle_image_paths_l = []
        vehicle_lidar2camera_l = [] # 外参
        vehicle_camera_intrinsics_l = [] # 内参
        for _, vehicle_camera_info in info["vehicle_cams"].items():
            vehicle_image_paths_l.append(vehicle_camera_info["data_path"])
            # extrinsic
            vehicle_camera2lidar = vehicle_camera_info["sensor2lidar"]
            vehicle_camera2lidar = np.vstack([vehicle_camera2lidar, [0.0, 0.0, 0.0, 1.0]])
            vehicle_lidar2camera = np.linalg.inv(vehicle_camera2lidar)
            vehicle_lidar2camera = vehicle_lidar2camera[:-1, :]
            vehicle_lidar2camera_l.append(vehicle_lidar2camera)

            vehicle_camera_intrinsics_l.append(vehicle_camera_info['camera_intrinsics'])  
        return vehicle_image_paths_l, vehicle_lidar2camera_l, vehicle_camera_intrinsics_l

    def get_inf_cams_info(self, info):
        inf_image_paths_l = []
        inf_lidar2camera_l = []
        inf_camera_intrinsics_l = []
        for _, infrastructure_camera_info in info["infrastructure_cams"].items():
            inf_image_paths_l.append(infrastructure_camera_info["data_path"])
            # lidar to camera transform
            infrastructure_camera2lidar = infrastructure_camera_info["sensor2lidar"]
            infrastructure_camera2lidar = np.vstack([infrastructure_camera2lidar, [0.0, 0.0, 0.0, 1.0]])
            infrastructure_lidar2camera = np.linalg.inv(infrastructure_camera2lidar)
            infrastructure_lidar2camera = infrastructure_lidar2camera[:-1, :]
            inf_lidar2camera_l.append(infrastructure_lidar2camera)
            # camera intrinsics
            inf_camera_intrinsics_l.append(infrastructure_camera_info["camera_intrinsics"])

        return inf_image_paths_l, inf_lidar2camera_l, inf_camera_intrinsics_l
    
    @staticmethod
    def load_lidar_points(lidar_path):
        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape(-1, 5)[:, :4]
        return points
        
    @abstractmethod
    def __getitem__(self, index):
        pass

    def retrieve_base_data(self, idx):
        info = self.data_infos[idx]

        data = OrderedDict()

        data[0] = OrderedDict()
        data[0]['ego'] = True 
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[0]['params'] = OrderedDict()
        data[1]['params'] = OrderedDict()

         # 世界坐标系: infra 的lidar坐标系
        data[1]['params']['lidar_pose'] = tfm_to_pose(np.array(info["vehicle2infrastructure"])) # c2w
        data[0]['params']['lidar_pose'] = tfm_to_pose(np.eye(4, 4))  # 世界坐标系为inf的lidar的

        data[0]['params']['vehicles'] = self.get_ann_info(info)  # 协同标签, 世界坐标系下

        if self.load_camera_file:
            # 相机可能有多个
            vehicle_cams_info = self.get_vehicle_cams_info(info)
            data[1]['camera_data'] = load_camera_data(vehicle_cams_info[0])
            data[1]['params']['camera'] = OrderedDict()
            data[1]['params']['camera']['extrinsic'] = vehicle_cams_info[1]  # lidar to camera
            data[1]['params']['camera']['intrinsic'] = vehicle_cams_info[2]

            inf_cams_info = self.get_inf_cams_info(info)
            data[0]['camera_data'] = load_camera_data(inf_cams_info[0])
            data[0]['params']['camera'] = OrderedDict()
            data[0]['params']['camera']['extrinsic'] = inf_cams_info[1]
            data[0]['params']['camera']['intrinsic'] = inf_cams_info[2]


        if self.load_lidar_file:
            data[1]['lidar_np'] = self.load_lidar_points(info["vehicle_lidar_path"]) # should be bin file
            # data[1]['lidar_np'] = self.load_lidar_points(info["infrastructure_lidar_path"])
            data[0]['lidar_np'] = self.load_lidar_points(info["registered_lidar_path"]) # todo: 使用融合了的path

        # Label for single side 单独的标签使用投影过后的标签
        data[1]['params']['vehicles_single'] = project_world_objects_tumtraf(data[0]['params']['vehicles'], data[0]['params']['lidar_pose'], self.veh_lidar_range)
        data[0]['params']['vehicles_single'] = data[0]['params']['vehicles'].copy()
        return data

    def generate_object_center_single(self,
                               cav_contents,
                               reference_lidar_pose,
                               **kwargs):
        """
        veh or inf 's coordinate
        """
        return self.post_processor.generate_object_center_tumtraf_single(cav_contents, "_single")
    
    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose,
                               **kwargs):
        return self.post_processor.generate_object_center_tumtraf(cav_contents, reference_lidar_pose)
    
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

    def generate_object_center_camera(self, cav_contents, reference_lidar_pose):
        raise NotImplementedError()

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask,
                flip=None, rotation=None, scale=None):
        """
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask,
                    'flip': flip,
                    'noise_rotation': rotation,
                    'noise_scale': scale}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask
    
if __name__ == "__main__":
    # data/tumtraf_v2x_cooperative_perception_dataset_processed/tumtraf_v2x_nusc_infos_train.pkl
    params = dict()
    params['input_source'] = ['lidar', 'camera']
    params['root_dir'] = '/hd_cache/datasets/tumtraf_processed/tumtraf_v2x_nusc_infos_train.pkl'
    dataset = TUMTRAFBaseDataset(params)
    xx = dataset.retrieve_base_data(56)
    pass
