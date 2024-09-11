# -*- coding: utf-8 -*-
# Author: Junjie Wang <junjie.wang@umu.se>

import torch
import torch.nn as nn

from opencood.models.sub_modules.base_single_module import PointPillar, Second, DeforEncoderFusion
from opencood.models.lift_splat_shoot import LiftSplatShoot
    
class ModelAgnosticBase(nn.Module):
    def __init__(self, args):
        super(ModelAgnosticBase, self).__init__()

        # 1) 训练哪一个agent 2) 使用哪一种方法
        if args['train_agent_ID'] < 0:
            self.model_v = self.build_model(args['method_v'], args) 
            self.model_i = self.build_model(args['method_i'], args) 
            self.model_fusion = self.build_model(args['method_fusion'], args)

        elif args['train_agent_ID'] == 0:
            self.model_v = self.build_model(args['method_v'], args)
            self.model_fusion = self.build_model(args['method_fusion'], args)  # 可以任意切换

        elif args['train_agent_ID'] == 1:
            self.model_i = self.build_model(args['method_i'], args)

        else:
            print("Please configure more agents!")
        
        self.train_agent_ID = args['train_agent_ID']

    def build_model(self, method, args):
        if 'point_pillar' in method:
            return PointPillar(args[method])  
        elif 'second' in method:
            return Second(args['second'])
        elif 'lss' in method:
            return LiftSplatShoot(args['lss'])
        elif 'defor_encoder_fusion' in method:
            return DeforEncoderFusion(args[method])

    def repack_data(self, data_dict, id):
        data = data_dict[id]
        packed_data = {}
        if 'processed_lidar' in data:
            voxel_features = data['processed_lidar']['voxel_features']
            voxel_coords = data['processed_lidar']['voxel_coords']
            voxel_num_points = data['processed_lidar']['voxel_num_points']
            packed_data.update({'voxel_features': voxel_features,
                            'voxel_coords': voxel_coords,
                            'voxel_num_points': voxel_num_points})
        
        if 'image_inputs' in data:
            packed_data.update({'image_inputs': data['image_inputs']})
        return packed_data

    def forward(self, data_dict):
        pairwise_t_matrix = data_dict['ego']['pairwise_t_matrix'] # B, cav_id, cav_id, 4, 4

        if self.train_agent_ID == -2:
            data_dict_v = self.repack_data(data_dict, 0)
            data_dict_i = self.repack_data(data_dict, 1)
            feature_v, _ = self.model_v(data_dict_v)
            feature_i, _ = self.model_i(data_dict_i) 
            # fusion module
            _, output_dict = self.model_fusion( [feature_v, feature_i], pairwise_t_matrix)
            return output_dict

        if self.train_agent_ID == -1 or -3:
            # vehicle
            data_dict_v = self.repack_data(data_dict, 0)
            data_dict_i = self.repack_data(data_dict, 1)

            with torch.no_grad():
                feature_v, _ = self.model_v(data_dict_v)
                feature_i, _ = self.model_i(data_dict_i)
                
            # fusion module
            _, output_dict = self.model_fusion( [feature_v, feature_i], pairwise_t_matrix)
            return output_dict

        else:
            single_batch_dict = self.repack_data(data_dict, self.train_agent_ID)
        
            if self.train_agent_ID == 0:
                feature_v, _ = self.model_v(single_batch_dict)
                _, output_dict = self.model_fusion( [feature_v], pairwise_t_matrix[:, 0:1, 0:1])
            else:
                _, output_dict = self.model_i(single_batch_dict)
     
            return output_dict
    