# -*- coding: utf-8 -*-
# Author: Junjie Wang <junjie.wang@umu.se>

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from opencood.models.sub_modules.base_single_module import PointPillar, Second, DeforEncoderFusion
from opencood.models.lift_splat_shoot import LiftSplatShoot

import loralib as lora

def conv3x3(in_planes, out_planes, lora_rank=0, stride=1):
    "3x3 convolution with padding"
    return lora.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, r=lora_rank)

def conv1x1(in_planes, out_planes, lora_rank=0, stride=1):
    "1x1 convolution with padding"
    return lora.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False, r=lora_rank)
    
class Adapter(nn.Module):
    def __init__(self, input_filter, output_filter, n_layers=3, lora_rank=0):
        super().__init__()
        
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Sequential(
                conv3x3(input_filter, input_filter, lora_rank),
                nn.BatchNorm2d(input_filter),
                ))
        self.layers = nn.ModuleList(layers)
        
        self.conv0 = nn.Sequential(
                lora.Conv2d(input_filter, output_filter, kernel_size=1, r=lora_rank),
                nn.BatchNorm2d(output_filter),
                # nn.ReLU(inplace=True)
            )

    def forward(self, x):      
        residual = x
        for layer in self.layers:
            x = layer(x)
            x += residual
            x = F.relu(x)
            residual = x
        x = self.conv0(x)
        return x
    
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
        
        n_downsample_layers =  args['n_downsample'] if 'n_downsample' in args else 0
        downsample_dict = OrderedDict()
        for name, in_out_channels in args['downsampler'].items():
            downsample_dict[name] = self.create_adapter(in_out_channels[0], in_out_channels[1], n_downsample_layers, 0)
        self.downsample_layers = nn.ModuleDict(downsample_dict)

    def create_adapter(self, input_filters, output_filters, n_adapter_layers, lora_rank):
        adapter_list = []
        for i in range(len(input_filters)):
            adapter_list.append(Adapter(input_filters[i], output_filters[i], n_adapter_layers, lora_rank))
        return nn.ModuleList(adapter_list)
    
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

        if self.train_agent_ID == -4:
            # vehicle
            data_dict_v = self.repack_data(data_dict, 0)
            data_dict_i = self.repack_data(data_dict, 1)
            with torch.no_grad():
                feature_v, _ = self.model_v(data_dict_v)
            feature_i, _ = self.model_i(data_dict_i)
            _, output_dict = self.model_fusion( [feature_v, feature_i], pairwise_t_matrix)
            return output_dict

        if self.train_agent_ID == -2:
            data_dict_v = self.repack_data(data_dict, 0)
            data_dict_i = self.repack_data(data_dict, 1)
            feature_v, _ = self.model_v(data_dict_v)
            feature_i, _ = self.model_i(data_dict_i) 
            # fusion module
            _, output_dict = self.model_fusion( [feature_v, feature_i], pairwise_t_matrix)
            return output_dict

        if self.train_agent_ID == -1 or self.train_agent_ID == -3:
            # vehicle
            data_dict_v = self.repack_data(data_dict, 0)
            data_dict_i = self.repack_data(data_dict, 1)

            with torch.no_grad():
                feature_v, _ = self.model_v(data_dict_v)
                feature_i, output_dict = self.model_i(data_dict_i)
            
                # sparsify non-ego agents
                prob = output_dict['cls_preds'].permute(0, 2, 3, 1).softmax(dim=-1)[..., 1]
                scale_factors =  [f.shape[-1] / prob.shape[-1] for f in feature_i]
                scales_masks = [F.interpolate(prob.unsqueeze(1), scale_factor=s, mode='nearest') > 0.5 for s in scale_factors]
            
            # downsample
            for agent_name, downsamplers in self.downsample_layers.items():
                feature_i =  [ downsampler(f) for f, downsampler in zip(feature_i, downsamplers)]
            
            feature_i = [f * mask for f, mask in zip(feature_i, scales_masks)]
            
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
    