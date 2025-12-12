# -*- coding: utf-8 -*-
# Author: Junjie Wang <junjie.wang@umu.se>

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from opencood.models.sub_modules.dynamic_layers import DynamicConv2d, DynamicBatchNorm2d

from opencood.models.sub_modules.base_single_module import PointPillar, Second, DeforEncoderFusion
from opencood.models.lift_splat_shoot import LiftSplatShoot
from opencood.models.meta_flow import MetaFlow, SingleFlowAdapter, MultiScaleFlowAdapter
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

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
                # lora.Conv2d(input_filter, output_filter, kernel_size=1, r=lora_rank),
                DynamicConv2d(input_filter, output_filter, kernel_size=1),
                DynamicBatchNorm2d(output_filter),
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
        
        if 'calibrate' in args and args['calibrate']:
            self.calibrate = True
            # create calibrate model
            self.meta_flow = MetaFlow(args['meta_flow'])
            if args['flow_adapter']['type'] == 'multi_scale':
                self.flow_adapter = MultiScaleFlowAdapter(args['flow_adapter'])
            elif args['flow_adapter']['type'] == 'single_scale':
                self.flow_adapter = SingleFlowAdapter(args['flow_adapter'])
        else:
            self.calibrate = False
        self.bev_h = args['defor_encoder_fusion']['bev_h']
        self.bev_w = args['defor_encoder_fusion']['bev_w']
        self.discrete_ratio = args['defor_encoder_fusion']['discrete_ratio']
        
        self.train_agent_ID = args['train_agent_ID']
        
        n_downsample_layers =  args['n_downsample'] if 'n_downsample' in args else 0
        if n_downsample_layers > 0:
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
    
    def get_normalized_transformation(self, pairwise_t_matrix_c):
        
        pairwise_t_matrix = pairwise_t_matrix_c.clone() # avoid in-place
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * self.bev_h / self.bev_w
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * self.bev_w / self.bev_h
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.discrete_ratio * self.bev_w) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.discrete_ratio * self.bev_h) * 2
        return pairwise_t_matrix
    # only used for flow module
    def project_feature_map(self, multi_scale_feature, pairwise_t_matrix, batch_index, agent_index=1):
        batch_features_projected =[]
        for level_feature in multi_scale_feature:
            T, _, h, w = level_feature.shape
            projected_level_feature = warp_affine_simple(level_feature, pairwise_t_matrix[batch_index:batch_index+1, 0, agent_index].repeat(T, 1, 1), (h, w))
            batch_features_projected.append(projected_level_feature)
        return batch_features_projected
 
    def generate_flow(self, batch_history_lidar, time_delay, num_levels, pairwise_t_matrix):
        # 这里假设只有两个 agent：ego 和 1； 对于 DairV2X 和v2v4real适用; 必须假的 0 是 ego， 1 是非 ego
        multiscale_features = [[] for _ in range(num_levels)]
        
        for batch_index, batch_lidars in enumerate(batch_history_lidar):
            with torch.no_grad():
                history_features, _ = self.model_i(batch_lidars)
            # 通过adapdation layer
            if hasattr(self, 'downsample_layers'):
                for agent_name, downsamplers in self.downsample_layers.items():
                    history_features =  [ downsampler(f) for f, downsampler in zip(history_features, downsamplers)]
            if hasattr(self.model_fusion, 'quantize_bit'):
                quantized_features = [self.model_fusion.quantize_feature_maps_FSQ(non_ego_feature, bitwidth=self.model_fusion.quantize_bit) for non_ego_feature in history_features]
            
            for i, (agent_name, adapter) in enumerate(self.model_fusion.adapters.items()):
                if agent_name == 'ego':
                    continue
                history_features = [module(quantized_features[j]) for j, module in enumerate(adapter)]
            # history_features = [module(quantized_features[j]) for j, module in enumerate(self.flow_adapter)]
                
            T_frame_features = self.project_feature_map(history_features, pairwise_t_matrix, batch_index, agent_index=1)
            for i in range(num_levels):
                multiscale_features[i].append(T_frame_features[i].unsqueeze(0))
        multiscale_features = [ torch.cat(x, dim=0) for x in multiscale_features] # [ (b, t, c0, h0, w0), (b, t, c1, h1, w1), (b, t, c2, h2, w2) ]
        
        multiscale_features = self.flow_adapter(multiscale_features)
        pred_offset = self.meta_flow(multiscale_features, time_delay) # b, h, w, 2
        pred_offsets = pred_offset.flatten(start_dim=1, end_dim=2).unsqueeze(2) # b, hw, 1, 2
        return pred_offsets
    def forward(self, data_dict):
        pairwise_t_matrix = data_dict['ego']['pairwise_t_matrix'] # B, cav_id, cav_id, 4, 4
        pairwise_t_matrix = self.get_normalized_transformation(pairwise_t_matrix)  # pairwise_t_matrix normalize过程提前
        cav_id_list = data_dict['ego']['cav_id_list']

        if self.train_agent_ID == -4: # backbone freeze
            # vehicle
            data_dict_v = self.repack_data(data_dict, 0)
            data_dict_i = self.repack_data(data_dict, 1)
            with torch.no_grad():
                feature_v, _ = self.model_v(data_dict_v)
            feature_i, _ = self.model_i(data_dict_i)
            _, output_dict = self.model_fusion( [feature_v, feature_i], pairwise_t_matrix, cav_id_list)
            return output_dict

        if self.train_agent_ID == -2: # end-to-end training
            data_dict_v = self.repack_data(data_dict, 0)
            data_dict_i = self.repack_data(data_dict, 1)
            feature_v, _ = self.model_v(data_dict_v)
            feature_i, _ = self.model_i(data_dict_i) 
            # fusion module
            _, output_dict = self.model_fusion( [feature_v, feature_i], pairwise_t_matrix, cav_id_list)
            return output_dict

        if self.train_agent_ID == -1 or self.train_agent_ID == -3 or self.train_agent_ID == -5 or self.train_agent_ID == -6 or self.train_agent_ID == -7: # adapter tuning; -5 train the flow module
            # vehicle
            data_dict_v = self.repack_data(data_dict, 0)
            data_dict_i = self.repack_data(data_dict, 1)

            with torch.no_grad():
                feature_v, _ = self.model_v(data_dict_v)
                feature_i, output_dict = self.model_i(data_dict_i)
            
                # sparsify non-ego agents
                # prob = output_dict['cls_preds'].permute(0, 2, 3, 1).softmax(dim=-1)[..., 1]
                # scale_factors =  [f.shape[-1] / prob.shape[-1] for f in feature_i]
                # scales_masks = [F.interpolate(prob.unsqueeze(1), scale_factor=s, mode='nearest') > 0.8 for s in scale_factors]
            
            # downsample
            if hasattr(self, 'downsample_layers'):
                for agent_name, downsamplers in self.downsample_layers.items():
                    feature_i =  [ downsampler(f) for f, downsampler in zip(feature_i, downsamplers)]
            
            # feature_i = [f * mask for f, mask in zip(feature_i, scales_masks)]
            # feature_i = [f.masked_fill(~mask, 1e-6) for f, mask in zip(feature_i, scales_masks)]
            if self.calibrate:
                batch_history_lidar = data_dict[1]['processed_lidar_history']
                time_delay = data_dict[1]['time_delay']
                pred_offsets = self.generate_flow(batch_history_lidar, time_delay, len(feature_i), pairwise_t_matrix)
                gt_offsets = data_dict[1]['offset'].flatten(start_dim=1, end_dim=2).unsqueeze(2)  # b, h, w, 2  -> # b, hw, 1, 2
                data_dict['ego']['label_dict'].update({'offset': gt_offsets}) # inplace changes
                # if self.train_agent_ID == -5:
                #     output_dict.update({'pred_offset': pred_offsets})
                #     return output_dict
                    
            else:
                pred_offsets = None
                gt_offsets = None
            # fusion module
            _, output_dict = self.model_fusion( [feature_v, feature_i], pairwise_t_matrix, cav_id_list, gt_offsets, pred_offsets)
            
            if self.calibrate:
                output_dict.update({'pred_offset': pred_offsets})
            return output_dict
            
        else:
            single_batch_dict = self.repack_data(data_dict, self.train_agent_ID, pairwise_t_matrix)
        
            if self.train_agent_ID == 0:
                feature_v, _ = self.model_v(single_batch_dict)
                pairwise_t_matrix  = torch.eye(4)[None, None, None, :, :].repeat(feature_v[0].shape[0], 2, 2, 1, 1).to(feature_v[0].device)  # B, 1, 1, 4, 4
                _, output_dict = self.model_fusion( [feature_v], pairwise_t_matrix, cav_id_list)
            else:
                _, output_dict = self.model_i(single_batch_dict)
     
            return output_dict
    