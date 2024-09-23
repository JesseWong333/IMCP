# -*- coding: utf-8 -*-
# Author: Junjie Wang <junjie.wang@umu.se>

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.defor_encoder_multi_scale import Block
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head
from opencood.models.sub_modules.downsample_conv import DownsampleConv

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

import loralib as lora

from mmdet.models.utils import LearnedPositionalEncoding
from torch.nn.init import normal_

def conv3x3(in_planes, out_planes, lora_rank=0, stride=1):
    "3x3 convolution with padding"
    return lora.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, r=lora_rank)

def conv1x1(in_planes, out_planes, lora_rank=0, stride=1):
    "1x1 convolution with padding"
    return lora.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False, r=lora_rank)

class AdapterDouble(nn.Module):
    def __init__(self, input_filter, output_filter, n_layers=3, lora_rank=0):
        super().__init__()
        # 针对每一个特征级别
        self.conv0 = nn.Sequential(
                lora.Conv2d(input_filter, output_filter, kernel_size=1, r=lora_rank),
                nn.BatchNorm2d(output_filter)
            )
        
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Sequential(
                conv3x3(output_filter, output_filter, lora_rank),
                nn.BatchNorm2d(output_filter),
                nn.ReLU(inplace=True),
                conv3x3(output_filter, output_filter),
                nn.BatchNorm2d(output_filter),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv0(x)
        residual = x
        for layer in self.layers:
            x = layer(x)
            # F(x)+x
            x += residual
            x = F.relu(x)
            residual = x
        return x
    
class Adapter(nn.Module):
    def __init__(self, input_filter, output_filter, n_layers=3, lora_rank=0):
        super().__init__()
        # 针对每一个特征级别
        self.conv0 = nn.Sequential(
                lora.Conv2d(input_filter, output_filter, kernel_size=1, r=lora_rank),
                nn.BatchNorm2d(output_filter),
                nn.ReLU(inplace=True)
            )
        
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Sequential(
                conv3x3(output_filter, output_filter, lora_rank),
                nn.BatchNorm2d(output_filter),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv0(x)
        residual = x
        for layer in self.layers:
            x = layer(x)
            # F(x)+x
            x += residual
            x = F.relu(x)
            residual = x
        return x

class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        if 'resnet' in args['base_bev_backbone'] and args['base_bev_backbone']['resnet']:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        self.seg_flag = args['single_seg_head'] if 'single_seg_head' in args else False
        
        if self.seg_flag:
            self.single_seg_head = nn.Conv2d(args['head_embed_dims'], 2, kernel_size=1)
     
        self.single_cls_head = nn.Conv2d(args['head_embed_dims'], args['anchor_number'], # 384
                                  kernel_size=1)
        self.single_reg_head = nn.Conv2d(args['head_embed_dims'], 7 * args['anchor_number'], # 384
                                  kernel_size=1)

    def forward(self, batch_dict):

        batch_dict = self.pillar_vfe(batch_dict)  # 这里是其提取特征的方式，不同的方法应该不同
        batch_dict = self.scatter(batch_dict)
        batch_dict, ret_dict = self.backbone(batch_dict)  # ret_dict存放了多个级别的特征
        if self.shrink_flag:
            batch_dict['spatial_features_2d'] = self.shrink_conv(batch_dict['spatial_features_2d'])
        psm_single = self.single_cls_head(batch_dict['spatial_features_2d'])
        rm_single = self.single_reg_head(batch_dict['spatial_features_2d'])
        output_dict = {'cls_preds': psm_single,
                        'reg_preds': rm_single}
        if self.seg_flag:
            seg_single = self.single_seg_head(batch_dict['spatial_features_2d'])
            output_dict.update({'seg_preds': seg_single})
        multi_feature = [ret_dict[key] for key in ret_dict]
        return multi_feature, output_dict

class Second(nn.Module):
    def __init__(self, args):
        super(Second, self).__init__()

        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                             np.array(args['voxel_size'])).astype(np.int64)
        self.vfe = MeanVFE(args['mean_vfe'],
                           args['mean_vfe']['num_point_features'])
        self.spconv_block = VoxelBackBone8x(args['spconv'],
                                            input_channels=args['spconv'][
                                                'num_features_in'],
                                            grid_size=grid_size)
        self.map_to_bev = HeightCompression(args['map2bev'])
        self.ssfa = SSFA(args['ssfa'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.head = Head(**args['head'])
        
    def forward(self, batch_dict):
        voxel_coords = batch_dict['voxel_coords']
        batch_size = voxel_coords[:,0].max() + 1 # batch size is padded in the first idx

        batch_dict.update({'batch_size': batch_size})

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        out = self.ssfa(batch_dict['spatial_features'])
        if self.shrink_flag:
            out = self.shrink_conv(out)

        pred = self.head(out)  # 这个有更多的输出，iou
        
        return [batch_dict['spatial_features']], pred

class DeforEncoderFusion(nn.Module):
    def __init__(self, model_cfg):
        super(DeforEncoderFusion, self).__init__()
        self.blocks = nn.ModuleList()

        block_cfgs = model_cfg["block_cfgs"]
        for block_cfg in block_cfgs:
            self.blocks.append(Block(*block_cfg))

        self.bev_h = model_cfg["bev_h"] # 100
        self.bev_w = model_cfg["bev_w"] # 252
        self.discrete_ratio = model_cfg["discrete_ratio"]

        self.embed_dims = model_cfg["embed_dims"]  # 128

        self.agent_names = model_cfg["agent_names"] # 
        self.feature_levels = model_cfg["feature_levels"] # 不同的agent 可以有不同的feature数量

        self.lora_rank = model_cfg["lora_rank"]

        # bev 
        self.bev_embedding = lora.Embedding(self.bev_h * self.bev_w, self.embed_dims, r = 0)
        self.bev_index = torch.linspace(0, self.bev_h * self.bev_w -1, self.bev_h * self.bev_w, dtype=torch.long).cuda()

        self.positional_encoding = LearnedPositionalEncoding(        
            num_feats=self.embed_dims//2,
            row_num_embed=self.bev_h,
            col_num_embed=self.bev_w)
        
        # 不同agent的不同level的feature_lvl没有可比性，每个都给
        agent_lvl_embeding_dict = {}
        for agent_name, feature_level in zip(self.agent_names, self.feature_levels):
            agent_lvl_embeding_dict[agent_name] = nn.Parameter(torch.Tensor(feature_level, self.embed_dims))
            normal_(agent_lvl_embeding_dict[agent_name])
        self.agent_lvl_embeds = nn.ParameterDict(agent_lvl_embeding_dict)
        
        # adapter
        n_adapter_layers =  model_cfg['n_adapters'] if 'n_adapters' in model_cfg else 0
        adapter_dict = OrderedDict()
        for name, in_out_channels in model_cfg['adapters'].items():
            adapter_dict[name] = self.create_adapter(in_out_channels[0], in_out_channels[1], name, n_adapter_layers, self.lora_rank)
        self.adapters = nn.ModuleDict(adapter_dict)

        self.cls_head = lora.Conv2d(model_cfg['head_embed_dims'], model_cfg['anchor_number'],
                                  kernel_size=1, r = 0)
        self.reg_head = lora.Conv2d(model_cfg['head_embed_dims'], 7 * model_cfg['anchor_number'],
                                  kernel_size=1, r = 0)

    # def create_adapter(self, input_filters, output_filters):
    #     adapter_list = []
    #     for i in range(len(input_filters)):
    #         adapter_list.append(nn.Sequential(
    #             nn.Conv2d(input_filters[i], output_filters[i], kernel_size=1),  # we use the same size filters as the privious upsample filters
    #             nn.BatchNorm2d(output_filters[i]),
    #         ))
    #     return nn.ModuleList(adapter_list)
    
    def create_adapter(self, input_filters, output_filters, name, n_adapter_layers, lora_rank):
        adapter_list = []
        if name == 'ego':
            for i in range(len(input_filters)):
                adapter_list.append(nn.Sequential(
                    nn.Conv2d(input_filters[i], output_filters[i], kernel_size=1),  # we use the same size filters as the privious upsample filters
                    nn.BatchNorm2d(output_filters[i]),
                ))
        else:
            for i in range(len(input_filters)):
                adapter_list.append(Adapter(input_filters[i], output_filters[i], n_adapter_layers, lora_rank))
        return nn.ModuleList(adapter_list)

    @staticmethod
    def get_reference_points(H, W, bs=1, device='cuda', dtype=torch.float):
        # H, W is
        ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d
    
    def get_normalized_transformation(self, pairwise_t_matrix_c):
        # todo: magic number
        pairwise_t_matrix = pairwise_t_matrix_c.clone() # avoid in-place
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * self.bev_h / self.bev_w
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * self.bev_w / self.bev_h
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.discrete_ratio * self.bev_w) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.discrete_ratio * self.bev_h) * 2
        return pairwise_t_matrix
    
    def forward(self, mlvl_feats, pairwise_t_matrix):
        # pairwise_t_matrix: # B, cav_id, cav_id, 4, 4

        #  这里需要考虑多种情形，包括多种级别的特征，多种类型的特征
        # mlvl_feats: [ [(B, C, H1, W1), (B, C, H2, W2), (B, C, H3, W3)], 
        #               [(B, C, H1, W1), (B, C, H2, W2)], 
        #                ]
        pairwise_t_matrix = self.get_normalized_transformation(pairwise_t_matrix)

        assert len(self.adapters) == len(mlvl_feats)
        # run adapter
        mlvl_feats_out = [ [] for _ in range(len(mlvl_feats))]

        for i, (agent_name, adapter) in enumerate(self.adapters.items()):
            for j, module in enumerate(adapter):
                mlvl_feats_out[i].append(module(mlvl_feats[i][j]))

        #
        agent_lvl_embeds = []
        for key, embeds in self.agent_lvl_embeds.items():
            agent_lvl_embeds.append(embeds)

        out = []
        batch_size = mlvl_feats_out[0][0].shape[0]
        for b in range(batch_size):

            # Step1: 取对应的batch_size
            mlvl_feats_b = []
            for i in range(len(mlvl_feats_out)):
                mlvl_feats_b.append( [level_f[b:b+1] for level_f in mlvl_feats_out[i]] )
     
            t_matrix = pairwise_t_matrix[b]
            feat_flatten = []
            spatial_shapes = []
            # Step2: 对非ego特征进行transform, 特征拉直， 加embedding
            for agent_index, per_agent_features in enumerate(mlvl_feats_b):
                for lvl, feat in enumerate(per_agent_features):
                    _, c, h, w = feat.shape
                    feat = warp_affine_simple(feat, t_matrix[0, agent_index:agent_index+1, :, :], (h, w)) 
                    spatial_shape = (h, w)
                    # lvl embeding
                    feat = feat.flatten(2).transpose(1, 2) # 1, h*w, c
                    feat = feat + agent_lvl_embeds[agent_index][None, lvl:lvl + 1, :]
                    spatial_shapes.append(spatial_shape)
                    feat_flatten.append(feat)

            feat_flatten = torch.cat(feat_flatten, 1) # 1, h*w+..., C
            ref_2d = self.get_reference_points(
               self.bev_h, self.bev_w, device=feat.device, dtype=feat.dtype) # 1, H*W, 1, 2
           
            ref_2d = ref_2d.repeat(1, 1, sum(self.feature_levels), 1)  # #1, H*W, total_feature_lvls, 2

            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat.device)

            spatial_shapes_self = [(self.bev_h, self.bev_w)]
            spatial_shapes_self = torch.as_tensor(spatial_shapes_self, dtype=torch.long, device=feat.device)

            bev_queries = self.bev_embedding(self.bev_index)  # H*W, C  # run_forward, 得到全部的

            bev_queries = bev_queries.unsqueeze(0) #  [1, H*W, C]
            bev_mask = torch.zeros((1, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(feat.dtype)
            bev_pos = self.positional_encoding(bev_mask).to(feat.dtype) # [1, num_feats*2, h, w]
            bev_pos = bev_pos.flatten(2).permute(0, 2, 1) # [1, C, h*w]->[1, h*w, C] 

            for _, block in enumerate(self.blocks):
                bev_queries = block(bev_queries, bev_pos, feat_flatten, ref_2d, spatial_shapes, spatial_shapes_self)  # [1, H*W, C]
            
            bev_queries = bev_queries.permute(0, 2, 1).view(1, self.embed_dims, self.bev_h, self.bev_w)  # 就是这个问题，其他的不行也是因为我没有permute
            out.append(bev_queries)

        fused_features = torch.cat(out, dim=0)
        psm = self.cls_head(fused_features)
        rm = self.reg_head(fused_features)
        output_dict = {'cls_preds': psm,
                        'reg_preds': rm}
        return fused_features, output_dict
    