import time
import numpy as np
import math
import os
import torch
import json
import imageio.v2 as imageio
from data_preprocess_tools import custom_vis
from opencood.utils import box_utils as box_utils
import opencood.utils.pcd_utils as pcd_utils
from tqdm import tqdm
import glob

def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
    return my_json

def get_3d_8points(object):
    h = object["3d_dimensions"]["h"] 
    w = object["3d_dimensions"]["w"] 
    l = object["3d_dimensions"]["l"] 

    yaw_lidar = object["rotation"]
    x = object["3d_location"]["x"]
    y = object["3d_location"]["y"]
    z = object["3d_location"]["z"]
    center_lidar = [x, y, z]
    liadr_r = np.matrix(
        [
            [math.cos(yaw_lidar), -math.sin(yaw_lidar), 0],
            [math.sin(yaw_lidar), math.cos(yaw_lidar), 0],
            [0, 0, 1],
        ]
    )
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    return corners_3d_lidar.T

def filter_point_cloud(pcb_np, pc_range=[-100.8, -40, -3.5, 100.8, 40, 1.5]):
    # -3.5已经比较低了，路端高5m, 人车高1到2m,信息都在
    pcb_filtered = pcb_np[ (pcb_np[:,0] > pc_range[0]) &  (pcb_np[:,0] < pc_range[3]) & (pcb_np[:,1] > pc_range[1]) & (pcb_np[:,1] < pc_range[4]) & (pcb_np[:,2] > pc_range[2]) & (pcb_np[:,2] < pc_range[5])]
    # print("trimed_point{}".format(  (pcb_np.shape[0]-pcb_filtered.shape[0])/pcb_np.shape[0]  ))
    return pcb_filtered

def read_GT(label_path, pc_range=[-100.8, -40, -3.5, 100.8, 40, 1.5]):
    # 读取target变成 N*8*3 的形式
    # inf使用的是 Cuboid Representation
    # json_file = os.path.join(data_dir, "infrastructure-side/label/virtuallidar_new/" + inf_frame_id + ".json")
    label_l = read_json(label_path)
    objects = []
    for object in label_l:
        if object["3d_dimensions"]['w'] < 1 or object["3d_dimensions"]['l'] < 1:  # 必须要过滤，小与1的物体，中间没有点， GT 为1m， bev格子1.25
            continue
        # filter out of range box, if the is out of the range of the pc range, filter it 
        x = object["3d_location"]["x"]
        y = object["3d_location"]["y"]
        z = object["3d_location"]["z"]

        if x < pc_range[0] or x > pc_range[3] or y < pc_range[1] or y > pc_range[4] or z < pc_range[2] or z > pc_range[5]:
            continue

        world_8_points = get_3d_8points(object)
        objects.append(np.asarray(world_8_points))  # inf标签里面比没有8个点
    
    if len(objects) > 0:
        return np.stack(objects, axis=0)  # 可能出现全部过滤掉的情形
    else:
        return np.zeros([0,3])
    
def vis_gt(label_path, data_dir, bev_shape=[180, 180]):
    file_name = label_path.split('/')[-1].split('.')[0]
    # file_name = file_name.split('_and_')[0]
    lidar_anchor, _ = pcd_utils.read_pcd(os.path.join(data_dir, file_name + '.pcd'))
    lidar_anchor = filter_point_cloud(lidar_anchor)

    anchor_GT = read_GT(label_path)

    # vis to box
    target = {}
    target["gt_box_tensor"] = torch.from_numpy(anchor_GT)
    vis_save_path =  os.path.join('tmp', 'bev_{}_{}.png'.format(file_name, "box"))
    
    anchor_image = custom_vis.visualize(target, torch.from_numpy(lidar_anchor),
                                    [-100.8, -40, -3.5, 100.8, 40, 1.5],
                                    None,
                                    method='bev',
                                    left_hand=True)
    anchor_image.save(vis_save_path)
    
    # vis offset maps
    # imgs = custom_vis.vis_offset_maps(offset_maps, points_masks, lidar_previous_projected)
    # import cv2
    # for i, img in enumerate(imgs):
    #     vis_save_path = os.path.join('tmp', 'bev_offset{}_{}.png'.format(anchor_id, i+1))
    #     cv2.imwrite(vis_save_path,img)

   
if __name__ == '__main__':
    data_dir = "/data/datasets/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side/velodyne"
    label_dir = "/data/datasets/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side/label/virtuallidar"
    label_l = glob.glob(os.path.join(label_dir, '*.json'))
    for label_path in tqdm(label_l):
        vis_gt(label_path, data_dir)