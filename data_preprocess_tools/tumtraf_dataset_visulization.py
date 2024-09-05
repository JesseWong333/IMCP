# tumtraf是什么问题

import time
import numpy as np
import math
import os
import os.path as ops
import torch
import json
import imageio.v2 as imageio
from data_preprocess_tools import custom_vis
from opencood.utils import box_utils as box_utils
import opencood.utils.pcd_utils as pcd_utils
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import glob
import opencood.utils.common_utils as common_utils
import cv2

def save_numpy(path, arr):
    with open(path, 'wb') as f:
        np.save(f, arr)

def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
    return my_json

def write_json(path_json, data):
    with open(path_json, "w") as f:
        json.dump(data, f)

def convert_tfm_matrix(rotation, translation):
    matrix = np.empty([4,4])
    matrix[0:3, 0:3] = rotation
    matrix[:, 3][0:3] = np.array(translation)[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1
    return matrix

def get_calibs(calib_path):
    calib = read_json(calib_path)
    if 'transform' in calib.keys():
        calib = calib['transform']
    rotation = calib['rotation']
    translation = calib['translation']
    return rotation, translation

def rev_matrix(rotation, translation):
    rotation = np.matrix(rotation)
    rev_R = rotation.I
    rev_R = np.array(rev_R)
    rev_T = - np.dot(rev_R, translation)
    return rev_R, rev_T

def mul_matrix(rotation_1, translation_1, rotation_2, translation_2):
    rotation_1 = np.matrix(rotation_1)
    translation_1 = np.matrix(translation_1)
    rotation_2 = np.matrix(rotation_2)
    translation_2 = np.matrix(translation_2)

    rotation = rotation_2 * rotation_1
    translation = rotation_2 * translation_1 + translation_2
    rotation = np.array(rotation)
    translation = np.array(translation)
    return rotation, translation

def trans_lidar_i2v(inf_lidar2world_path, veh_lidar2novatel_path,
                    veh_novatel2world_path, system_error_offset=None):
    inf_lidar2world_r, inf_lidar2world_t = get_calibs(inf_lidar2world_path)  # r: rotation, t: translation
    if system_error_offset is not None:
        inf_lidar2world_t[0][0] = inf_lidar2world_t[0][0] + system_error_offset['delta_x']
        inf_lidar2world_t[1][0] = inf_lidar2world_t[1][0] + system_error_offset['delta_y']

    veh_novatel2world_r, veh_novatel2world_t = get_calibs(veh_novatel2world_path)
    veh_world2novatel_r, veh_world2novatel_t = rev_matrix(veh_novatel2world_r, veh_novatel2world_t)
    inf_lidar2novatel_r, inf_lidar2novatel_t = mul_matrix(inf_lidar2world_r, inf_lidar2world_t,
                                                          veh_world2novatel_r, veh_world2novatel_t)

    veh_lidar2novatel_r, veh_lidar2novatel_t = get_calibs(veh_lidar2novatel_path)
    veh_novatel2lidar_r, veh_novatel2lidar_t = rev_matrix(veh_lidar2novatel_r, veh_lidar2novatel_t)
    inf_lidar2lidar_r,  inf_lidar2lidar_t = mul_matrix(inf_lidar2novatel_r, inf_lidar2novatel_t,
                                                       veh_novatel2lidar_r, veh_novatel2lidar_t)

    return inf_lidar2lidar_r, inf_lidar2lidar_t  # inf雷达 到 veh的雷达

def boxes_to_corners_3d(boxes3d, order):
    """
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    Parameters
    __________
    boxes3d: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, l, w, h, heading], or [x, y, z, h, w, l, heading]

               (x, y, z) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners3d: np.ndarray or torch.Tensor
        (N, 8, 3), the 8 corners of the bounding box.


    opv2v's left hand coord 
    
    ^ z
    |
    |
    | . x
    |/
    +-------> y

    """

    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    boxes3d_ = boxes3d

    if order == 'hwl':
        boxes3d_ = boxes3d[:, [0, 1, 2, 5, 4, 3, 6]]

    template = boxes3d_.new_tensor((
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    )) / 2

    corners3d = boxes3d_[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3),
                                                   boxes3d_[:, 6]).view(-1, 8,
                                                                        3)
    corners3d += boxes3d_[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def convert2conerpoints(box):
    x = box[0]
    y = box[1]
    z = box[2]
    # hwl
    h = box[3]
    w = box[4]
    l = box[5]

    yaw_lidar = box[6]
    
    center_lidar = [x, y, z]
    liadr_r = np.matrix(
        [
            [math.cos(yaw_lidar), -math.sin(yaw_lidar), 0],
            [math.sin(yaw_lidar), math.cos(yaw_lidar), 0],
            [0, 0, 1],
        ]
    )
    # corners_3d_lidar = np.matrix(
    #     [
    #         [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
    #         [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
    #         [0, 0, 0, 0, h, h, h, h],
    #     ]
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    return corners_3d_lidar.T


def read_GT(label_f, pc_range=[-72., -72., -8., 72., 72., 6]):
    with open(label_f) as f:
        label_d = json.load(f)
  
    frames = label_d['openlabel']['frames']
    for prop in frames.values():
        object_dict = prop['objects']
    
    objects = []
    for token, object_info in object_dict.items():
        one_sample = dict()
        object_data = object_info['object_data']
        one_sample['obj_type'] = object_data['type']
        # occluded_state = object_data['cuboid']['attributes']["text"][0]['val']

        loc = object_data['cuboid']['val'][:3]
        dim = object_data['cuboid']['val'][7:]
        rot = np.asarray(object_data['cuboid']['val'][3:7], dtype=np.float32)  # Quaternion in x,y,z,w

        rot_temp = Rotation.from_quat(rot)
        rot_temp = rot_temp.as_euler('xyz', degrees=False)

        yaw = rot_temp[2]

        gt_box = np.concatenate([loc, dim[2], dim[1], dim[0], yaw], axis=None)
        world_8_points = boxes_to_corners_3d(gt_box[None, :], order='hwl') # 这个是opencood中的函数

        objects.append(np.asarray(world_8_points))  # inf标签里面比没有8个点
    
    if len(objects) > 0:
        return np.concatenate(objects, axis=0)  # 可能出现全部过滤掉的情形
    else:
        return np.zeros([0,3])

def choose_nearest_boxes(anchor_box, all_overlap_boxes):
    # anchor_box: 4,2
    # all_overlap_boxes;

    if len(all_overlap_boxes) == 1:
        return all_overlap_boxes[0]

    if len(all_overlap_boxes) == 0:
        return None

    anchor_center = anchor_box.mean(axis=0)

    min_dist = 1e6
    nearest_boxes = None
    for overlap_box in all_overlap_boxes:
        dist = np.sum(np.square(overlap_box-anchor_center))
        if dist < min_dist:
            nearest_boxes = overlap_box
    return nearest_boxes


def track_boxes(anchor_GT, previous_GT_l):
    # anchor_GT:  N,4,2
    # previous_GT_l: [N',4,2] a list #将每个box在不同帧的结果一一对应
    # 在100ms内，车辆肯定有重合；路口最快20m/s (72km/h), 100ms最多行驶2m, 小轿车大于4m

    results = [[anchor_GT[i]] for i in range(anchor_GT.shape[0]) ]
    for previous_GT in previous_GT_l:
        # 
        for result in results:
            if result[-1] is None:
                continue
            overlap_boxes = []
            for i in range(previous_GT.shape[0]):   
                # 选择重叠，且最近的那个    
                if isRectsOverlap(result[-1], previous_GT[i]):
                    overlap_boxes.append(previous_GT[i])
            boxes = choose_nearest_boxes(result[-1], overlap_boxes) # boxes可以为空
            result.append(boxes)
    return results

def estimate_transform(boxes):
    # 传输进来的是一个list的boxes; boxes[0]是当前帧， boxes[1]是前一帧
    # 预测l-1个transform
    transformations = []
    for i in range(1, len(boxes)):
         if boxes[i] is not None:
            transformations.append(getTransform(boxes[0], boxes[i]))
    return transformations

# def filter_points(points, bev_shape=[100, 252]):
#     # points: N*4
#     # 返回：N'*4
#     out_points = []
#     for i in range(points.shape[0]):
#         point = points[i]
#         if (0 < point[0] < bev_shape[1]) and (0 < point[1] < bev_shape[0]):
#             out_points.append(point)

#     if len(out_points) > 0:
#         return np.vstack(out_points)
#     else:
#         return None

def filter_points(points, bev_shape=[100, 252]):
    # x_boolean = (points[:, 2] <= (bev_shape[1]-1)) & (points[:, 2] <= (bev_shape[1]-1)) & (points[:, 2] >= 0)
    # y_boolean = np.logical_and(points[:, 3] <= bev_shape[0]-1,  points[:, 3] >= 0)
    # xy_boolean = np.logical_and(x_boolean, y_boolean)
    boolean_1 = (points[:, 0] >= 0) & (points[:, 0] <= (bev_shape[1]-1))
    boolean_2 = (points[:, 1] >= 0) & (points[:, 1] <= (bev_shape[0]-1))
    boolean_3 = (points[:, 2] >= 0) & (points[:, 2] <= (bev_shape[1]-1))
    boolean_4 = (points[:, 3] >= 0) & (points[:, 3] <= (bev_shape[0]-1))
    boolean_all = boolean_1 & boolean_2 & boolean_3 & boolean_4
    points =points[boolean_all, :]
    return points
    
def get_point_transformation(rects, transformations, bev_shape=[100, 252]):
    # 找到anchor矩形框的所有点， 投影到之前的帧
    # rects: 6, 4, 2
    # input: a list of transformation, transformations少一个
    points_ = getPointsInQuad(rects[0])
    point_trans = []
    for i in range(len(transformations)):
        points = points_.copy()
        transfomed_points = applyTransform(points, transformations[i]) # N,2
        # 转换为以左上角为原点的坐标
        # points[:, 1] = -(points[:, 1] - bev_shape[0])
        # transfomed_points[:, 1] = -(transfomed_points[:, 1] - bev_shape[0])
        # filter out points
        transform_tuple = np.concatenate([points, transfomed_points], axis=1) # 原坐标 -> 新坐标
        transform_tuple = filter_points(transform_tuple)
        if transform_tuple.shape[0] > 0:
            point_trans.append(transform_tuple)
    return point_trans

def scale_boxes(boxes, pc_range=[-100.8, -40, -3.5, 100.8, 40, 1.5], bev_shape=[100, 252] ):
    # boxes: N, 4, 2
    # scale_y = 100 / (40*2)
    # scale_x = 252 / (100.8*2)
    # 转换之后是以左下为原点的！！
    scale_y = bev_shape[0] / (pc_range[4] - pc_range[1])
    scale_x = bev_shape[1] / (pc_range[3] - pc_range[0])

    boxes[:, :, 0] = (boxes[:, :, 0] - pc_range[0])* scale_x
    boxes[:, :, 1] = (boxes[:, :, 1] - pc_range[1])* scale_y
    return boxes

def update_offset_map(transform_tuple, offset_map, bev_shape=[100, 252]):
    # transform_tuple: 原始坐标， anchor的坐标点, N,2
    # transform_tuple: 变换后的坐标点, N, 2
    points = transform_tuple[:, :2].astype(np.int64)
    transformed_points = transform_tuple[:, 2:]
    offsets = transformed_points - points # (N, 2)
    # 在 transform_tuple[0]的位置记录下这个offset
    offset_map[points[:,1], points[:,0], :] = offsets
    return offset_map

def vis_point_track(points_tracks, lidars, path):
    # 将points_track 和 lidar转换为， 图像 和 序列
    # trajs is S, N, 2
    # rgbs is S, C, H, W
    
    S = len(lidars)  
    # 不同点的数量不一样多， 我是靠原始的点来表明是哪一个点的； 原来是只要一帧丢失就全部丢失
    # 转化为直接查询结构 {"原始点Hash+帧": 点} 这样定位的时候就可以直接查找
    point_hash = {}
    for one_box_track in points_tracks:
        anchor_points = one_box_track[0]
        for i in range(anchor_points.shape[0]):
            point_hash[str(int(anchor_points[i][0]))+ '_' + str(int(anchor_points[i][1])) + '_0' ]  = anchor_points[i][:2]  # 元素为2
        
        for n_frame, point in enumerate(one_box_track):
            for i in range(point.shape[0]):    
                point_hash[str(int(point[i][0]))+ '_' + str(int(point[i][1])) + '_' + str(n_frame+1) ]  = point[i][2:]

    # BEV是[100, 252]， 雷达图可以画得很大，之前是 2000,800, 画为8倍[800, 2016]
    BEV_images = [custom_vis.convert_lidar_to_BEV_image(lidar).canvas for lidar in lidars]
    
    images = custom_vis.summ_traj2ds_on_rgbs(point_hash, BEV_images)
    images[0].save(path, format='gif', save_all=True, append_images=images[1:], duration=500,loop=0)
    
def get_offset_maps(points_tracks, n_frame, bev_shape=[100, 252]):
    # 将变换过后的点的位置记为fale;
    # points_tracks：
    offset_maps = [ np.zeros((bev_shape[0], bev_shape[1], 2), dtype=np.float32) for _ in range(n_frame)]
    point_masks = [np.full((bev_shape[0], bev_shape[1]), False, dtype=bool) for _ in range(n_frame)]  # 为True的地方表示该地方被遮挡
    for one_boxes_points in points_tracks:  # 对每一个boxes
        for i, point_trans in enumerate(one_boxes_points):  # 对每一帧
            transfered_points = point_trans[:, 2:].astype(np.int32)
            point_masks[i][transfered_points[:,1], transfered_points[:,0]] = True
    
    for one_boxes_points in points_tracks:
        for i, point_trans in enumerate(one_boxes_points):
            offset_maps[i] = update_offset_map(point_trans, offset_maps[i])
            ori_points = point_trans[:, :2].astype(np.int32)
            point_masks[i][ori_points[:,1], ori_points[:,0]] = False
    return offset_maps, point_masks

def filter_point_cloud(pcb_np, pc_range=[-72., -72., -8., 72., 72., 6]):
    # -3.5已经比较低了，路端高5m, 人车高1到2m,信息都在
    pcb_filtered = pcb_np[ (pcb_np[:,0] > pc_range[0]) &  (pcb_np[:,0] < pc_range[3]) & (pcb_np[:,1] > pc_range[1]) & (pcb_np[:,1] < pc_range[4]) & (pcb_np[:,2] > pc_range[2]) & (pcb_np[:,2] < pc_range[5])]
    # print("trimed_point{}".format(  (pcb_np.shape[0]-pcb_filtered.shape[0])/pcb_np.shape[0]  ))
    return pcb_filtered

def vis_gt(label_path, data_dir, bev_shape=[180, 180]):
    file_name = label_path.split('/')[-1].split('.')[0]
    # file_name = file_name.split('_and_')[0]
    lidar_anchor, _ = pcd_utils.read_pcd(os.path.join(data_dir, file_name + '.pcd'))
    lidar_anchor = filter_point_cloud(lidar_anchor)

    anchor_GT = read_GT(label_path)

    # vis to box
    target = {}
    target["gt_box_tensor"] = torch.from_numpy(anchor_GT)
    vis_save_path =  os.path.join('tmp', 'bev_{}_{}.gif'.format(file_name, "box"))
    
    anchor_image = custom_vis.visualize(target, torch.from_numpy(lidar_anchor),
                                    [-72., -72., -8., 72., 72., 6],
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
    data_dir = '/hd_cache/datasets/tumtraf/val/point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered'
    label_l = glob.glob('/hd_cache/datasets/tumtraf/val/labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/*.json')
    for label_path in tqdm(label_l):
        vis_gt(label_path, data_dir)
      