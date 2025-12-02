import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


################################# Camera Calibration ##############################################
# refer to https://gist.github.com/hshi74/edabc1e9bed6ea988a2abd1308e1cc96

ROBOT2CAM_POS = np.array([1.2274124573982026, -0.009193338733170697, 0.3683118830445561])
ROBOT2CAM_QUAT_INITIAL = np.array([0.015873920322366883, -0.18843429010734952, -0.009452363954531973, 0.9819120071477938])

OFFSET_POS=np.array([0.0, -0.01, 0.008])
OFFSET_ORI_X=R.from_euler('x', -1.1, degrees=True)
OFFSET_ORI_Y=R.from_euler('y', 1.1, degrees=True)
OFFSET_ORI_Z=R.from_euler('z', -1.6, degrees=True)

ROBOT2CAM_POS = ROBOT2CAM_POS + OFFSET_POS
ori = R.from_quat(ROBOT2CAM_QUAT_INITIAL) * OFFSET_ORI_X * OFFSET_ORI_Y * OFFSET_ORI_Z
ROBOT2CAM_QUAT = ori.as_quat()


# 仿真环境 cam2world_gl 外参
cam2world_gl = np.array([
    [1.0, 0.0, 0.0, -0.032],
    [0.0, 0.8, -0.6, -0.45],
    [0.0, 0.6, 0.8, 1.35],
    [0.0, 0.0, 0.0, 1.0]
])

# 取逆得到 robot2cam_mat
def inverse_extrinsic_matrix(matrix):
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = R_inv
    inv_matrix[:3, 3] = t_inv
    return inv_matrix

robot2cam_mat = inverse_extrinsic_matrix(cam2world_gl)

REALSENSE_SCALE = 0.001
# 新的 320x240 分辨率下的内参
fx = 358.64218
fy = 358.64218
cx = 160.0
cy = 120.0
intrinsic_matrix = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])

T_link2viz = np.eye(4)

transform_realsense_util = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# 默认 image_size 改为 mask 分辨率
image_size = (1920, 1080)
##############################################



def inverse_extrinsic_matrix(matrix):
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = R_inv
    inv_matrix[:3, 3] = t_inv
    return inv_matrix

def restore_original_pcd(transformed_points):
    xyz = transformed_points[:, :3]
    rgb = transformed_points[:, 3:]

    # 先做相机外参逆变换
    points_hom = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    xyz_trans = (robot2cam_mat @ points_hom.T).T[:, :3]
    # 再将 x 轴取反
    xyz_trans[:, 0] *= -1
    restored_points = np.hstack((xyz_trans, rgb))
    np.save('pcd_restore_debug.npy', restored_points)
    print(f"[DEBUG] restore_original_pcd: saved to pcd_restore_debug.npy, shape: {restored_points.shape}, xyz min: {xyz_trans.min(axis=0)}, max: {xyz_trans.max(axis=0)}")
    return restored_points

def project_points_to_image(point_cloud, K, R=np.eye(3), T=np.zeros(3)):
    points_3d = point_cloud[:, :3]
    points_3d = (R @ points_3d.T).T + T
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    K_proj = np.hstack((K, np.zeros((3, 1))))
    points_2d_homogeneous = (K_proj @ points_3d_homogeneous.T).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2, np.newaxis]

    return points_2d


def filter_points_by_mask(points, mask, intrinsic_matrix, image_size):
    projected_points = project_points_to_image(points, intrinsic_matrix, R=np.eye(3), T=np.zeros(3))
    pixel_coords = np.floor(projected_points).astype(int)
    # 调试：输出投影坐标和有效点统计
    print(f"[DEBUG] pixel_coords min: {pixel_coords.min(axis=0)}, max: {pixel_coords.max(axis=0)}")
    print(f"[DEBUG] mask shape: {mask.shape}, image_size: {image_size}")
    valid_points = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_size[0]) & \
                   (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_size[1])
    print(f"[DEBUG] valid_points count: {np.sum(valid_points)} / {len(points)}")
    if np.sum(valid_points) > 0:
        mask_values = mask[pixel_coords[valid_points, 1], pixel_coords[valid_points, 0]]
        print(f"[DEBUG] mask_values True count: {np.sum(mask_values)} / {len(mask_values)}")
    else:
        mask_values = np.array([])
        print("[DEBUG] No valid points in mask range!")
    final_mask = np.zeros(len(points), dtype=bool)
    final_mask[valid_points] = mask_values
    filtered_points = points[final_mask]
    print(f"[DEBUG] filtered_points shape: {filtered_points.shape}")
    return filtered_points

def trans_pcd(points):
    # 直接使用 robot2cam_mat 作为外参
    points_xyz = points[..., :3] * REALSENSE_SCALE
    point_homogeneous = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
    point_homogeneous = transform_realsense_util @ point_homogeneous.T
    point_homogeneous = T_link2viz @ point_homogeneous
    point_homogeneous = robot2cam_mat @ point_homogeneous
    point_homogeneous = point_homogeneous.T

    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz
    
    return points

def restore_and_filter_pcd(pcd_robot, mask, intrinsic_matrix=intrinsic_matrix, image_size=image_size):
    # 只筛选首帧，其余帧直接返回原始点云
    if hasattr(restore_and_filter_pcd, 'frame_idx'):
        restore_and_filter_pcd.frame_idx += 1
    else:
        restore_and_filter_pcd.frame_idx = 0

    if restore_and_filter_pcd.frame_idx == 0:
        pcd_cam = restore_original_pcd(pcd_robot)
        filtered_points = filter_points_by_mask(pcd_cam, mask, intrinsic_matrix, image_size)
        filtered_points = trans_pcd(filtered_points)
        return filtered_points
    else:
        # 非首帧直接返回原始点云（可根据实际需求调整）
        return pcd_robot
