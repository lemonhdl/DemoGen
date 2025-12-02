import numpy as np
import h5py
import os
import cv2

def _decode_rgb_image(rgb_dataset):
    """解码首帧RGB图像数据集"""
    camera_bits = rgb_dataset[0]
    camera_img = cv2.imdecode(np.frombuffer(camera_bits, np.uint8), cv2.IMREAD_COLOR)
    return camera_img

def save_first_rgb_frame_from_h5(h5_file_path, output_dir):
    # 创建输出目录
    rgb_output_dir = os.path.join(output_dir, 'rgb')
    os.makedirs(rgb_output_dir, exist_ok=True)

    # 打开h5文件
    with h5py.File(h5_file_path, 'r') as h5_file:
        rgb_data = h5_file['observation/head_camera/rgb']
        # 解码首帧RGB图像
        rgb_frame = _decode_rgb_image(rgb_data)
        # 保存首帧，文件名为000000.png
        rgb_frame_path = os.path.join(rgb_output_dir, 'mask.jpg')
        save_image(rgb_frame, rgb_frame_path)

def save_image(image_array, file_path):
    from PIL import Image
    image = Image.fromarray(image_array)
    image.save(file_path)

def new_func(i):
    h5_file_path = f'/home/lemonhdl/workspace/DemoGen/data/datasets/robotwin_ori/beat_block_hammer/loop1-8-all/data/episode{i}.hdf5'
    output_dir = f'/home/lemonhdl/workspace/DemoGen/data/sam_mask/beat_block_hammer/block'
    os.makedirs(output_dir, exist_ok=True)
    save_first_rgb_frame_from_h5(h5_file_path, output_dir)

new_func(i=0)  # 示例调用，处理episode0.hdf5
