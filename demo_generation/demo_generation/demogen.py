# from re import T
# from turtle import st
# 强制使用 Agg 后端，解决 FigureCanvasTkAgg 无 tstring_rgb 报错
import matplotlib
matplotlib.use('Agg')
from diffusion_policies.common.replay_buffer import ReplayBuffer
# from regex import I
# import pcd_visualizer
import numpy as np
import copy
import os
import zarr
from termcolor import cprint
from demo_generation.mask_util import restore_and_filter_pcd
import imageio
from scipy.spatial import cKDTree
from tqdm import tqdm
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import logging


class DemoGen:
    def __init__(self, cfg):
        self.data_root = cfg.data_root
        self.source_name = cfg.source_name
        
        self.task_n_object = cfg.task_n_object
        self.use_linear_interpolation = cfg.use_linear_interpolation
        self.interpolate_step_size = cfg.interpolate_step_size

        self.use_manual_parsing_frames = cfg.use_manual_parsing_frames
        self.parsing_frames = cfg.parsing_frames
        self.mask_names = cfg.mask_names

        self.gen_name = cfg.generation.range_name
        self.object_trans_range = cfg.trans_range[self.gen_name]["object"]
        self.target_trans_range = cfg.trans_range[self.gen_name]["target"]

        self.n_gen_per_source = cfg.generation.n_gen_per_source
        self.render_video = cfg.generation.render_video
        if self.render_video:
            cprint("[NOTE] Rendering video is enabled. It takes ~10s to render a single generated trajectory.", "yellow")
        self.gen_mode = cfg.generation.mode

        source_dir = os.path.join(self.data_root, "datasets", "source")
        source_zarr = os.path.join(source_dir, self.source_name + ".zarr")
        # If exact path does not exist, try to fuzzy-match available zarr directories
        if not os.path.exists(source_zarr):
            try:
                candidates = [d for d in os.listdir(source_dir) if self.source_name in d]
            except Exception:
                candidates = []
            if len(candidates) == 1:
                matched = candidates[0]
                source_zarr = os.path.join(source_dir, matched)
                cprint(f"[INFO] source zarr not found at expected path. Using fuzzy match: {matched}", 'yellow')
            elif len(candidates) > 1:
                cprint(f"[ERROR] Multiple candidate zarrs found for source_name '{self.source_name}': {candidates}", 'red')
                raise FileNotFoundError(f"Multiple candidate zarrs found for source_name '{self.source_name}': {candidates}")
            else:
                cprint(f"[ERROR] Source zarr not found: {source_zarr}", 'red')
                # 如果是交互式 shell，让用户输入一个 HDF5 或 zarr 的绝对路径
                try:
                    import sys
                    if sys.stdin.isatty():
                        user_path = input('未找到 source zarr。请输入 HDF5 (.h5/.hdf5) 或 zarr 目录的绝对路径，或直接回车放弃：').strip()
                        if not user_path:
                            raise FileNotFoundError(f"Source zarr not found: {source_zarr}")
                        if not os.path.exists(user_path):
                            cprint(f"[ERROR] 指定路径不存在: {user_path}", 'red')
                            raise FileNotFoundError(f"Specified path does not exist: {user_path}")
                        source_zarr = user_path
                    else:
                        raise FileNotFoundError(f"Source zarr not found: {source_zarr}")
                except KeyboardInterrupt:
                    raise FileNotFoundError(f"Source zarr not found: {source_zarr}")

        self._load_from_zarr(source_zarr)

    def _load_from_zarr(self, zarr_path):
        cprint(f"Loading data from {zarr_path}", "blue")
        # Support both zarr and hdf5 files. If path ends with .hdf5 or .h5,
        # read HDF5 and construct an in-memory ReplayBuffer-compatible dict.
        if str(zarr_path).endswith('.hdf5') or str(zarr_path).endswith('.h5'):
            try:
                import h5py
            except Exception:
                raise RuntimeError('h5py is required to load .hdf5 files. Please install h5py in your environment.')
            with h5py.File(os.path.expanduser(zarr_path), 'r') as f:
                # Try multiple common names used across datasets/collectors
                def _read_first(keys):
                    for k in keys:
                        if k in f:
                            try:
                                return f[k][:]
                            except Exception:
                                return None
                    return None

                # state can be 'agent_pos', 'state', or 'endpose'
                state = _read_first(['agent_pos', 'state', 'endpose'])
                if state is None:
                    raise KeyError('HDF5 file missing "agent_pos" or "state" or "endpose" dataset')

                # action can be 'action' or 'joint_action/vector'
                action = _read_first(['action', 'joint_action/vector', 'joint_action/vector'])
                if action is None:
                    # try to reconstruct an action if left/right arms exist
                    if 'joint_action' in f and isinstance(f['joint_action'], h5py.Group):
                        # try to concatenate available sub-datasets
                        parts = []
                        for name in ['left_arm', 'left_gripper', 'right_arm', 'right_gripper', 'vector']:
                            if f['joint_action'].get(name) is not None:
                                parts.append(f['joint_action'][name][:])
                        if parts:
                            try:
                                action = np.concatenate(parts, axis=-1)
                            except Exception:
                                action = None
                if action is None:
                    raise KeyError('HDF5 file missing "action" or "joint_action/vector" dataset')

                # point cloud dataset names
                point_cloud = _read_first(['point_cloud', 'pointcloud', 'pointCloud'])
                if point_cloud is None:
                    raise KeyError('HDF5 file missing "point_cloud" or "pointcloud" dataset')

                if 'episode_ends' in f:
                    episode_ends = f['episode_ends'][:]
                else:
                    # fallback: single episode covering entire length
                    episode_ends = np.array([state.shape[0]], dtype=np.int64)

                root = {
                    'meta': {
                        'episode_ends': episode_ends
                    },
                    'data': {
                        'state': state,
                        'action': action,
                        'point_cloud': point_cloud
                    }
                }
                self.replay_buffer = ReplayBuffer(root=root)
        else:
            # assume zarr
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=['state', 'action', 'point_cloud'])

        self.n_source_episodes = self.replay_buffer.n_episodes
        self.demo_name = os.path.basename(zarr_path).split(".")[0]
    
    def generate_trans_vectors(self, trans_range, n_demos, mode="random"):
        """
        Argument: trans_range: (2, 3)
            [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        Return: A list of translation vectors. (n_demos, 3)
        """
        x_min, x_max, y_min, y_max = trans_range[0][0], trans_range[1][0], trans_range[0][1], trans_range[1][1]
        if mode == "grid":
            n_side = int(np.sqrt(n_demos))
            # print(f"n_side: {n_side}, n_demos: {n_demos}")
            if n_side ** 2 != n_demos or n_demos == 1:
                raise ValueError("In grid mode, n_demos must be a squared number larger than 1")
            x_values = [x_min + i / (n_side - 1) * (x_max - x_min) for i in range(n_side)]
            y_values = [y_min + i / (n_side - 1) * (y_max - y_min) for i in range(n_side)]
            xyz = list(set([(x, y, 0) for x in x_values for y in y_values]))
            # assert len(xyz) == n_demos
            return np.array(xyz)
        elif mode == "random":
            xyz = []
            for _ in range(n_demos):
                x = np.random.random() * (x_max - x_min) + x_min
                y = np.random.random() * (y_max - y_min) + y_min
                xyz.append([x, y, 0])
            return np.array(xyz)
        else:
            raise NotImplementedError
        
    def generate_offset_trans_vectors(self, offsets, trans_range, n_demos, mode="grid"):
        """
        For each point (translation vector) generate n_demos in trans_range.
        points_pos: (2, n_points)
            [[x1, x2, ..., xn], [y1, y2, ..., yn]]
        # NOTE: Small-range offsets are used in the experiments in our paper. However, we later found it is 
            in many times unnecessary, if we add random jitter augmentations to the point clouds when training the policy.
        """
        trans_vectors = []

        for x_offset, y_offset in zip(offsets[0], offsets[1]):
            trans_range_offset = copy.deepcopy(trans_range)
            trans_range_offset[0][0] += x_offset
            trans_range_offset[1][0] += x_offset
            trans_range_offset[0][1] += y_offset
            trans_range_offset[1][1] += y_offset
            trans_vector = self.generate_trans_vectors(trans_range_offset, n_demos, mode)
            trans_vectors.append(trans_vector)

        return np.concatenate(trans_vectors, axis=0)
    
    def get_objects_pcd_from_sam_mask(self, pcd, demo_idx, object_or_target="object"):
        assert object_or_target in ["object", "target"]
        mask = imageio.imread(os.path.join(self.data_root, f"sam_mask/{self.source_name}/{demo_idx}/{self.mask_names[object_or_target]}.jpg"))
        mask = mask > 128
        filtered_pcd = restore_and_filter_pcd(pcd, mask)
        # If filter removed all points, dump debug info for inspection and warn
        if filtered_pcd.size == 0:
            debug_dir = os.path.join(self.data_root, 'debug_masks', self.source_name, str(demo_idx))
            os.makedirs(debug_dir, exist_ok=True)
            try:
                # save original pcd and mask for inspection
                np.save(os.path.join(debug_dir, f"{object_or_target}_pcd_original.npy"), pcd)
                # save mask image (as uint8) if available
                imageio.imwrite(os.path.join(debug_dir, f"{object_or_target}_mask.png"), (mask.astype(np.uint8) * 255))
            except Exception:
                pass
            cprint(f"[WARN] Filtered {object_or_target} point cloud is empty for demo {demo_idx}. Saved debug files to {debug_dir}", 'yellow')
        return filtered_pcd
    
    def generate_demo(self):
        if self.task_n_object == 1:
            self.one_stage_augment(self.n_gen_per_source, self.render_video, self.gen_mode)
        elif self.task_n_object == 2:
            self.two_stage_augment(self.n_gen_per_source, self.render_video, self.gen_mode)
        else:
            raise NotImplementedError
        
    def parse_frames_two_stage(self, pcds, demo_idx, ee_poses, distance_mode="ee2pcd", threshold_1=0.8375, threshold_2=0.3, threshold_3=0.8376):
        """
        There are two ways to parse the frames of whole trajectory into object-centric segments: (1) Either by comparing the distance between 
            the end-effector and the object point cloud, (2) Or by manually specifying the frames when `self.use_manual_parsing_frames = True`.
        This function implements the first way. While it is an automatic process, you need to tune the distance thresholds to achieve a clean parse.
        Since DemoGen requires very few source demos, it is also feasible (actually recommended) to manually specify the frames for parsing.
        To manually decide the parsing frames, you can set the translation vectors to zero, run the DemoGen code, render the videos, and check the
            frame_idx on the left top of the video.
        """
        assert distance_mode in ["ee2pcd", "pcd2pcd"]
        skill_1_frame = 0
        motion_2_frame = 0
        skill_2_frame = 0
        stage = 1 # 1: to skill-1, 2: to motion-2, 3: to skill-2
        for i in range(pcds.shape[0]):
            object_pcd = self.get_objects_pcd_from_sam_mask(pcds[i], demo_idx, "object")
            target_pcd = self.get_objects_pcd_from_sam_mask(pcds[i], demo_idx, "target")
            # [DEBUG] 输出已去除
            # ...existing code...
            if stage == 1:
                if distance_mode == "ee2pcd":            
                    if self.average_distance_to_point_cloud(ee_poses[i], object_pcd) <= threshold_1:
                        # visualizer.visualize_pointcloud(pcds[i])
                        stage = 2
                        skill_1_frame = i
                        object_origin_pcd = object_pcd
                elif distance_mode == "pcd2pcd":
                    obj_bbox = self.pcd_bbox(object_pcd)
                    tar_bbox = self.pcd_bbox(target_pcd)
                    source_pcd = pcds[i].copy()
                    _, _, ee_pcd = self.pcd_divide(source_pcd, [obj_bbox, tar_bbox])
                    if self.chamfer_distance(object_pcd, ee_pcd) <= threshold_1:
                        stage = 2
                        skill_1_frame = i
                        object_origin_pcd = object_pcd
                        
            elif stage == 2:
                if distance_mode == "ee2pcd":
                    if self.average_distance_to_point_cloud(ee_poses[i], object_origin_pcd) >= threshold_2:
                        stage = 3
                        motion_2_frame = i
                        # visualizer.visualize_pointcloud(pcds[i])
                elif distance_mode == "pcd2pcd":
                    tar_bbox = self.pcd_bbox(target_pcd)
                    source_pcd = pcds[i].copy()
                    _, ee_obj_pcd = self.pcd_divide(source_pcd, [tar_bbox])
                    if self.chamfer_distance(ee_obj_pcd, object_origin_pcd) >= threshold_2:
                        stage = 3
                        motion_2_frame = i
                        
            elif stage == 3:
                if distance_mode == "ee2pcd":
                    if self.average_distance_to_point_cloud(ee_poses[i], target_pcd) <= threshold_3:
                        skill_2_frame = i
                        # visualizer.visualize_pointcloud(pcds[i])
                        break
                elif distance_mode == "pcd2pcd":
                    tar_bbox = self.pcd_bbox(target_pcd)
                    source_pcd = pcds[i].copy()
                    _, ee_obj_pcd = self.pcd_divide(source_pcd, [tar_bbox])
                    if self.chamfer_distance(ee_obj_pcd, target_pcd) <= threshold_3:
                        skill_2_frame = i
                        break
        print(f"Stage 1: {skill_1_frame}, Pre-2: {motion_2_frame}, Stage 2: {skill_2_frame}")
        return skill_1_frame, motion_2_frame, skill_2_frame

    def two_stage_augment(self, n_demos, render_video=False, gen_mode='random'):
        """
        An implementation of the DemoGen augmentation process for manipulation tasks involving two objects. More specifically, the task contains
            4 sub-stages: (1) Motion-1, (2) Skill-1, (3) Motion-2, (4) Skill-2.
        TODO: Refactor the code to support tasks involving any number of objects and manipulation stages.
        """
        # Prepare translation vectors
        trans_vectors = []      # [n_demos, 6 (obj_xyz + targ_xyz)]
        if gen_mode == 'random':
            for _ in range(n_demos):
                obj_xyz = self.generate_trans_vectors(self.object_trans_range, 1, mode="random")[0]
                targ_xyz = self.generate_trans_vectors(self.target_trans_range, 1, mode="random")[0]
                trans_vectors.append(np.concatenate([obj_xyz, targ_xyz], axis=0))
        elif gen_mode == 'grid':
            def check_fourth_power(arr):
                fourth_roots = np.power(arr, 1/4)
                return np.isclose(fourth_roots, np.round(fourth_roots))
            assert check_fourth_power(n_demos), "n_demos must be a fourth power"
            sqrt_n_demos = int(np.sqrt(n_demos))
            obj_xyz = self.generate_trans_vectors(self.object_trans_range, sqrt_n_demos, mode="grid")
            targ_xyz = self.generate_trans_vectors(self.target_trans_range, sqrt_n_demos, mode="grid")
            for o_xyz in obj_xyz:
                for t_xyz in targ_xyz:
                    trans_vectors.append(np.concatenate([o_xyz, t_xyz], axis=0))
        else:
            raise NotImplementedError

        generated_episodes = []
        # For every source demo
        for i in range(self.n_source_episodes):
            cprint(f"Generating demos for source demo {i}", "blue")
            source_demo = self.replay_buffer.get_episode(i)
            pcds = source_demo["point_cloud"]
            
            if self.use_manual_parsing_frames:
                skill_1_frame = self.parsing_frames["skill-1"]
                motion_2_frame = self.parsing_frames["motion-2"]
                skill_2_frame = self.parsing_frames["skill-2"]
            else:
                ee_poses = source_demo["state"][:, :3]
                skill_1_frame, motion_2_frame, skill_2_frame = self.parse_frames_two_stage(pcds, i, ee_poses)
            print(f"Skill-1: {skill_1_frame}, Motion-2: {motion_2_frame}, Skill-2: {skill_2_frame}")
            
            pcd_obj = self.get_objects_pcd_from_sam_mask(pcds[0], i, "object")
            pcd_tar = self.get_objects_pcd_from_sam_mask(pcds[0], i, "target")
            # If either object or target point cloud is empty after masking, skip this source demo
            if pcd_obj is None or pcd_obj.shape[0] == 0 or pcd_tar is None or pcd_tar.shape[0] == 0:
                cprint(f"[WARN] Skipping source demo {i} because object/target point cloud is empty after SAM mask.", 'yellow')
                continue

            obj_bbox = self.pcd_bbox(pcd_obj)
            tar_bbox = self.pcd_bbox(pcd_tar)

            # Generate demos according to translation vectors
            for trans_vec in tqdm(trans_vectors):
                obj_trans_vec = trans_vec[:3]
                tar_trans_vec = trans_vec[3:6]
                ############## start of generating one episode ##############
                current_frame = 0

                traj_states = []
                traj_actions = []
                traj_pcds = []
                trans_sofar = np.zeros(3)

                ############# stage {motion-1} starts #############
                trans_togo = obj_trans_vec.copy()
                source_demo = self.replay_buffer.get_episode(i)
                start_pos = source_demo["state"][0][:3] - source_demo["action"][0][:3] # home state
                end_pos = source_demo["state"][skill_1_frame-1][:3] + trans_togo
                
                if self.use_linear_interpolation:
                    step_action = (end_pos - start_pos) / skill_1_frame
                else:
                    xy_stage_frame = skill_1_frame
                    step_actions = []
                    z_action = end_pos[2] - start_pos[2]
                    xy_action = end_pos[:2] - start_pos[:2]
                    
                    if z_action != 0:
                        z_action = np.sign(z_action) * round(np.abs(z_action), 3)
                        z_step_num = int(np.abs(z_action) / 0.015)
                        for _ in range(z_step_num):
                            step_actions.append(np.array([0, 0, np.sign(z_action) * 0.015]))
                            xy_stage_frame -= 1
                    
                    if xy_stage_frame > 0:
                        action = xy_action / xy_stage_frame
                        for _ in range(xy_stage_frame):
                            step_actions.append(np.array([*action, 0]))
                            
                    # inverse the step_actions
                    step_actions = step_actions[::-1]
                
                for j in range(skill_1_frame):
                    if not self.use_linear_interpolation:
                        step_action = step_actions[j]
                        
                    source_action = source_demo["action"][current_frame]
                    traj_actions.append(np.concatenate([step_action, source_action[3:]], axis=0))
                    trans_this_frame = step_action - source_action[:3]
                    
                    trans_sofar[:2] += trans_this_frame[:2] # for x y only

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_obj, pcd_tar, pcd_robot = self.pcd_divide(source_pcd, [obj_bbox, tar_bbox])
                    # visualizer.visualize_pointcloud(pcd_robot)
                    pcd_obj = self.pcd_translate(pcd_obj, obj_trans_vec)
                    pcd_tar = self.pcd_translate(pcd_tar, tar_trans_vec)
                    pcd_robot = self.pcd_translate(pcd_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_robot, pcd_obj, pcd_tar], axis=0))

                    current_frame += 1
                ############## stage {motion-1} ends #############
                
                ############# stage {skill-1} starts #############
                is_stage_motion2 = current_frame >= motion_2_frame
                while not is_stage_motion2:
                    action = source_demo["action"][current_frame].copy()
                    traj_actions.append(action)

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_tar, pcd_obj_robot = self.pcd_divide(source_pcd, [tar_bbox])
                    pcd_tar = self.pcd_translate(pcd_tar, tar_trans_vec)
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_obj_robot, pcd_tar], axis=0))

                    current_frame += 1
                    is_stage_motion2 = current_frame >= motion_2_frame
                ############## stage {skill-1} ends #############

                ############# stage {motion-2} starts #############
                trans_togo = tar_trans_vec - obj_trans_vec
                start_pos = source_demo["state"][motion_2_frame][:3] - source_demo["action"][motion_2_frame][:3]
                end_pos = source_demo["state"][skill_2_frame-1][:3] + trans_togo
                
                if self.use_linear_interpolation:
                    step_action = (end_pos - start_pos) / (skill_2_frame - motion_2_frame)
                else:
                    xy_stage_frame = skill_2_frame - motion_2_frame
                    step_actions = []
                    z_action = end_pos[2] - start_pos[2]
                    xy_action = end_pos[:2] - start_pos[:2]
                    
                    if z_action != 0:
                        z_action = np.sign(z_action) * round(np.abs(z_action), 3)
                        z_step_num = int(np.abs(z_action) / 0.015)
                        for _ in range(z_step_num):
                            step_actions.append(np.array([0, 0, np.sign(z_action) * 0.015]))
                            xy_stage_frame -= 1
                    
                    if xy_stage_frame > 0:
                        action = xy_action / xy_stage_frame
                        for _ in range(xy_stage_frame):
                            step_actions.append(np.array([*action, 0]))

                for k in range(skill_2_frame - motion_2_frame):
                    if not self.use_linear_interpolation:
                        step_action = step_actions[k]
                    source_action = source_demo["action"][current_frame]
                    traj_actions.append(np.concatenate([step_action, source_action[3:]], axis=0))
                    trans_this_frame = step_action - source_action[:3]
                    
                    trans_sofar[:2] += trans_this_frame[:2] # for x y only

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)

                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_tar, pcd_obj_robot = self.pcd_divide(source_pcd, [tar_bbox])
                    pcd_tar = self.pcd_translate(pcd_tar, tar_trans_vec)
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_obj_robot, pcd_tar], axis=0))

                    current_frame += 1
                ############## stage {motion-2} ends #############
                    
                ############# stage {skill-2} starts #############
                later_frames = self.translate_all_frames(source_demo, tar_trans_vec, current_frame)
                ############# stage {skill-2} ends #############

                generated_episode = {
                    "state": np.concatenate([traj_states, later_frames["state"]], axis=0) if len(traj_states) > 0 else later_frames["state"],
                    "action": np.concatenate([traj_actions, later_frames["action"]], axis=0) if len(traj_actions) > 0 else later_frames["action"],
                    "point_cloud": np.concatenate([traj_pcds, later_frames["point_cloud"]], axis=0) if len(traj_pcds) > 0 else later_frames["point_cloud"]
                }
                generated_episodes.append(generated_episode)

                if render_video:
                    vfunc = np.vectorize("{:.3f}".format)
                    video_name = f"{i}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}]_tar[{np.round(tar_trans_vec[0], 3)},{np.round(tar_trans_vec[1], 3)}].mp4"
                    video_path = os.path.join(self.data_root, "videos", self.source_name, self.gen_name, video_name)
                    self.point_cloud_to_video(generated_episode["point_cloud"], video_path, elev=20, azim=30)
                # self._examine_episode(generated_episode, aug_setting, i, obj_trans_vec, tar_trans_vec)
                ############## end of generating one episode ##############

        # save the generated episodes
        save_path = os.path.join(self.data_root, "datasets", "generated", f"{self.source_name}_{self.gen_name}_{n_demos}.zarr")
        self.save_episodes(generated_episodes, save_path)
        
    def parse_frames_one_stage(self, pcds, demo_idx, ee_poses, distance_mode="pcd2pcd", threshold_1=0.23):
        assert distance_mode in ["ee2pcd", "pcd2pcd"]
        for i in range(pcds.shape[0]):
            object_pcd = self.get_objects_pcd_from_sam_mask(pcds[i], demo_idx, "object")
            if distance_mode == "pcd2pcd":
                obj_bbox = self.pcd_bbox(object_pcd)
                source_pcd = pcds[i].copy()
                _, pcd_ee = self.pcd_divide(source_pcd, [obj_bbox])
                if self.chamfer_distance(pcd_ee, object_pcd) <= threshold_1:
                    print(f"Stage starts at frame {i}")
                    start_frame = i
                    break
            elif distance_mode == "ee2pcd":
                if self.average_distance_to_point_cloud(ee_poses[i], object_pcd) <= threshold_1:
                    print(f"Stage starts at frame {i}")
                    start_frame = i
                    break
        return start_frame

    def one_stage_augment(self, n_demos, render_video=False, gen_mode='random'):
        # Prepare translation vectors
        trans_vectors = []      # [n_demos, 6 (obj_xyz + targ_xyz)]
        if gen_mode == 'random':
            trans_vectors = self.generate_trans_vectors(self.object_trans_range, n_demos, mode="random")
        elif gen_mode == 'grid':
            def check_squared_number(arr):
                return np.isclose(np.sqrt(arr), np.round(np.sqrt(arr)))
            assert check_squared_number(n_demos), "n_demos must be a squared number"
            trans_vectors = self.generate_trans_vectors(self.object_trans_range, n_demos, mode="grid")


        generated_episodes = []

        for i in tqdm(range(self.n_source_episodes)):
            cprint(f"Generating demos for source demo {i}", "blue")
            source_demo = self.replay_buffer.get_episode(i)
            pcds = source_demo["point_cloud"]
            # visualizer.visualize_pointcloud(pcds[0])
            
            if self.use_manual_parsing_frames:
                skill_1_frame = self.parsing_frames["skill-1"]
            else:
                ee_poses = source_demo["state"][:, :3]
                skill_1_frame = self.parse_frames_one_stage(pcds, i, ee_poses)
            print(f"Skill-1: {skill_1_frame}")
            
            pcd_obj = self.get_objects_pcd_from_sam_mask(pcds[0], i, "object")
            obj_bbox = self.pcd_bbox(pcd_obj)

            for obj_trans_vec in tqdm(trans_vectors):
                ############## start of generating one episode ##############
                current_frame = 0

                traj_states = []
                traj_actions = []
                traj_pcds = []
                trans_sofar = np.zeros(3)

                ############# stage {motion-1} starts #############
                trans_togo = obj_trans_vec.copy()
                source_demo = self.replay_buffer.get_episode(i)
                start_pos = source_demo["state"][0][:3] - source_demo["action"][0][:3] # home state
                end_pos = source_demo["state"][skill_1_frame-1][:3] + trans_togo
                
                if self.use_linear_interpolation:
                    step_action = (end_pos - start_pos) / skill_1_frame
                else:
                    xy_stage_frame = skill_1_frame
                    step_actions = []
                    z_action = end_pos[2] - start_pos[2]
                    xy_action = end_pos[:2] - start_pos[:2]
                    
                    if z_action != 0:
                        z_action = np.sign(z_action) * round(np.abs(z_action), 3)
                        z_step_num = int(np.abs(z_action) / 0.015)
                        for _ in range(z_step_num):
                            step_actions.append(np.array([0, 0, np.sign(z_action) * 0.015]))
                            xy_stage_frame -= 1
                    
                    if xy_stage_frame > 0:
                        action = xy_action / xy_stage_frame
                        for _ in range(xy_stage_frame):
                            step_actions.append(np.array([*action, 0]))
                            
                    # inverse the step_actions
                    step_actions = step_actions[::-1]
                
                for j in range(skill_1_frame):
                    if not self.use_linear_interpolation:
                        step_action = step_actions[j]
                        
                    source_action = source_demo["action"][current_frame]
                    traj_actions.append(np.concatenate([step_action, source_action[3:]], axis=0))
                    trans_this_frame = step_action - source_action[:3]
                    trans_sofar[:2] += trans_this_frame[:2] # for x y only

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_obj, pcd_robot = self.pcd_divide(source_pcd, [obj_bbox])
                    pcd_obj = self.pcd_translate(pcd_obj, obj_trans_vec)
                    pcd_robot = self.pcd_translate(pcd_robot, trans_sofar)
                    traj_pcds.append(np.concatenate([pcd_robot, pcd_obj], axis=0))

                    current_frame += 1
                ############## stage {motion-1} ends #############
                num_frames = source_demo["state"].shape[0]
                ############# stage {skill-1} starts #############
                while current_frame < num_frames:
                    action = source_demo["action"][current_frame].copy()
                    traj_actions.append(action)

                    # "state" and "point_cloud" consider the accumulated translation
                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans_sofar
                    traj_states.append(state)
                    
                    pcd_obj_robot = source_demo["point_cloud"][current_frame].copy()
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                    traj_pcds.append(pcd_obj_robot)

                    current_frame += 1
                    ############## stage {skill-1} ends #############

                generated_episode = {
                    "state": traj_states,
                    "action": traj_actions,
                    "point_cloud": traj_pcds
                }
                generated_episodes.append(generated_episode)

                if render_video:
                    vfunc = np.vectorize("{:.3f}".format)
                    video_name = f"{i}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}].mp4"
                    video_path = os.path.join(self.data_root, "videos", self.source_name, self.gen_name, video_name)
                    self.point_cloud_to_video(generated_episode["point_cloud"], video_path, elev=20, azim=30)
                ############## end of generating one episode ##############
        
        # save the generated episodes
        save_path = os.path.join(self.data_root, "datasets", "generated", f"{self.source_name}_{self.gen_name}_{n_demos}.zarr")
        self.save_episodes(generated_episodes, save_path)

    def _examine_episode(self, episode, aug_setting, episode_id, obj_trans, tar_trans):
        """
        Examine the episode to see if the point cloud is correct
        """
        cprint(f"Examine episode {episode_id}", "green")
        
        debug_save_dir = f"/home/zhengrong/data/dp3/debug/demo_generation/{self.demo_name}/{aug_setting}/{episode_id}-{vfunc(obj_trans)}-{vfunc(tar_trans)}"
        os.makedirs(debug_save_dir, exist_ok=True)
        cprint(f"Saving episode examination to {debug_save_dir}", "yellow")
        vis = visualizer.Visualizer()
        ep_len = episode["state"].shape[0]
        pcds = []
        for i in range(0, 200, 1):
            # print(f"Frame {i}")
            cprint(f"action {i}: {vfunc(episode['action'][i])}", "blue")
            cprint(f"state {i}: {vfunc(episode['state'][i][:3])}", "blue")
            pcd = episode["point_cloud"][i]
            pcds.append(pcd)
            vis.save_visualization_to_file(pointcloud=pcd, file_path=os.path.join(debug_save_dir, f"frame_{i}.html"))

        # vis.preview_in_open3d(pcds)

    def _examine_actions(self, demo_trajectory):
        vfunc = np.vectorize("{:.2e}".format)
        actions = demo_trajectory["action"][:30]
        ee_actions = actions[:, :3]
        intensity = np.linalg.norm(ee_actions, axis=1)
        print(f"ee_actions: {vfunc(ee_actions)}")
        print(f"inensity: {vfunc(intensity)}")

    def save_episodes(self, generated_episodes, save_dir):
        # Save each generated episode using HDF5 layout compatible with source data
        import h5py
        base_dir = save_dir.replace('.zarr', '_episodes')
        os.makedirs(base_dir, exist_ok=True)
        cprint(f"Saving each episode to {base_dir}", "green")
        # determine starting index by scanning existing files (so numbering is continuous)
        import glob
        existing = glob.glob(os.path.join(base_dir, 'episode*.hdf5'))
        start_idx = 0
        if existing:
            # parse existing indices like 'episode4.hdf5' or 'episode_4.hdf5' and continue after the max
            try:
                import re
                nums = []
                for p in existing:
                    m = re.search(r'episode(?:_)?(\d+)\.hdf5$', os.path.basename(p))
                    if m:
                        nums.append(int(m.group(1)))
                if nums:
                    start_idx = max(nums) + 1
                else:
                    start_idx = len(existing)
            except Exception:
                start_idx = len(existing)

        for rel_idx, ep in enumerate(generated_episodes):
            idx = start_idx + rel_idx
            ep_agent_pos = np.array(ep["state"])  # shape: (T, state_dim)
            ep_point_cloud = np.array(ep["point_cloud"])  # shape: (T, N, 6)
            ep_action = np.array(ep["action"])  # shape: (T, action_dim)
            T = ep_agent_pos.shape[0]
            ep_episode_ends = np.array([T])
            ep_path = os.path.join(base_dir, f'episode{idx}.hdf5')

            # Create HDF5 with keys similar to source collector
            with h5py.File(ep_path, 'w') as f:
                # endpose / state
                f.create_dataset('endpose', data=ep_agent_pos, compression='gzip')

                # joint_action group: prefer storing 'vector' to match source
                ja_grp = f.create_group('joint_action')
                try:
                    ja_grp.create_dataset('vector', data=ep_action, compression='gzip')
                except Exception:
                    # fallback: store as plain dataset under joint_action
                    f.create_dataset('joint_action', data=ep_action, compression='gzip')

                # loop_counter and loop_times
                loop_counter = np.arange(T, dtype=np.int64)
                f.create_dataset('loop_counter', data=loop_counter, compression='gzip')
                f.create_dataset('loop_times', data=np.array(T, dtype=np.int32))

                # object_pos: placeholder (shape T x 7)
                obj_grp = f.create_group('object_pos')
                obj_grp.create_dataset('obj0', data=np.zeros((T, 7), dtype=np.float32), compression='gzip')

                # observation group placeholders for head/left/right cameras
                obs_grp = f.create_group('observation')
                for cam in ['head_camera', 'left_camera', 'right_camera']:
                    cg = obs_grp.create_group(cam)
                    # cam2world_gl: (T,4,4)
                    cg.create_dataset('cam2world_gl', data=np.zeros((T, 4, 4), dtype=np.float32), compression='gzip')
                    # depth: placeholder (T, H, W) use (240,320) as in source if available
                    try:
                        cg.create_dataset('depth', data=np.zeros((T, 240, 320), dtype=np.float64), compression='gzip')
                    except Exception:
                        # in case too large, create a tiny placeholder
                        cg.create_dataset('depth', data=np.zeros((T, 4, 4), dtype=np.float64), compression='gzip')
                    cg.create_dataset('extrinsic_cv', data=np.zeros((T, 3, 4), dtype=np.float32), compression='gzip')
                    cg.create_dataset('intrinsic_cv', data=np.zeros((T, 3, 3), dtype=np.float32), compression='gzip')
                    # rgb: store empty bytes per-frame as placeholder
                    dt = h5py.string_dtype(encoding='utf-8')
                    rgb_ds = cg.create_dataset('rgb', shape=(T,), dtype=dt)
                    rgb_ds[:] = ['' for _ in range(T)]

                # pointcloud: name matches source ('pointcloud') and shape (T, N, 6)
                f.create_dataset('pointcloud', data=ep_point_cloud.astype(np.float32), compression='gzip')

                # episode_ends
                f.create_dataset('episode_ends', data=ep_episode_ends, compression='gzip')

            cprint(f'Saved: {ep_path}', 'cyan')

    @staticmethod
    def point_cloud_to_video(point_clouds, output_file, fps=15, *args, **kwargs):
        """
        Converts a sequence of point cloud frames into a video.

        Args:
            point_clouds (list): A list of (N, 6) numpy arrays representing the point clouds.
            output_file (str): The path to the output video file.
            fps (int, optional): The frames per second of the output video. Defaults to 15.
        """
        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        all_points = np.concatenate(point_clouds, axis=0)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        writer = imageio.get_writer(output_file, fps=fps)
        logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

        for frame, points in enumerate(point_clouds):
            ax.clear()
            color = points[:, 3:]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='.')

            ax.set_box_aspect([1.6, 2.2, 1])
            # 坐标轴范围与zarr可视化一致，直接用所有点的实际范围
            ax.set_xlim(np.min(all_points[:, 0]), np.max(all_points[:, 0]))
            ax.set_ylim(np.min(all_points[:, 1]), np.max(all_points[:, 1]))
            ax.set_zlim(np.min(all_points[:, 2]), np.max(all_points[:, 2]))

            ax.tick_params(axis='both', which='major', labelsize=8)
            formatter = FormatStrFormatter('%.2f')
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.zaxis.set_major_formatter(formatter)

            # 使用matplotlib默认视角，不设置elev和azim
            # ax.view_init()  # 可选：如需强制默认视角可加此行
            ax.text2D(0.05, 0.95, f'Frame: {frame}', transform=ax.transAxes, fontsize=14, 
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = img[..., :3]  # 只保留RGB通道
            writer.append_data(img)

        writer.close()
        plt.close(fig)

    @staticmethod
    def pcd_divide(pcd, bbox_list):
        """
        pcd: (n, 6)
        bbox_list: list of (2, 3), [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        :return: list of pcds
        """
        masks = []
        selected_pcds = []
        for bbox in bbox_list:
            assert np.array(bbox).shape == (2, 3)
            masks.append(np.all(pcd[:, :3] > bbox[0], axis=1) & np.all(pcd[:, :3] < bbox[1], axis=1))
        # add the rest of the points to the last mask
        masks.append(np.logical_not(np.any(masks, axis=0)))
        for mask in masks:
            selected_pcds.append(pcd[mask])
        total_selected = np.sum([len(p) for p in selected_pcds])
        if total_selected != pcd.shape[0]:
            # Fallback: ensure no points are lost due to boundary conditions or boolean casting
            cprint(f"[WARN] pcd_divide selected {total_selected} points but input has {pcd.shape[0]}. Adjusting masks to include remainder.", 'yellow')
            # recompute masks but make them mutually exclusive: assign points to first matching bbox
            masks = []
            assigned = np.zeros(pcd.shape[0], dtype=bool)
            for bbox in bbox_list:
                mask = np.all(pcd[:, :3] > bbox[0], axis=1) & np.all(pcd[:, :3] < bbox[1], axis=1)
                # remove already assigned points to avoid double counting
                mask = mask & (~assigned)
                masks.append(mask)
                assigned |= mask
            # remainder (points not assigned to any bbox)
            remainder = ~assigned
            masks.append(remainder)
            selected_pcds = [pcd[mask] for mask in masks]
        return selected_pcds

    @staticmethod
    def pcd_translate(pcd, trans_vec):
        """
        Translate the points with trans_vec
        pcd: (n, 6)
        trans_vec (3,)
        """
        pcd_ = pcd.copy()
        pcd_[:, :3] += trans_vec
        return pcd_

    @staticmethod
    def translate_all_frames(source_demo, trans_vec, start_frame=0):
        """
        Translate all frames in the source demo by trans_vec from start_frame
        """
        source_states = source_demo["state"]
        source_actions = source_demo["action"]
        source_pcds = source_demo["point_cloud"]
        # states = source_states[start_frame:] + np.array([*trans_vec, *trans_vec, *trans_vec])
        states = source_states[start_frame:].copy()
        states[:, :3] += trans_vec
        actions = source_actions[start_frame:]
        pcds = source_pcds[start_frame:].copy()   # [T, N_points, 6]
        pcds[:, :, :3] += trans_vec
        assert states.shape[0] == actions.shape[0] == pcds.shape[0] == source_states.shape[0] - start_frame
        return {
            "state": states,
            "action": actions,
            "point_cloud": pcds
        }

    @staticmethod
    def chamfer_distance(pcd1, pcd2):
        tree1 = cKDTree(pcd1[:, :3])
        tree2 = cKDTree(pcd2[:, :3])

        distances1 = [tree2.query(point[:3], k=1)[0] for point in pcd1]
        distances2 = [tree1.query(point[:3], k=1)[0] for point in pcd2]

        chamfer_dist = (np.mean(distances1) + np.mean(distances2)) / 2
        return chamfer_dist
    
    @staticmethod
    def average_distance_to_point_cloud(target_point, point_cloud):
        target_point = np.array(target_point)
        point_cloud = np.array(point_cloud)
        
        if point_cloud.shape[1] != target_point.shape[0]:
            point_cloud_coords = point_cloud[:, :3]
        else:
            point_cloud_coords = point_cloud
        
        distances = np.linalg.norm(point_cloud_coords - target_point, axis=1)
        average_distance = np.mean(distances)
        
        return average_distance
    
    @staticmethod
    def pcd_bbox(pcd, relax=False):
        min_vals = np.min(pcd[:, :3], axis=0)
        max_vals = np.max(pcd[:, :3], axis=0)
        if relax:
            min_vals -= 0.01
            max_vals += 0.01
            min_vals[2] = 0.0
        return np.array([min_vals, max_vals])

