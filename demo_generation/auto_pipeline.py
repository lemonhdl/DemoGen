#!/usr/bin/env python3
"""
auto_pipeline.py
-----------------
这个脚本是一个交互式的管道（pipeline）框架，用来把单个 HDF5 轨迹文件：

- 导出 RGB 帧（命名为 6 位零填充的 jpg，例如 `000000.jpg`）
- 将首帧复制到 `data/sam_mask/<h5_basename>/0/`（与仓库中 `get_mask.py` 的约定路径一致）
- 调用 SAM mask 生成脚本（`get_mask.py`），并在此处暂停以便人工编辑/放置配置
- 运行 `gen_demo.py`（由用户提供具体命令）来生成增强后的 episode
- 根据生成结果写入 `loop_times.txt`，并可选调用描述/指令生成脚本

注意：该脚本较为保守，不会自动修改你的 config 或强制安装依赖；对于 SAM 部分若缺少 `lang_sam`，请先安装或手动准备 mask 文件。

使用方式：
    cd DemoGen/demo_generation
    python3 auto_pipeline.py

脚本采用交互方式询问 HDF5 路径、get_mask.py 路径以及 gen_demo 的运行命令。
"""

import os
import shutil
import subprocess
import h5py
import io
import numpy as np
from PIL import Image
import cv2
import time
import shutil


def save_rgb_frames_from_h5(h5_path, out_dir, camera_preference=('head_camera', 'left_camera', 'right_camera')):
    """
    从 HDF5 中寻找 RGB 帧并保存为 6 位零填充的 JPG 文件。

    针对常见的数据布局做了几种查找策略：
    1. 优先查找 `observation/<camera>['rgb']`（按传入的 camera_preference 顺序）。
    2. 若未命中，则在 `observation` 下查找任何包含 `rgb` 的子 group。
    3. 最后回退到顶层的 `rgb` / `images` / `frames` 等候选名称。

    参数：
        h5_path: HDF5 路径
        out_dir: 输出目录（会被创建）
        camera_preference: 优先使用的相机组名顺序

    返回：保存的文件路径列表（按帧序）
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    with h5py.File(h5_path, 'r') as f:
        rgb_ds = None

        # 优先在 observation 下查找 camera.rgb
        if 'observation' in f:
            obs = f['observation']
            for cam in camera_preference:
                if cam in obs and 'rgb' in obs[cam]:
                    rgb_ds = obs[cam]['rgb']
                    break
            # 回退：找任何包含 rgb 的 group
            if rgb_ds is None:
                for name, item in obs.items():
                    if isinstance(item, h5py.Group) and 'rgb' in item:
                        rgb_ds = item['rgb']
                        break

        # 更后备的顶层候选项
        if rgb_ds is None:
            for candidate in ('rgb', 'images', 'frames'):
                if candidate in f:
                    rgb_ds = f[candidate]
                    break

        if rgb_ds is None:
            raise RuntimeError('在 HDF5 中未找到 rgb 数据。期望路径示例：observation/<camera>/rgb 或 顶层 rgb。')

        # 逐帧读取并保存为 JPG
        n = rgb_ds.shape[0]
        for i in range(n):
            v = rgb_ds[i]
            out_path = os.path.join(out_dir, f"{i:06d}.jpg")

            # 有些 HDF5 存储的是已经编码的 bytes（例如 jpeg bytes），有些则是 HxWxC 的 numpy 数组
            if isinstance(v, (bytes, bytearray, np.bytes_)) or (hasattr(v, 'tobytes') and getattr(v, 'dtype', None) is not None and v.dtype.kind == 'S'):
                # 尝试作为二进制图像解码（参考 hdf52rgb_all.py）
                try:
                    # 将可能的 bytes-like 对象转换为 bytes
                    b = bytes(v)
                    # 用 OpenCV 解码（得到 BGR），然后转换为 RGB 保存，确保通道顺序正确
                    arr = np.frombuffer(b, dtype=np.uint8)
                    cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if cv_img is None:
                        # fallback: 使用 PIL 直接解码
                        img = Image.open(io.BytesIO(b)).convert('RGB')
                        img.save(out_path, quality=95)
                    else:
                        # 严格遵循 hdf52rgb_all.py：直接保存 OpenCV 解码结果（不做通道重排）
                        Image.fromarray(cv_img).save(out_path, quality=95)
                except Exception:
                    # 退回到把 v 当作数组处理
                    arr = np.asarray(v)
                    Image.fromarray(arr).convert('RGB').save(out_path, quality=95)
            else:
                arr = np.asarray(v)
                # 如果元素是 bytes 字符串数组
                if arr.dtype.kind == 'S':
                    try:
                        b = arr.tobytes()
                        arr2 = np.frombuffer(b, dtype=np.uint8)
                        cv_img = cv2.imdecode(arr2, cv2.IMREAD_COLOR)
                        if cv_img is None:
                            Image.open(io.BytesIO(b)).convert('RGB').save(out_path, quality=95)
                        else:
                            # 严格遵循 hdf52rgb_all.py：直接保存 OpenCV 解码结果（不做通道重排）
                            Image.fromarray(cv_img).save(out_path, quality=95)
                    except Exception:
                        Image.fromarray(np.frombuffer(arr, dtype=np.uint8)).convert('RGB').save(out_path, quality=95)
                else:
                    # 正常数组 HxWxC
                    if arr.ndim == 3:
                        Image.fromarray(arr.astype('uint8')).convert('RGB').save(out_path, quality=95)
                    else:
                        raise RuntimeError('无法识别的 rgb 帧格式（既不是编码 bytes 也不是 HxWxC 数组）')

            saved.append(out_path)

    return saved


def copy_first_frame_to_sam(saved_frames, h5_path):
    """
    将导出的首帧复制到仓库约定的 sam_mask 路径：
        DemoGen/demo_generation/data/sam_mask/<h5_basename>/0/

    这样 `get_mask.py` 在默认行为下就能找到要处理的首帧。
    """
    base = os.path.splitext(os.path.basename(h5_path))[0]
    dest_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sam_mask', base, '0')
    os.makedirs(dest_dir, exist_ok=True)

    if not saved_frames:
        raise RuntimeError('没有可复制的帧，saved_frames 为空')

    shutil.copy2(saved_frames[0], os.path.join(dest_dir, os.path.basename(saved_frames[0])))
    return dest_dir


def run_get_mask_script(get_mask_py):
    """
    以子 shell 调用 `get_mask.py`。子 shell 会 `source ~/.bashrc`，以便可用 `doge` 等别名。

    返回值：脚本的返回码（0 表示成功）
    """
    cmd = f"bash -lc 'source ~/.bashrc >/dev/null 2>&1; python3 {get_mask_py}'"
    print('Running mask script:', cmd)
    rc = subprocess.call(cmd, shell=True)
    return rc


def write_looptimes_for_generated(gen_output_dir, out_loop_times_path):
    """
    遍历生成目录中的 `episode*.hdf5`，尝试读取 `loop_times` 并把它们写入一个单行的 `loop_times.txt` 文件（以空格分隔）。

    这是为了与上游使用 `loop_times.txt` 的流程兼容。
    """
    eps = sorted([p for p in os.listdir(gen_output_dir) if p.startswith('episode') and p.endswith('.hdf5')])
    loop_times = []

    for ep in eps:
        p = os.path.join(gen_output_dir, ep)
        try:
            with h5py.File(p, 'r') as f:
                if 'loop_times' in f:
                    lt = f['loop_times'][:]
                    if np.isscalar(lt) or getattr(lt, 'shape', ()) == ():
                        loop_times.append(int(lt))
                    else:
                        loop_times.append(int(lt[0]))
                elif 'loop_times' in f.attrs:
                    loop_times.append(int(f.attrs['loop_times']))
                else:
                    loop_times.append(1)
        except Exception:
            # 任何读取失败都退回为 1，保证文件可写
            loop_times.append(1)

    with open(out_loop_times_path, 'w') as fh:
        for lt in loop_times:
            fh.write(f"{lt} ")

    return out_loop_times_path


def main():
    print('交互式 DemoGen 开始...')

    # 安全输入封装：捕获 Ctrl+C (KeyboardInterrupt) 并重新提示输入
    def safe_input(prompt=''):
        while True:
            try:
                return input(prompt)
            except KeyboardInterrupt:
                # 首次 Ctrl+C：提示并等待 1 秒，在此期间若再次 Ctrl+C 则退出程序
                print('\n输入被中断（Ctrl+C）。若在 1 秒内再次按 Ctrl+C，则退出程序；或等待继续并重新输入...')
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    print('\n检测到二次 Ctrl+C，退出程序')
                    raise
                continue
            except EOFError:
                # 将 EOF (Ctrl+D) 视为空输入并返回空字符串
                print('\n检测到 EOF（Ctrl+D），视为空输入。')
                return ''

    # 1) 询问 HDF5 路径
    h5_input = safe_input('请复制粘贴一条源 HDF5 文件的绝对路径（输入 skip 表示已完成此步并跳过；留空则退出）：').strip()
    if not h5_input:
        print('已中止')
        return

    frames = []
    frames_dir = None
    if h5_input.lower() == 'skip':
        # 跳过 HDF5 导出，询问用户是否已有导出帧目录
        frames_dir = safe_input('你选择跳过 HDF5 导出。如果你已有导出的帧，请输入帧目录绝对路径（输入 skip 则继续跳过帧相关步骤）：').strip()
        if frames_dir and frames_dir.lower() != 'skip':
            if not os.path.exists(frames_dir):
                print('指定的帧目录不存在：', frames_dir)
                return
            frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            if not frames:
                print('在指定目录未找到图片文件，继续但没有可用帧')
        else:
            # frames_dir 为 skip 或为空，表示完全跳过帧相关的步骤
            frames = []
        # 询问用于后续写 loop_times 的 src_dir（默认取帧目录的父目录或当前目录）
        if frames_dir:
            default_src_dir = os.path.dirname(frames_dir)
        else:
            default_src_dir = os.getcwd()
        src_dir = safe_input(f'请输入源数据根目录（用于写 loop_times，默认: {default_src_dir}）：').strip() or default_src_dir
        h5_path = None
    else:
        h5_path = h5_input
        if not os.path.exists(h5_path):
            print('未找到文件：', h5_path)
            return
        src_dir = os.path.dirname(h5_path)

    # 根据 h5 路径推断 task_name 与 idx
    # 例如: /.../beat_block_hammer_loop/loop1-8-all/data/episode0.hdf5
    # task_name = basename(third ancestor) = beat_block_hammer_loop
    # idx = episode number extracted from filename (episode0 -> 0)
    def _infer_task_and_idx(path):
        p0 = os.path.dirname(path)  # parent (e.g., .../data)
        p1 = os.path.dirname(p0)    # e.g., .../loop1-8-all
        p2 = os.path.dirname(p1)    # e.g., .../beat_block_hammer_loop
        task_name = os.path.basename(p2) or os.path.basename(p1) or 'unknown'
        base = os.path.splitext(os.path.basename(path))[0]
        import re
        m = re.search(r'episode(\d+)', base)
        if m:
            idx = int(m.group(1))
        else:
            # fallback: try to find any digits in the basename
            m2 = re.search(r'(\d+)', base)
            idx = int(m2.group(1)) if m2 else 0
        return task_name, idx

    task_name, idx = _infer_task_and_idx(h5_path)
    # 可视化输出目录：DemoGen/data/sam_mask/<task_name>/<idx>/vis_all
    vis_out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sam_mask', task_name, str(idx), 'vis_all')

    # 2) 导出帧到 HDF5 同目录（避免额外路径管理）
    # 2) 导出帧或使用已有帧
    print('正在将帧保存到可视化目录 (vis_all) 或使用已有帧...')
    if h5_path:
        try:
            frames = save_rgb_frames_from_h5(h5_path, vis_out_dir)
            print('逐帧rgb可视化已保存到：', vis_out_dir)
        except Exception as e:
            print('提取帧失败：', e)
            return
    else:
        # h5 跳过的情况下，若 frames_dir 已设置则 frames 已被填充
        if not frames:
            print('未使用 HDF5 也未提供帧目录，跳过与帧相关的步骤。')

    if frames:
        print(f'已保存/找到帧数量: {len(frames)}，首帧: {frames[0]}')

    # 3) 把首帧复制到 sam_mask 的约定目录，供 get_mask.py 使用
    # 3) 把首帧复制到 sam_mask 的约定目录，供 get_mask.py 使用
    sam_dest = None
    copy_prompt = safe_input('准备复制首帧到 sam_mask 文件夹以便 get_mask.py 使用（按 Enter 执行；输入 skip 表示已完成并跳过此步）：').strip()
    if copy_prompt.lower() == 'skip':
        print('已跳过复制首帧步骤。')
    else:
        if frames:
            try:
                sam_dest = copy_first_frame_to_sam(frames, h5_path if h5_path else frames[0])
                print('首帧已复制到', sam_dest)
            except Exception as e:
                print('复制首帧失败：', e)
                # 继续，但 sam_dest 可能为 None
        else:
            # 没有可用帧
            provided = safe_input('当前没有可用帧，若你已手动复制首帧请输入该文件所在目录（或输入 skip 继续跳过）：').strip()
            if provided and provided.lower() != 'skip':
                if os.path.exists(provided):
                    sam_dest = provided
                else:
                    print('指定路径不存在：', provided)
                    sam_dest = None
            else:
                print('跳过复制首帧。')

    # 4) SAM mask 步骤：纯交互式，让用户在指定目录准备好掩码
    # 说明：脚本会把首帧复制到 sam_dest（由 copy_first_frame_to_sam 返回）。
    # 用户需在该目录中手动放置掩码文件（PNG/JPG），准备好后按回车继续。
    print('\n下一步：准备 SAM 掩码（mask）')
    print('首帧已复制到：', sam_dest)
    default_mask_dir = sam_dest
    mask_input = safe_input(f'请输入准备掩码的目录路径（默认: {default_mask_dir}；或输入 skip 表示掩码已准备好/跳过）：').strip()
    if mask_input.lower() == 'skip':
        mask_dir = None
        print('跳过掩码准备步骤。')
    else:
        mask_dir = mask_input or default_mask_dir
        if mask_dir is None:
            print('未指定掩码目录，继续但不会检测掩码文件。')
        else:
            if not os.path.exists(mask_dir):
                print('指定的掩码目录不存在，已创建：', mask_dir)
                os.makedirs(mask_dir, exist_ok=True)
            print('请在该目录放置掩码文件（例如 PNG/JPG），准备好后按回车继续。目录：', mask_dir)
            safe_input()

    # 检查掩码目录中是否存在掩码文件（png/jpg）以便提示用户
    mask_files = []
    try:
        for fname in os.listdir(mask_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_files.append(os.path.join(mask_dir, fname))
    except Exception:
        mask_files = []

    if mask_files:
        print(f'检测到 {len(mask_files)} 个掩码文件（示例: {mask_files[0]}）。')
    else:
        print('未检测到掩码文件。请确认你已将掩码放在目录中。')

    # 5) 暂停：让用户创建/调整 demogen 的 config 或其它手动必要文件
    cfg_input = safe_input('现在请为该源创建 demogen 配置并放置好，准备好后按回车继续（或输入 skip 表示已完成）：').strip()
    if cfg_input.lower() == 'skip':
        print('跳过 demogen 配置准备步骤。')

    # 6) 运行 gen_demo：使用仓库自带的 `gen_demo.sh`，并引导用户输入参数以减少拼写错误
    print('\n现在将使用仓库中的 `gen_demo.sh` 脚本来运行生成。')
    print('如果你更倾向于直接提供完整命令，也可以直接输入（留空则使用交互式参数）')
    direct_cmd = safe_input('若要直接提供完整命令请粘贴到此（输入 skip 跳过 gen_demo；留空使用交互式参数）：').strip()
    demo_dir = os.path.dirname(__file__)
    if direct_cmd.lower() == 'skip':
        print('跳过 gen_demo 执行。')
    elif direct_cmd:
        full = f"bash -lc 'cd {demo_dir} && source ~/.bashrc >/dev/null 2>&1; {direct_cmd}'"
        print('正在运行（直接命令）：', full)
        subprocess.call(full, shell=True)
    else:
        # 基于文件结构提供合理默认值
        default_task = task_name
        default_range = 'test'
        default_mode = 'random'
        default_n = '40'
        default_render = 'True'

        task = safe_input(f'任务名 `task`（对应仓库中 config 名称，默认: {default_task}）：').strip() or default_task
        gen_range = safe_input(f'生成范围 `gen_range`（例如: test，默认: {default_range}）：').strip() or default_range
        gen_mode = safe_input(f'生成模式 `gen_mode`（例如: random 或 grid，默认: {default_mode}）：').strip() or default_mode
        n_gen_per_source = safe_input(f'每源生成数量 `n_gen_per_source`（整数，默认: {default_n}）：').strip() or default_n
        render_video = safe_input(f'是否渲染视频 `render_video`（True/False，默认: {default_render}）：').strip() or default_render

        # 最终命令使用仓库里的 `gen_demo.sh` wrapper
        sh_cmd = f"sh gen_demo.sh {task} {gen_range} {gen_mode} {n_gen_per_source} {render_video}"

        # 优先使用 `conda run -n demogen` 来在指定 conda 环境中执行，避免找不到包
        # 如果系统 PATH 中找不到 conda，则回退到 eval/activate 的方式（会尝试 source ~/.bashrc）
        if shutil.which('conda'):
            print('检测到 `conda` 可用，使用 `conda run -n demogen` 执行命令。')
            # 使用列表形式避免 shell 转义问题
            rc = subprocess.call(['conda', 'run', '-n', 'demogen', '--no-capture-output', 'bash', '-lc', sh_cmd])
        else:
            full = f"bash -lc 'cd {demo_dir} && eval \"$(conda shell.bash hook)\" >/dev/null 2>&1 || true; source ~/.bashrc >/dev/null 2>&1; conda activate demogen >/dev/null 2>&1 && {sh_cmd}'"
            print('未在 PATH 中找到 `conda`，尝试通过 source ~/.bashrc + conda activate 来激活环境。将要执行：', full)
            rc = subprocess.call(full, shell=True)

        if rc != 0:
            print(f'警告：gen_demo 执行返回非零退出码：{rc}。请在终端手动运行下面的命令以调试：')
            if shutil.which('conda'):
                print(f"conda run -n demogen bash -lc '{sh_cmd}'")
            else:
                print(f"cd {demo_dir} && eval \"$(conda shell.bash hook)\"; conda activate demogen; {sh_cmd}")
        if rc != 0:
            print(f'警告：gen_demo 脚本返回非零退出码：{rc}。如果是在未激活 conda 环境下运行，考虑手动运行：\n')
            print(f'  cd {demo_dir} && conda activate demogen && sh gen_demo.sh {task} {gen_range} {gen_mode} {n_gen_per_source} {render_video}')

    # 7) 后处理：尝试根据生成的 episode 写 loop_times.txt
    gen_root = os.path.join(src_dir, 'datasets', 'generated')
    gen_dirs = []
    if os.path.exists(gen_root):
        gen_dirs = sorted([os.path.join(gen_root, d) for d in os.listdir(gen_root)], key=os.path.getmtime)

    if gen_dirs:
        last = gen_dirs[-1]
        if last.endswith('_episodes'):
            ep_dir = last
        else:
            cand = [os.path.join(last, f) for f in os.listdir(last) if f.endswith('_episodes')]
            ep_dir = cand[0] if cand else last
    else:
        ep_dir = os.path.join(src_dir, 'datasets', 'generated')

    # 7) 后处理：写 loop_times，可由用户跳过
    do_loops = safe_input(f'是否尝试写入 loop_times.txt（会在 {src_dir} 下写入；输入 skip 跳过，按回车执行）：').strip()
    if do_loops.lower() == 'skip':
        print('跳过写入 loop_times.txt。')
    else:
        print('尝试根据以下生成的 episode 写入 loop_times.txt：', ep_dir)
        loop_times_path = os.path.join(src_dir, 'loop_times.txt')
        try:
            write_looptimes_for_generated(ep_dir, loop_times_path)
            print('已写入 loop_times 到', loop_times_path)
        except Exception as e:
            print('写入 loop_times 失败：', e)

    # 8) 可选：调用描述/指令生成脚本
    do_desc = safe_input('是否运行指令生成（会调用 description/gen_episode_instructions.sh）（输入 skip 跳过；输入 y 运行）？[y/N]：').strip().lower()
    if do_desc == 'skip' or do_desc == 'n' or do_desc == '':
        print('跳过指令生成步骤。')
    elif do_desc == 'y':
        task_name = safe_input('输入 task_name（用于描述脚本）：').strip()
        task_config = safe_input('输入 task_config（用于描述脚本）：').strip()
        language_num = safe_input('输入 language_num（例如：1）：').strip() or '1'
        # 说明：描述脚本位置相对于仓库结构，若你的实际路径不同，请手动调整
        desc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RoboTwin', 'description'))
        cmd = f"bash -lc 'cd {desc_dir} && bash gen_episode_instructions.sh {task_name} {task_config} {language_num}'"
        print('正在运行：', cmd)
        subprocess.call(cmd, shell=True)

    print('管道执行完成。')


if __name__ == '__main__':
    main()
