auto_pipeline 使用说明
=====================

概述
----
`auto_pipeline.py` 是一个交互式小工具，用来把单个 HDF5 源轨迹导出为可视化帧、把首帧复制到 `data/sam_mask/<source>/0/` 供手动制作 SAM 掩码、等待人工准备掩码后再继续进行 gen_demo 等后续步骤。

快速上手（推荐）
----------------
1. 进入脚本目录：

```bash
cd /home/liaohaoran/code/DemoGen/demo_generation
```

2. 激活 `demogen` 环境（已在 `~/.bashrc` 中添加 `doge` 快捷）：

```bash
# 推荐：使用别名（已添加到 ~/.bashrc），在一个新的交互 shell 中运行：
doge
# 或者手动激活：
# conda activate demogen
```

3. 运行交互脚本：

```bash
python3 auto_pipeline.py
```

脚本流程（简要）
----------------
- 脚本会询问一个源 HDF5 的绝对路径（例如 `/path/to/episode0.hdf5`）。
- 导出该 HDF5 中的 RGB 帧到 HDF5 同目录（文件名 `000000.jpg`, `000001.jpg`, ...）。
- 将首帧复制到 `DemoGen/data/sam_mask/<h5_basename>/0/`，等待你在该目录下手动放置/编辑掩码文件（PNG/JPG）。脚本会在检测到掩码后继续。
- 脚本会暂停，等待你准备好 `demogen` 的配置（config）并按回车继续。
- 可在交互中提供用于运行 `gen_demo` 的命令，脚本会在该目录内执行该命令（注意：脚本不会自动修改你的 config）。
- 运行完成后，脚本会尝试根据生成目录写 `loop_times.txt`，并可选择调用描述生成脚本。

注意事项
--------
- 掩码步骤为手动交互：当前脚本不再尝试自动运行 `get_mask.py`（用户可手动运行或在准备目录置入掩码文件）。
- 若想恢复自动化 mask（需安装额外依赖，如 `lang_sam`），请确保 `demogen` 环境中已正确安装所有依赖。
- 我已修复 `~/.bashrc` 中的语法错误；如遇到启动时的报错请先重新打开一个终端以加载修复后的配置。

示例（测试）
-----------
下面是在本仓库中用作快速测试的命令（示例文件 `test_dummy.h5` 已被本地用于验证）：

```bash
# 在仓库的 demo_generation 目录下运行（会交互）
python3 auto_pipeline.py
# 当脚本询问 HDF5 路径时，输入：
# /home/liaohaoran/code/DemoGen/demo_generation/test_dummy.h5
# 然后按多次回车以接受默认值并跳过 gen_demo 步骤
```

常见问题
--------
- Q: 为什么脚本检测到的掩码是首帧（jpg）？
  A: 脚本只是检测目标目录内的图片文件以便给出提示。你可以覆盖该目录中的图片（用真实掩码文件替换）或手动放置掩码文件。

- Q: 我想脚本自动运行 `get_mask.py`，该怎么做？
  A: 我可以把 `get_mask.py` 改为接收输入/输出路径参数，并把 `auto_pipeline.py` 加回自动化选项；但这要求目标环境已正确安装 `lang_sam` 及其依赖（可能包括大型依赖，如 PyTorch）。如果需要，我可以帮你实现并执行安装（安装可能较耗时）。

文件位置
--------
- 脚本：`demo_generation/auto_pipeline.py`
- 使用说明：`demo_generation/USAGE.md`

如需我把这段说明合并回仓库根 README 或创建 PR，请告诉我。