# RTMW 高尔夫 P1–P10 工作流

使用 **RTMW-X 2D** 权重 `models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth` 预测普通视频中的人体：YOLOX 人物框 → RTMW 2D 人体 + **SwingTrace YOLO11m 杆身／杆头／握把框** → 球检测与球杆跟踪 → MixSTE 17+2 → 原有 P1–P10 规则。当前入口不加载 RTMW3D、球杆 ViTPose 或球杆分割模型。选帧定义和阈值见根目录 `FRAME_RECOGNITION.md`；检测框的适配与逐帧依据见本目录 [FRAME_RULES.md](FRAME_RULES.md)。

## 安装与自检

在仓库根目录使用 Windows PowerShell，推荐 Python 3.12：

```powershell
powershell -ExecutionPolicy Bypass -File rtmw_workflow/scripts/setup.ps1
.\rtmw_workflow\.venv\Scripts\python.exe -m rtmw_workflow self-test
```

环境安装到 `rtmw_workflow/.venv`；源码下载到 `rtmw_workflow/vendor`。安装需要联网，正式推理使用本地权重。需要 `models/swingtrace-yolo11m.pt`，以及已有 RTMW、YOLOX、MixSTE、YOLOE 和初始 ROI 的 YOLO26n。`self-test` 仅使用合成数据及一张空白图，不读取视频。

可选 NVIDIA GPU 加速：

```powershell
powershell -ExecutionPolicy Bypass -File rtmw_workflow/scripts/setup.ps1 -Cuda
.\rtmw_workflow\.venv\Scripts\python.exe -m rtmw_workflow self-test --club-device cuda
.\rtmw_workflow\.venv\Scripts\python.exe -m rtmw_workflow batch --input-dir pro_refs --club-device cuda
```

`--club-device cuda` 将 SwingTrace 球杆检测放到 GPU，使用 FP32、640 输入、每帧一次全图预测；RTMW 的设备由 `--pose-device` 单独控制。默认均为 CPU。CUDA 不可用时明确报错。当前已完成接入自检，尚未运行 SwingTrace 的完整视频质量或速度验证；旧 ViTPose 计时不代表新模型性能。

官方 MMPose 固定提交：`759b39c13fea6ba094afc1fa932f51dc1b11cbf9`。2D X 配置为 `rtmw-x_8xb704-270e_cocktail14-256x192.py`，输入宽 192、高 256，输出 COCO-WholeBody 133 点；保留官方 `flip_test=True`。加载使用 `strict=True`，已核验 836 项权重完全匹配。现有独立环境即可运行，无需重新安装依赖。

`requirements.txt` 自动读取 `constraints.txt` 固定依赖版本。YAPF 的配置缓存保存在本目录 `.cache`，避免受限环境下写入用户配置目录造成长时间等待。

Windows CPU 使用 `mmcv-lite`。`official.py` 仅加载该模型需要的官方模块，跳过其他架构和训练数据集的包级批量导入，因而不需要 MMCV CUDA 扩展、MMDetection、chumpy 或 xtcocotools。模型、仿射变换、预处理、SimCC 解码和坐标还原均来自未修改的官方源码；不提供替代算子。这是 RTMW 专用推理环境，不用于训练或运行其他 MMPose 模型。

本入口使用官方 `TopdownPoseEstimator`、`RTMWHead` 和 `SimCCLabel`，经 LoadImage、GetBBoxCenterScale、TopdownAffine、PackPoseInputs 进行推理。无需相机内参、根深度或 RTMPose3D 项目扩展。

## 运行

```powershell
# 单段
.\rtmw_workflow\.venv\Scripts\python.exe -m rtmw_workflow video pro_refs/005.mp4

# 根目录全部视频，不递归读取图片合集
.\rtmw_workflow\.venv\Scripts\python.exe -m rtmw_workflow batch --input-dir pro_refs

# 指定视角、惯用手、线程及输出目录
.\rtmw_workflow\.venv\Scripts\python.exe -m rtmw_workflow video pro_refs/008.mp4 --view face_on --handedness right --cpu-threads 8 --output-dir rtmw_workflow/outputs/example
```

批量运行可用 `--priority w03.mp4 w09.mp4 008.mp4 009.mp4` 优先检查短片、正面和多人物素材；仍会处理目录中的全部视频。`batch_summary.json` 持续更新 `status`、`current_video` 和已完成结果；结束时自动生成 `validation.json`。可重新审计已完成结果，无需再次推理：

```powershell
.\rtmw_workflow\.venv\Scripts\python.exe rtmw_workflow/scripts/audit_results.py rtmw_workflow/outputs/validation_005_cuda
```

`--checkpoint` 指定其他兼容 X 权重；`--golfer-box X1 Y1 X2 Y2` 指定原图像素中的球手区域。默认自动选 ROI，并使用已有 YOLOX 连续性与骨架恢复机制。YOLO26n 只用于初始 ROI，不融合或补充人体关节。

默认 CPU、8 线程，`--view` 和 `--handedness` 默认为 `auto`。输出目录必须为空或不存在，避免覆盖已有结果。默认使用北京时间批次名：`rtmw_workflow/outputs/<批次>/<视频名>/`。

## 输出与坐标

| 文件 | 内容 |
| --- | --- |
| `wholebody.npz` | 每帧 133 点：`image_2d`、`confidence`、`joints`、`frame_index`、`times_ms`、`timestamp_fallback` |
| `wholebody.json` | 模型版本、坐标说明、检测框、裁剪恢复记录与缺失统计 |
| `swingtrace_detections.json` | 三类原始检测框、置信度、球手关联结果、框中心代理点及缺失原因 |
| `frame_evidence.json` | P1–P10 实际来源、规则说明、三类选中框、杆身代理倾角、杆头与球距离、P7 诊断 |
| `rtmw_2d.mp4` | 完整 133 点二维骨架叠加视频 |
| `P1.jpg` … `P10.jpg` | 原有流程的二维人体与球杆关键帧 |
| `ten_frames.jpg` | 十帧总览 |
| `identified_frames.json` / `.npz` | 选帧、时间、置信度、判定来源与对应姿态 |
| `*_3d.json` / `.npz`、`P*_3d.jpg` | 原有 MixSTE 流程与球杆、击球诊断 |
| `label_comparison.json` | 与仓库已有部分人工标注区间的帧差；未标注阶段不计分 |
| `run_status.json` | 单段状态、耗时、异常详情 |
| 批次根目录 `batch_summary.json` | 全批次成功／失败汇总 |

`image_2d` 取官方 2D 估计器已经还原到输入图像的 `pred_instances.keypoints`，只再加外层球手裁剪偏移，单位为原图像素，x 向右、y 向下。COCO 前 17 点沿用现有人体映射；手指和面部点保存并显示。检测失败用 NaN 坐标、零置信度保留。适配器明确拒绝三维形状，防止误读旧模型结果。

新 `wholebody.json` 使用 `schema_version=2`、`native_3d_available=false`；NPZ 不再生成 `camera_3d`。下游选帧所用三维仍由现有 MixSTE 17+2 提供。历史 RTMW3D 输出保留在原目录，不能与新模型结果混用。

`times_ms` 优先取视频解码时间戳；无效或不递增时按 FPS 补一步，并记录 fallback。原选帧模块内部沿用原有 FPS 时间约定；`identified_frames.json` 中 `time_ms` 是实际解码帧时间，`subframe_ms` 是原选帧模块的亚帧估计，两者在变帧率视频上可能不同。

每个关键帧的 `source` 说明实际依据，例如 `wrist_return` 表示 P7 使用双手回落信号，未确认触球。原算法未输出 P1/P4/P7 的数值置信度，因此保留 null 并附解释；其余选帧分数也不是经过标定的正确概率。

渲染时逐帧叠加 SwingTrace 的全部原始检测框：青色 `shaft`、紫色 `head`、绿色 `hands/grip`，标签附检测置信度。显示于 `rtmw_2d.mp4`、三维对照视频的原画面侧、P1–P10 图片及十帧总览。漏检类别不画框，不沿用上一帧结果；原始框可能包含未被球手关联选中的候选，选帧采用的关联结果仍以 JSON 为准。渲染复用保存/缓存的检测结果，不重新运行模型。

原流程若明确报告 `no_corroborated_wrist_rise_and_return`，工作流将该视频记为失败，保留原流程诊断，不导出可信十帧清单。`validation.json` 的 `validation_status` 区分文件/阶段审计是否通过；`batch_summary.json` 中程序执行成功不代表人工确认动作正确。

批处理中单段失败会保留诊断并继续处理其余视频。退出码 0 表示全部成功，2 表示部分失败，1 表示初始化失败。不完整或骨架缺失的视频不会绕过原质量检查；缺少合法十帧时不生成伪造的完整结果。

## 测试

```powershell
.\rtmw_workflow\.venv\Scripts\python.exe -m unittest discover -s rtmw_workflow/tests -v
```

`self-test` 覆盖三类别 ID、多人关联、漏检、不可信框拒绝、手动球位置、原有选帧/球杆/骨架回归；同时严格加载 RTMW 2D 权重，在同一张合成图的两个平移框上验证坐标还原，再做 SwingTrace 单张合成图推理。当前报告为 `outputs/self_test_rtmw2d.json`。`check` 会读取 005 的一帧，只有明确需要该检查时才运行。

## GPU 分阶段 FP32

```powershell
.\rtmw_workflow\.venv\Scripts\python.exe -m rtmw_workflow video pro_refs/005.mp4 --pose-device cuda --club-device cuda --cpu-threads 6
```

先在 GPU 上推理所有人体帧并缓存 CPU 结果，再将人体模型移回 CPU，使用 GPU 运行球杆与原有选帧流程，避免同时在显存驻留两个大模型。两模型均保留 FP32。`stage_timing.json` 保存阶段耗时；`run_status.json` 的 `seconds` 包含人体阶段、原流程及输出，不包含共享模型初始化。

`batch --resume --output-dir <原批次目录>` 跳过已完成条目，并保留中断尝试。仅可续跑相同 RTMW 2D 权重和 SwingTrace 的批次；旧 RTMW3D 或 ViTPose 批次须使用新目录。旧后台批次已停止。

`--club-checkpoint models/swingtrace-yolo11m.pt` 可指定同类权重；`--ball-position X Y` 指定原图中球的放置位置，省去准备阶段球位置搜索，但触球/球离开仍需独立图像证据。默认用现有 YOLOE 自动寻找球。

参考：[RTMW 论文](https://arxiv.org/html/2407.08634v1)、[官方 RTMPose3D](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d)。
