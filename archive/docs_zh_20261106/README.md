# 高尔夫姿态工作流

请在项目文件夹中，使用其虚拟环境运行以下 PowerShell 命令：

```powershell
cd J:\Golf_Posture
```

## 1. YOLO26n 工作流

按顺序运行各阶段：参考（references）、视频（video），然后评分（score）。

```powershell
.\.venv\Scripts\python.exe workflow.py references
.\.venv\Scripts\python.exe workflow.py video pro_refs/w07.mp4
.\.venv\Scripts\python.exe workflow.py score
```

参考命令默认读取 `pro_refs/Standard_10_frames/` 中的十张参考图。
将 `pro_refs/w07.mp4` 替换为你的视频路径；包含空格的路径需加引号。

模型：双类别 `golfpose_detector_2cls_yolox_s.pth` 用于球手/球杆检测框，
`yolo26n-pose.pt` 用于身体姿态，`golfpose_club_ViTPose_huge.pth` 用于球杆
标记点，MixSTE2 **17+2**（`golfpose_17+2_33.6_30.5_59.5.bin`）用于 3D
转换。球杆模型只有 `shaft` 和 `hosel` 参与 3D 提升；其杆跟/杆趾
点会被丢弃，与联合工作流一致。`--lifter 17+5` 则保留全部五个点。
检测器和 GolfPose 权重检查点位于 `golfpose_checkpoints/` 下。

P1-P10 提取和挥杆窗口遵循 HackMotion 的 P 位置分类；
参见 [WORKFLOW.md](WORKFLOW.md) 中的“P-position extraction and swing window update (2026-11-04)”
（P 位置提取与挥杆窗口更新）一节。

输出以及最近一次完成运行的指针：`runs/workflows/`。

## 2. GolfPose ViTPose 联合工作流

这一独立工作流使用单类别 `golfpose_detector_1cls_yolox_s.pth`
检测器和单个 22 关节 `golfpose_golfer_ViTPose_huge.pth` 姿态模型，
随后接 MixSTE2 **17+2**（`golfpose_17+2_33.6_30.5_59.5.bin`）。只有 17 个
身体关节以及 `shaft` 和 `hosel` 参与 3D 提升；模型的 `heel`、`toe_down`
和 `toe_up` 通道被丢弃，因为它们在手机视频上不可靠
（参见 [ViTPose 工作流 README](golfpose_workflow/README.md)）。这些
权重检查点位于 `golfpose_checkpoints/` 下。

```powershell
.\.venv\Scripts\python.exe -m golfpose_workflow --reference
.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/w07.mp4
.\.venv\Scripts\python.exe -m golfpose_workflow --score
```

输出以及最近一次完成运行的指针：`runs/golfpose_workflow/`。
参考/视频命令每次都会重新运行 2D 推理。关于检测器跟踪、
参考回退策略和提取条件，请参见
[ViTPose 工作流 README](golfpose_workflow/README.md)。

**当前样例的局限：** 在 `w07.mp4` 上基于裁剪框的 ViTPose 推理无法
满足现有的 P5/P6 提取条件。该视频命令会报告
选帧不完整，且不会发布已完成的十帧运行结果。随后的
`--score` 会使用之前已完成的运行（如果存在）；它不会对
失败的尝试评分。只有在审核过预测结果之后，
才可通过 `--positions-json` 提供审核过的帧索引。

### 当检测器漏检球手时初始化裁剪框

两个视频工作流都会在 2D 推理之前推导出一个默认的球手 ROI。YOLO26n
在 12 个采样帧上运行一遍，在每帧中取置信度足够的最大人物，
并保留与其中位数一致的检测框。这些框的并集会为容纳
球杆而扩展：左右各扩展框高的 0.15，上方 0.25，下方 0.05。每个片段耗时
1-5 s，且仅在 GolfPose 检测器漏检时使用：在 golfpose_workflow 中是单类别 YOLOX，
在 `workflow.py` 中是双类别 YOLOX 的人物检测框（替代
整帧）。检测结果始终优先。ROI 及其依据记录在
`<stem>_3d.json` 的 `pose_backend.initial_roi_source` / `initial_roi_details` 下，
并在每帧中记录为 `auto-golfer-roi`。使用 `--no-auto-golfer-box` 可关闭此功能。

如果仍然没有初始 ROI，ViTPose 就没有可处理的裁剪区域。
工作流会将这种情况与低置信度的姿态根节点分开报告，并在
失败视频的输出文件夹中保存 `pose_inference_diagnostics.json` 和 `.npz`。

使用 `--golfer-box X1 Y1 X2 Y2`（两个工作流均可；它会覆盖自动 ROI）提供一个包含球手与球杆的初始矩形，坐标采用
解码后视频的原始像素坐标。所提供的 ROI 会一直保留，
直到 YOLOX 检测到球手；所有姿态坐标仍来自同一个
22 关节 ViTPose 模型。对于当前 592x1280 的 `pro_refs/010.mp4`，该检测框已
通过完整的 252 帧运行和自动十帧导出得到验证：

```powershell
.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/010.mp4 --golfer-box 140 220 500 900 --view face_on
```

请根据每个视频的取景选择合适的坐标，需同时包含球手和
球杆。该框必须位于解码后的帧之内。姿态置信度阈值和 P1-P10
提取条件仍然适用。

## 输出与评分

| 阶段 | YOLO26n 结果集 | ViTPose 结果集 |
| --- | --- | --- |
| 参考 | `runs/workflows/references/` | `runs/golfpose_workflow/references/` |
| 视频 | `runs/workflows/videos/<video-stem>/` | `runs/golfpose_workflow/videos/<video-stem>/` |
| 评分 | `runs/workflows/scores/<video-run-name>/` | `runs/golfpose_workflow/scores/<video-run-name>/` |

重复运行会保留已有结果，并选用带时间戳的文件夹。每个
已完成的视频都会导出 `P1.jpg` 到 `P10.jpg`、对应的 `P*_3d.jpg`、
`identified_frames.npz/json`、完整序列的坐标/置信度，以及一个渲染好的
跟踪视频。自动完成要求十个互不相同且递增的源帧
索引，以及有限值的 3D 姿态。两个评分器使用相同的 14 项姿态标准和
已保存的坐标，并且每个工作流各自维护最新参考/视频指针。

如需对明确指定的已完成文件夹（而非最近一次运行）评分：

```powershell
.\.venv\Scripts\python.exe workflow.py score runs/workflows/videos/w07 --references runs/workflows/references
.\.venv\Scripts\python.exe -m golfpose_workflow --score --run runs/golfpose_workflow/videos/<completed-video-run> --references runs/golfpose_workflow/references
```

请将 `<completed-video-run>` 以及任何带时间戳的文件夹名替换为实际
已完成的输出文件夹。使用 `--output-dir <empty-folder>` 可为某个阶段指定明确的
输出目录。

## 常用选项

两个视频工作流都接受 `--handedness right|left`、
`--view face_on|down_the_line`、`--positions-json <reviewed-file.json>`、
`--pose-cpu-threads <count>`，以及可选的 `--track-club-path`。
两条视频路线都接受 `--golfer-box X1 Y1 X2 Y2` 以覆盖自动
球手 ROI，并接受 `--no-auto-golfer-box` 以禁用该功能。
两个参考命令都接受 `--input-dir <reference-sheet-folder>`。

```powershell
.\.venv\Scripts\python.exe workflow.py references --help
.\.venv\Scripts\python.exe workflow.py video --help
.\.venv\Scripts\python.exe workflow.py score --help
.\.venv\Scripts\python.exe -m golfpose_workflow --help
```

关于原始工作流中的数据约定、检测器行为、验证
以及缓存运行选项，请参见 [WORKFLOW.md](WORKFLOW.md)。评分公式请参见
[SCORING_METHODOLOGY.md](SCORING_METHODOLOGY.md)。P1–P10 帧识别逻辑请参见
[FRAME_RECOGNITION.md](FRAME_RECOGNITION.md)。
以下说明描述的是早期实验，而非当前的默认设置。

---

# 尝试 1

使用 Ultralytics YOLO26n-pose（`yolo26n-pose.pt`）对 `images/` 中的图像进行姿态预测，该模型即[姿态估计文档](https://docs.ultralytics.com/tasks/pose)中介绍的 COCO 关键点模型。

## 运行方法

```powershell
pip install ultralytics
python predict_pose.py
```

该脚本加载 `images/` 下的每张图像，绘制预测的骨架，并写出：

- 标注后的图像，位于 `runs/pose/predict/`
- 关节坐标，位于 `runs/pose/predict/keypoints.json`

每条人物记录包含 17 个 COCO 关节（从鼻子到双脚踝）、像素坐标 `x`/`y`、每个关节的置信度，以及人物检测框。

## 结果

| 项目 | 结果 |
| --- | --- |
| 预测图像数 | 178 |
| 推理耗时 | 640×640 下每张图像 22.6 ms |
| 单人 | 172 张图像 |
| 三人 | `pro_part_01.png` 到 `pro_part_04.png` |
| 八人 | `pro_sequence.jpg` 和 `pro_sequence.png` |
| 关键点 | 3,400 |
| 置信度低于 0.5 的关键点 | 716 |
| 有手腕置信度低于 0.5 的人物 | 106 |
| 关键点置信度 | 最小值 0.006，中位数 0.977，均值 0.787 |

单个球手的挥杆帧能被可靠检测。职业球手参考图是拼贴图，因此模型会为图中的每个人形都返回一副骨架。关节置信度的中位数很高，但手腕和其他被遮挡的关节丢失得相当频繁，基于这些点构建的挥杆角度会不稳定。

## 未来可能的改进

- 将 `pro_part_*` 和 `pro_sequence` 拆分为每张图像只含一名球手，或只保留目标人物（最大检测框，或最靠近图像中心的人形）。当前的 JSON 把每个检测到的人形都当作球手。
- 丢弃或插值低于置信度阈值的关节，尤其是手腕。在检测到的人物中有 106 个至少有一只手腕置信度偏低，而手腕正是估计球杆杆身所依赖的关节。
- 在每段挥杆序列中跟踪同一名球手，并在时间上平滑关节轨迹。目前每帧都单独预测，因此当远侧手臂被身体挡住时，关节位置可能跳变。
- 利用已保存的关键点计算高尔夫指标：脊柱倾角、肩部倾角、髋部连线、膝关节弯曲以及前导侧手腕位置。目前 JSON 只包含原始坐标。
- 一旦已知相机视角（正面视角（face-on）或后方视角（down-the-line））和球手的左右手习惯，就将 COCO 的左/右映射为前导侧和后侧。
- 当关节定位精度比当前的 22.6 ms 更重要时，从 YOLO26n-pose 换用更大的姿态权重检查点。nano 模型是该系列中精度最低的姿态变体。
- 将 `imgsz` 提高到 640 以上，或在推理前先裁剪到球手，使手腕和脚踝保留更多像素。
- 添加球杆、握把和球的点位。17 个 COCO 关节不包含球杆，因此相对于杆身的姿态目前尚未测量。
- 在比较各次尝试之前，对 `images/` 中重复的挥杆副本去重，以免同一帧被重复计数。

# 挥杆平面轨迹

`trace_swing_plane.py` 在单个视频上绘制平滑后的杆头路径，从第一个被接受的杆身一直到最后一个。球杆掩膜来自训练好的 YOLO26n-seg 权重。YOLO26n-pose 提供双手位置，以便将该掩膜的握把端与杆头区分开。

该脚本需要由 `prepare_golf_club_seg.py` 和 `train_golf_club_seg.py`（已归档至 `archive/scripts/`；如需重新训练，请将其移回根目录）生成的 `runs/segment/golf_club_train/weights/best.pt`，以及姿态权重检查点 `yolo26n-pose.pt`。

## 运行方法

```powershell
pip install ultralytics
python -m golf_workflows.trace_swing_plane
python -m golf_workflows.trace_swing_plane user_swings\1785648828688618000_swing.mp4
python -m golf_workflows.trace_swing_plane user_swings\1785648828688618000_swing.mp4 --mediapipe-wrists
```

如果不提供路径，脚本会从 `user_swings/` 下随机选取一个视频。可接受的扩展名为 `.mov`、`.mp4`、`.avi`、`.mkv` 和 `.webm`。

## 工作流

每帧在一次遍历中处理。屏幕上的路径只包含截至该帧已被接受的部分。

1. **姿态。** YOLO26n-pose 以 `imgsz` 640 读取视频。如果找到多个人，最大检测框对应的即为球手。
2. **双手。** 锚点是置信度至少为 0.25 的手腕的平均位置。如果两只手腕都偏弱，则使用双肩。如果双肩也缺失，则使用球手检测框的中心。
3. **球杆。** 训练好的分割模型在同一帧上运行（`conf` 0.25，retina 掩膜，最多 3 个检测结果）。只保留类别名为高尔夫球杆的检测框。
4. **杆身。** 球杆轮廓的主轴即为杆身。短于 12 像素的轮廓会被忽略。主/次轴方差比低于 8 的轮廓只被视为杆头点而不含握把，从而避免圆形的误检掩膜凭空生成杆身。
5. **两端。** 杆身离双手较近的一端是握把，另一端是杆头。如果既没有双手位置，也没有之前的杆头，则较低的一端为杆头。
6. **选择。** 当存在多个球杆掩膜时，胜出者是更长、更细长且握把更靠近双手的检测结果。
7. **稳定性。** 在 6 帧内跳动超过帧对角线 45% 的杆头会被丢弃。在下一帧中，新的杆身会向前一个杆身混合（新值占 72%），使平面不会抖动。超过 6 帧的间隔会开启新的片段，且不与之前的路径相连。
8. **阶段。** 读取完片段后，`swing_phases.py` 根据杆头路径标注准备站位、上杆顶点、击球和送杆（仅在球杆缺失时使用手腕高度）。路径颜色按这些阶段变化，而不是从开始到结束的单一渐变。
9. **绘制。** 视频在第二遍中写出，使每一帧都能使用阶段标签。视频的帧率来自姿态读取器，回退值为 30。

## 绘制内容

颜色从第一个样本的蓝色过渡到最新样本的红色。

- 已链接样本的平滑杆头路径。每一步绘制为八小段曲线；无法拟合曲线时则绘制为直线段。一个圆环标记第一个杆头。
- 当本帧产生了样本时，在当前杆头上绘制一个实心圆点。
- 一个标有开始和结束的小图例。

## 输出

文件写入 `runs/swing_plane/` 下，使用源视频的文件名主干（stem）：

- `*_swing_plane.avi` — 从片段开始到结束逐步构建的杆头路径
- `*_swing_plane.jpg` — 最后一帧，包含完整路径
- `*_swing_plane.json` — 各阶段的帧索引，以及每个被接受样本的一条记录：帧索引、置信度、是否延续上一个样本（`linked`）、握把 `[x, y]` 或 `null`，以及杆头 `[x, y]`
- `*_phases.png` — 平滑后的杆头高度和速度，并标出阶段分界线

## 示例

| 项目 | 结果 |
| --- | --- |
| 源视频 | `user_swings/1785648828688618000_swing.mp4` |
| 写出帧数 | 66 |
| 挥杆平面样本数 | 53 |
| 视频 | `runs/swing_plane/1785648828688618000_swing_swing_plane.avi` |
| 静态图 | `runs/swing_plane/1785648828688618000_swing_swing_plane.jpg` |

# GolfPose 3D

`golf_pose_3d.py` 使用 GolfPose-3D（即 [GolfPose](https://github.com/MingHanLee/GolfPose)（ICPR 2024）中的 MixSTE2 模型）将每个挥杆视频提升为 3D 球手和球杆。2D 输入来自上文使用的同一批 Ultralytics 模型：球手使用 YOLO26n-pose，球杆使用训练好的 YOLO26n-seg 权重。未使用 GolfPose 的 MMPose 和 MMDet 2D 模型。它们的环境仅支持 Linux 且为 CUDA 构建，而本项目在 Windows 上以 CPU 运行。

## 环境配置

```powershell
pip install ultralytics einops timm==0.9.16 local-attention==1.9.0 mediapipe scipy
```

- 模型代码原样内置于 `third_party/golfpose/common/`，GolfPose 的 `LICENSE` 和源提交记录位于 `third_party/golfpose/SOURCE.txt`。该文件导入的是较旧的 `timm.models.helpers` API，因此 timm 固定为 0.9.16 版本。
- 将权重检查点放在 `golfpose_checkpoints/golfpose_17+2_33.6_30.5_59.5.bin`。它对应 GolfPose README 中 GolfPose-3D (GC) N=19 那一行，大小约 406 MB。

## 运行方法

```powershell
python -m golf_workflows.golf_pose_3d
python -m golf_workflows.golf_pose_3d user_swings\1785648828688618000_swing.mp4
python -m golf_workflows.golf_pose_3d --all
python -m golf_workflows.golf_pose_3d user_swings\1785648828688618000_swing.mp4 --mediapipe-wrists
```

如果不提供路径，则使用 `user_swings/` 下的一个随机视频。`--all` 会处理该目录下的所有视频，且只加载一次模型。

## 工作流

1. **2D 检测。** 每帧都经过 YOLO26n-pose 和球杆分割模型。球手、双手、球杆选择和跳变过滤与 `trace_swing_plane.py` 中的完全相同，因此 2D 杆头与挥杆路径脚本一致。
2. **关键点转换。** GolfPose 期望输入 Human3.6M 顺序的 17 个身体关节，其后是球杆点位。COCO 关节的转换方式如下：
   - 髋、膝、踝（作为脚）、肩、肘和手腕直接映射。
   - `root` 是双髋的中点，`thorax` 是双肩的中点，`spine` 是 `root` 与 `thorax` 的中点。
   - `neck_base` 位于从 `thorax` 到鼻子的中点。`head` 位于耳朵和眼睛之外 25% 处，沿远离 `thorax` 的方向测量。
   - 球杆的握把端成为 `shaft`，杆头端成为 `hosel`，由此得到 17+2 布局。
   - 置信度低于 0.25 的关节视为缺失。
3. **补缺与平滑。** 缺失的关节和球杆点位通过时间上的线性插值补齐，片段两端则保持第一个或最后一个有效值。随后每个补齐后的关节都会用一个短时移动平均（约 0.15 s，与 GolfPosePro 的手腕滤波思路相同）进行平滑，再送入 3D 模型。
4. **阶段。** 准备站位、上杆顶点、击球和送杆来自 `swing_phases.py`。3D 提升只使用该窗口，因此准备站位之前和收杆之后的空闲帧不会被填充进模型。
5. **归一化。** GolfPose 是在 1280x720 相机上训练的。挥杆以球手为中心，并缩放到使球手检测框的中位数占虚拟 1280x720 画布的 70%。随后按 GolfPose 的方式归一化：x 和 y 除以宽度，乘以 2，再平移 `[1, h / w]`。
6. **3D 提升。** MixSTE2 在 CPU 上以 243 帧窗口运行，每个窗口约 3 秒：
   - 短于 243 帧的片段通过重复最后一帧进行填充。
   - 更长的片段使用更多窗口，最后一个窗口与片段末尾对齐。
   - 每个窗口都与其镜像结果（x 取反，左右互换）取平均，与 GolfPose 评估中的做法相同。
   - 输出是相对根节点的坐标，单位为米，采用相机坐标轴。
7. **渲染。** 左侧面板是视频帧，叠加转换后的 2D 骨架和平滑后的 2D 杆头路径，按挥杆阶段着色。右侧面板是 3D 视图，包含骨架、以 shaft 到 hosel 连线表示的球杆，以及 3D hosel 路径。阶段窗口之外的帧在 3D 一侧留空。

在任一脚本上使用 `--mediapipe-wrists`，当 MediaPipe Pose 中对应标记点可见时，会用它补全置信度低于 0.25 的 YOLO 手腕。它不会替换姿态模型或球杆分割。

## 输出

文件写入 `runs/golf_pose_3d/` 下，使用源视频的文件名主干（stem）：

- `*_3d.avi` — 左右并排的视频
- `*_3d.jpg` — 该视频的最后一帧
- `*_3d.npz` — `joints`（19 个名称）、`keypoints_2d` 和 `keypoints_2d_valid`（像素坐标及每个关节的检测标志）、`keypoints_2d_filled`、`keypoints_2d_normalized`（模型输入）、`keypoints_3d`（帧数 × 19 × 3，单位为米）、`fps`、`frame_size`
- `*_3d.json` — 权重检查点、帧数、归一化参数、检测到球杆的帧、每个关节的插值帧，以及每根 3D 骨骼长度的均值和标准差

## 示例

| 项目 | `1785648828688618000_swing.mp4` | `1785642572528296000_swing.mov` |
| --- | --- | --- |
| 帧数 | 66 | 331 |
| 准备站位 / 上杆顶点 / 击球 / 收杆 | 11 / 27 / 44 / 54 | 144 / 160 / 177 / 187 |
| 3D 窗口 | 第 11–54 帧，一个填充窗口 | 第 144–187 帧，一个填充窗口 |
| 检测到球杆 | 53 帧 | 270 帧 |
| 该窗口内前臂长度变化 | 约 2 cm | 约 2 cm |
| shaft 到 hosel 的长度 | 0.54 ± 0.19 m | 0.54 ± 0.19 m |

长片段的速度峰值位于第 177 帧，与被选为击球的帧相同。该窗口之外的帧保留在 2D 路径中，在 3D 视图中留空。

## 局限性

- GolfPose-3D 是在其自有动作捕捉数据集的真值 2D 上训练的。而这里的 `neck_base`、`head` 和 `spine` 是从 COCO 关节估计得到的，且 YOLO 会引入噪声，因此误差预计会大于论文中的 33.6 mm。当鼻子被遮挡或球手低头看球时，`neck_base` 是最不稳定的关节。
- 除非提供实测的标记点间距，3D 杆身长度仍然是 MixSTE 学到的先验。五点 2D 监督能改善投影几何，但并不能把未经标定的重建变成对球杆物理长度的测量。
- 球杆 ViTPose 模型可以提供杆跟（heel）和杆趾（toe）点位，但遮挡和过宽的回退裁剪框可能产生低置信度输出。原始坐标和置信度都会保留；不可靠的点被标记为无效，且只在补齐后的 3D 输入中被插补。
- 输出是相对根节点的。整体位置以及相对于相机的绝对尺度均未测量。

# 挥杆阶段

`swing_phases.py` 是改编自 [GolfPosePro](https://github.com/ryanboscobanze/GolfPosePro) 的时间层。该 notebook 跟踪 MediaPipe 手腕，并根据其高度和速度切分挥杆。这里则用分割得到的杆头进行同样的切分，因为手腕只是球杆的替代物。

- 用 `scipy.ndimage.uniform_filter1d`（约 0.15 s）平滑路径。
- 速度为该路径的 `np.gradient`。
- **准备站位**是速度持续高于峰值 12% 之前的那一帧。
- **上杆顶点**是在击球之前、杆头离准备站位位置最远的帧。
- **击球**是已回到准备站位位置附近的速度最快的帧。
- **送杆**是速度降到击球速度 35% 以下的位置。

只有当杆头样本少于八个时才使用手腕高度。`--mediapipe-wrists` 是可选项，仅用于修补较弱的 YOLO 手腕。


## 实验性的 17+5 3D 提升模型（lifter）

`golfpose_17+5_39.2_32.3_62.8.bin` 是一个 22 关节的 MixSTE2 权重检查点：17 个身体关节加上 `shaft`、`hosel`、`heel`、`toe_down` 和 `toe_up`。其学习到的空间嵌入为 `[1, 22, 512]`，时间嵌入为 `[1, 243, 512]`。它无法直接接受 15+5 / 20 点张量，也无法直接接受现有的 19 点张量。

两种尺寸的提升模型都受支持。原始的 17+2 权重检查点仍为默认，其文件得以保留。17+5 模式有单独的默认输出目录。

```powershell
python -m golf_workflows.standard_frames --lifter 17+5 --input-dir pro_refs/Standard_10_frames --output-dir runs/standard_frames_17plus5_20261027
python -m golf_workflows.golf_pose_3d user_swings/<swing>.mp4 --lifter 17+5 --output-dir runs/golf_pose_3d_17plus5

# Original model and reference generation:
python -m golf_workflows.standard_frames --lifter 17+2
```

默认的 `workflow.py` 路径现在使用 `golfpose_club_ViTPose_huge.pth` 来
观测全部五个球杆标记点。分割模型仍负责选出高尔夫
球杆并提供裁剪框；ViTPose-Huge 热图则严格按顺序提供 `shaft`、`hosel`、
`heel`、`toe_down` 和 `toe_up`。低于
置信度阈值的标记点保持为 NaN/False，且只在补齐后的模型
输入中被插值。使用 `--club-pose-backend segmentation --lifter 17+2` 可恢复
早先的双代理点路径。

在 P1 正面视角参考上试用本地保存的 22 关键点 DEKR 模型，得到的预测置信度极低且落在球手之外。因此未将其作为实测标记点的来源。DEKR 权重检查点文件仍然保留。

新的参考运行会写出全部 22 个预测关节、分开的相机渲染和融合参考渲染、原始 2D 点、每个关节的有效性、补齐后的 2D 输入以及插补元数据。参考融合使用与之前相同的正面视角/后方视角方法，对齐方式基于现有的身体/杆身骨骼选择。`runs/standard_frames/` 得以保留，并仍是评分的默认参考。

参考球杆可视化不会对球杆重新缩放，也不会将其连接到手腕。
如果某个视角具有全部五个可靠的 GolfPose 标记点，其对齐后的 MixSTE 球杆可以
单独保存为 `reference_3d_club_overlay.jpg` 以及 `joints.npz` 中的
`reference_3d_club_overlay`。用于评分的数值型 `reference_3d`
永远不会被该叠加结果替换。
普通的 `reference_3d.jpg` 只渲染身体，以避免把
插补的或超出画面的球杆通道当作观测到的几何呈现。

使用 `--club-crop-backend official-yolox-2cls` 可采用论文中的独立球杆
检测器路线。它运行官方发布的双类别
YOLOX-s 权重检查点（`person`、`club`），并将选中的球杆检测框传给
`golfpose_club_ViTPose_huge.pth`。现有的分割裁剪框仍为
默认，也可通过 `--club-crop-backend segmentation` 显式选择。

官方关节定义：https://github.com/MingHanLee/GolfPose/blob/main/configs/mmpose/_base_/datasets/golfswing_golfer.py


## ViTPose-Huge 2D 后端配合 17+5 提升模型

已保存的 COCO ViTPose-Huge 权重检查点可通过 `standard_frames.py` 和 `golf_pose_3d.py` 中的 `--pose-backend vitpose-huge` 使用。现有的 `yolo26n-pose` 后端仍为默认。

```powershell
python -m golf_workflows.standard_frames --pose-backend vitpose-huge --club-pose-backend vitpose-huge --lifter 17+5 --input-dir pro_refs/Standard_10_frames --output-dir runs/standard_frames_vitpose_huge_17plus5
python -m golf_workflows.golf_pose_3d user_swings/<swing>.mp4 --pose-backend vitpose-huge --club-pose-backend vitpose-huge --lifter 17+5 --output-dir runs/golf_pose_3d_vitpose_huge_17plus5
```

保留的 YOLO26n-pose 模型负责定位球手的检测框。ViTPose-Huge 使用 256x192 的带填充 UDP 裁剪、RGB 归一化、原图/镜像热图平均以及 MMPose 的 UDP/DARK 解码器，替换其身体关键点。视频帧使用检测框 IoU 关联来跟踪同一名球手，没有匹配时则选择最大检测框。单张参考图片则各自独立处理。

`vitpose_tracking.py` 使用原生 PyTorch CPU 算子实现了 32 层、1280 维的 Transformer 及热图头。所有官方发布的状态张量均以严格模式加载；仅用于训练的 dropout 被禁用。COCO 身体导出模型保留了一个未使用的类别 token 和 193 个位置槽，而 GolfPose 球杆导出模型则能正确加载其较新的无类别 token、192 位置布局。这样就无需在当前 Windows 环境中依赖 MMPreTrain 或 MMCV 的编译算子。ViTPose 后端默认使用四个 PyTorch CPU 线程；视频 CLI 可通过 `--pose-cpu-threads` 覆盖该值。解码器的来源说明和许可证位于 `third_party/golfpose/VITPOSE_SOURCE.txt` 和 `MMPOSE_LICENSE`。

17 个 COCO 身体标记点会被转换为现有的 GolfPose 身体顺序。
独立的球杆 ViTPose 权重检查点使用相同的 192x256 UDP/DARK 推理
实现，并配有五通道头。它接收选中的分割
检测框，并直接填充关节 17-21。`joints.npz` 和视频的 `*_3d.npz` 会保留
原始坐标、有效性掩膜、补齐后的输入以及五个置信度值，从而使
实测的与插值得到的标记点仍可区分。

每个参考视角的 `joints.json` 记录 ViTPose 的原始 COCO 坐标、置信度响应、裁剪仿射矩阵、推理时间以及转换后的 GolfPose 关节。`joints.npz` 存储 22 关节的 2D/3D 数组以及有效性/补齐信息。新的运行还提供对比总览和验证记录。现有的参考文件夹和权重检查点均被保留；不会自动选择新的参考集用于评分。


对于经过审核的 `b85ed0b4db566e63bce190824c2f2196.mp4` 视频试验：

```powershell
python -m golf_workflows.golf_pose_3d pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4 --pose-backend vitpose-huge --lifter 17+5 --pose-cpu-threads 8 --positions-json runs/golf_pose_3d/b85ed0b4db566e63bce190824c2f2196/selected_frames.json --output-dir runs/golf_pose_3d_vitpose_huge_17plus5
```

`--positions-json` 会复用经过人工目视审核、从零开始计数的 P1-P10 索引，同时重新计算 2D 和 3D 姿态。它会校验源文件名、覆盖范围、顺序和帧边界。审核过的击球帧会被保留，而不会被基于球的距离判断所替换。视频时间戳描述的是经过剪辑的慢动作源视频。省略该选项则保持自动位置选择。

`--pose-cpu-threads` 默认为四。在当前 CPU 上的计时检查为该视频选择了八个线程；与四线程相比，热图差异低于 `4e-7`。源权重检查点和先前的输出均被保留。该视频试验单独保存在 `runs/golf_pose_3d_vitpose_huge_17plus5/` 下。
