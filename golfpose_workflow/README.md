# 联合 GolfPose 工作流

在 `J:\Golf_Posture` 目录下使用项目虚拟环境运行：

```powershell
.\.venv\Scripts\python.exe -m golfpose_workflow --reference
.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/w07.mp4
.\.venv\Scripts\python.exe -m golfpose_workflow --score
```

本工作流使用单个姿态权重检查点运行全新的 2D 姿态推理：
`golfpose_checkpoints/golfpose_golfer_ViTPose_huge.pth`。其原生的 22 个通道
包含 17 个身体关节，其后依次为 shaft、hosel、heel、toe_down 和 toe_up。
在姿态推理之前，由 `golfpose_checkpoints/golfpose_detector_1cls_yolox_s.pth`
定位唯一的 `golfer` 类别。选中的检测框沿用现有的 1.25 倍扩展，生成 192x256 的 UDP 裁剪框
（crop）。UDP/DARK 解码将坐标直接映射回
原始图像；翻转平均、置信度 >= 0.10 以及画面内有效性判定均保持不变。

3D 提升（lifting）使用 MixSTE2 **17+2**（`golfpose_17+2_33.6_30.5_59.5.bin`）：即 17 个身体
关节加上 `shaft`（握把侧标记）和 `hosel`。heel、toe_down 和 toe_up
通道及其置信度在 3D 提升之前被丢弃；保存的数组为
19 关节，`<stem>_3d.json` 在 `dropped_2d_landmarks` 下列出这些通道。原因如下：

- 在 `010.mp4` 上，球手裁剪框约将 13 个源像素映射到一个热图单元，而
  一号木杆头的宽度只有约 25 px。三个杆头点的置信度中位数最低
  （0.58-0.63），它们与杆颈距离的第 95 百分位数约为
  250 px；在下杆和收杆阶段，它们经常散布在整个画面中。
- 阶段检测、球杆精修和评分只使用杆身（shaft）和杆颈（hosel）。
- GolfPose 17+2 3D 提升模型（lifter）报告的误差低于 17+5
  （MPJPE 33.6 vs 39.2 mm；身体 30.5 vs 32.3；球杆 59.5 vs 62.8）。
- GolfPose 的另一条球杆路线（双类别 YOLOX 球杆框 + 仅针对球杆的
  ViTPose）在 `010.mp4` 的十帧上做了测试，效果更差：仅在 3/10 帧中
  检测到球杆类别，得分为 0.01-0.08，其中一次还落在错误的物体上；
  而回退到球手框时得到的杆身/杆颈点对会塌缩到一起。

杆身和杆颈仍来自 22 关节模型，在下杆阶段模糊的帧中可能出错；
球杆精修会对低置信度的点对进行插值。

对于视频，通过检测框的重叠来保持球手身份；检测器出现空缺时，
姿态推理使用最近一次检测到的 ROI，并记录为 `tracked-detector-roi`。
若检测器一开始就漏检，则不会产生观测姿态。视频绝不会回退（fallback）到
整帧的姿态裁剪。静态参考面板在检测失败时可以回退到整个面板，
记录为 `reference-panel-fallback`。检测器置信度与
姿态置信度分别保存；被跟踪的 ROI 不会被标记为新的检测。
本工作流中没有单独的球杆姿态模型。

默认情况下，会先推导出一个自动球手 ROI：用 YOLO26n 对 12 个
采样帧进行检测，取相互一致的球手框的并集，并为球杆留出扩展余量。
它仅在 YOLOX 漏检时使用，记录为 `auto-golfer-roi`，可通过
`--no-auto-golfer-box` 禁用。`pro_refs/w09.mp4` 上 YOLOX 从未触发，现在无需
手动框即可运行。如需覆盖，请显式初始化 ROI：

```powershell
.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/010.mp4 --golfer-box 140 220 500 900 --view face_on
```

该矩形为 `(x1, y1, x2, y2)`，单位是原始解码帧的像素。这些数值
已通过 592x1280 分辨率、共 252 帧的 `010.mp4` 完整运行验证；对于其他
视频，请选择合适的框。同一个 22 关节模型处理该裁剪区域，同时 YOLOX 持续进行
检测。裁剪元数据会将 `manual-golfer-roi` 与检测器
观测或保留的已检测 ROI 区分开。此选项仅用于初始化 ROI，
不会启用整帧的视频姿态回退。

当根节点失败时，会在视频输出文件夹中保存 `pose_inference_diagnostics.json/.npz`。
它们区分两种情况：因缺少检测器
框而完全没有进行姿态推理，以及进行了姿态推理但根节点未通过置信度/图像边界检验。
以 ROI 初始化的 `010.mp4` 完整运行在全部 252 帧中都得到了观测到的根节点，
并导出了十个有限值的 22 关节姿态。输出位于
`runs/golfpose_workflow/videos/010_roi_20261104/`；验证详情位于
`runs/diagnose_010_20261104/validation.json`。全部 72 项测试通过。

检测器修订版通过了全部 67 项测试。在 `w07.mp4` 上，单类别检测仅在
7/53 帧中可用；其余 46 帧使用了保留的检测器 ROI。裁剪后的
推理已完成，但其 3D 提升后的杆身未通过现有的 P5 阈值
（最小垂直分量约为 0.357，而要求为 <= 0.35）。
因此 P6 仍不可用，完成性验证拒绝了该
自动十帧导出。规则保持不变；对于该样本，请先审核预测结果，再
提供 `--positions-json`。诊断信息位于
`runs/validation_detector_routing_20261103/pipeline_validation.json`。
MixSTE2 17+2 将这些预测提升到 3D。安装可选的球检测模型后，
仍可通过共享流水线进行视频球位时间修正。
其 CLIP 分词器依赖已安装在项目环境中，并在
`requirements.txt` 中固定为 Ultralytics CLIP 提交
`7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40`。

代码和说明位于 `golfpose_workflow/`。结果单独存放：

```text
runs/golfpose_workflow/
  latest_references.json
  latest_video.json
  latest_score.json
  references/                  ten reference folders and three combined previews
  videos/<video-stem>/          full tracking results and ten extracted frames
  scores/<video-run-name>/      score.json and comparison pictures
```

重复运行会生成新的带时间戳的文件夹。已完成的结果会被保留。
原有的 `workflow.py` 保持其 YOLO26n 默认设置以及它自己在 `runs/workflows/` 中的
latest 指针。此处的 `--score` 默认只使用本工作流的指针；
`--run` 和 `--references` 可以显式选择其他已完成的文件夹。
`--output-dir` 可以为任意阶段显式指定一个空的输出文件夹。

参考（reference）沿用现有的基于整副骨架置信度、在正面视角（face-on）
与后方视角（down-the-line）之间进行的选择，外加独立的整杆来源（donor）选择，
以及在所有符合条件的球杆观测都不完整时的仅杆身回退。背后视角（rear）仍仅用于诊断。

视频的帧识别与 workflow.py 完全共用（golf_workflows/golf_pose_3d.process），规则见 [FRAME_RECOGNITION.md](../FRAME_RECOGNITION.md)：
P1、P4 由球杆和手部运动确定；P7 以击球瞬间为先（球离开原位、杆头到达球），动作捕捉其次；
其余位置在 3D 上用动态规划联合选取，并给出每个位置的置信度。

球杆精修和窗口融合默认启用。提取时会在相邻位置之间预留
间隔，判定完成需要十个互不相同且递增的
索引、有限值的 19 关节姿态及其对应的源图像。
不会用十个均匀间隔的帧来替代。如果自动获得的证据
不足，命令会报告失败，而不会发布已完成的运行；
必要时请审核索引，并使用 `--positions-json <file>` 重试。

每个视频会导出 `P1.jpg` 至 `P10.jpg`、对应的十张 `P*_3d.jpg`、
`identified_frames.npz/json`、全序列 2D/3D 数组及置信度，以及
渲染后的跟踪视频。源帧索引从零开始。推断得到的球杆标记
与观测点保持区分。

评分直接调用现有的比较器，采用相同的 14 项姿态判据、
左右手习惯、归一化、偏航对齐和角度评分规则。它使用
已保存的 P1-P10 坐标，不会重新选帧。评分公式参见
[SCORING_METHODOLOGY.md](../SCORING_METHODOLOGY.md)。
