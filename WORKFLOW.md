# Three golf posture workflows

Run each stage with one command from `J:\Golf_Posture`:

```powershell
python workflow.py references
python workflow.py video pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4
python workflow.py score
```

Scoring uses the latest successfully completed reference and video workflows.
Each command accepts `--output-dir`. Run `python workflow.py <stage> --help`
for options. `python -m golf_workflows` is an alternative launcher.

## 推理思路与模型调用说明

项目采用“先识别二维关键点，再提升为三维姿态，最后按姿态指标评分”的流程。
人体、球杆和三维重建由不同模型负责，阶段识别与评分则由几何规则和数值计算完成。
以下说明对应 `python workflow.py` 的默认配置：`vitpose-huge`、`17+5`，
视频开启球杆 refinement，CPU 线程数为 8；底层模块的独立命令可能有不同默认值。

### 默认调用哪些模型

| 模型 | 权重文件 | 输入与职责 | 调用阶段 |
| --- | --- | --- | --- |
| YOLO26n-pose | `yolo26n-pose.pt` | 检测人物框，为 ViTPose 提供人体裁剪区域；默认模式下最终人体关键点使用 ViTPose 的结果 | references、video |
| ViTPose-Huge | `golfpose_checkpoints/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth` | 根据人物框裁剪并估计 17 个 COCO 二维人体关键点；这是 COCO 人体模型，不检测球杆点 | references、video |
| 自训练 YOLO26n-seg | `runs/segment/golf_club_train/weights/best.pt` | 分割球杆，结合轮廓主轴和手腕位置估计握把端与杆头端 | references、video |
| GolfPose MixSTE2 17+5 | `golfpose_checkpoints/golfpose_17+5_39.2_32.3_62.8.bin` | 将二维人体与球杆关键点序列提升为三维坐标 | references、video |
| YOLOE 分割模型，可选 | `yoloe-26n-seg.pt` | 在文件存在时以 `golf ball` 类别检测球，辅助修正击球时刻 | 仅 video |

使用 `--pose-backend yolo26n-pose` 时，直接使用 YOLO 的人体关键点，不调用 ViTPose。
使用 `--lifter 17+2` 时，改用
`golfpose_checkpoints/golfpose_17+2_33.6_30.5_59.5.bin`，输入为 17 个人体点和 2 个球杆点。
`--from-run` 复用已完成结果，不重新调用上述神经网络；`score` 也不调用这些模型。

### references：从参考图片到十个三维标准姿态

1. **拆分视角。** 每张 P1-P10 标准图包含正面、沿目标线和背面三个视角，
   先裁剪成单人图片。十张标准图正常对应 30 次单视角姿态处理。
2. **人体二维姿态。** YOLO 找到人物框，ViTPose 将其裁剪为 192×256 输入，
   通过关键点热图解码得到 17 个 COCO 关键点。推理采用原图与水平翻转结果平均。
   然后将 COCO 点转换为 GolfPose 的人体关节顺序；髋中心、脊柱等部分点由现有点组合得到。
3. **球杆二维姿态。** 球杆分割模型输出轮廓，通过 PCA 主轴确定两端，
   靠近手腕的一端映射为 `shaft`，另一端映射为 `hosel`。
   它们是分割几何得到的代理点，不是独立训练的 GolfPose 标记点检测结果。
4. **组织 17+5 输入。** 当前独立观测的球杆点只有 `shaft`、`hosel`。
   `heel`、`toe_down`、`toe_up` 缺失时，在模型输入中填到 `hosel` 位置，
   原始坐标仍为缺失值、有效标记仍为 False。因此 22 点三维输出不代表检测到了全部 22 个二维点。
5. **二维提升为三维。** 根据人体框将坐标缩放、平移到虚拟 1280×720 画布并归一化。
   单张图片没有真实时间序列，因此将同一帧重复补足 MixSTE 的 243 帧输入窗口，
   再进行原始与镜像输入的三维预测平均，得到以根关节为原点的单视角三维姿态。
   MixSTE 利用空间注意力学习人体与球杆的关联，利用时间注意力处理轨迹；
   对重复静态帧而言，不存在额外的真实运动信息。
6. **融合参考。** 将各视角姿态归一化并对齐，以正面姿态为基础，
   使用沿目标线视角替换其对应侧向分量，保存 `joints.npz:reference_3d`，
   并渲染 `reference_3d.jpg`。背面结果保留供检查，不参与当前参考融合。

这种融合不是基于已标定相机的多视角三角测量，输出的长度与深度不能直接视为精确物理测量。
`references` 不进行挥杆起止时间识别，也不运行视频的球杆轨迹 refinement 或球检测。

### video：从视频到 P1-P10

视频逐帧复用人物检测、ViTPose 人体关键点和球杆分割流程，并通过人物框重叠进行
单人关联、通过球杆跳变过滤和轨迹处理减少抖动。默认 refinement 在确定时间窗口之前，
用球杆角度/投影长度补全轨迹，并结合以人体为参照的手腕升降判断起杆、顶部、击球和结束。
球检测若可用，只能在手腕返回附近的时间窗口与距离约束内修正击球帧。
这些是启发式规则，不是额外训练的挥杆阶段分类模型。

确定时间窗口后，MixSTE 对真实二维序列执行三维提升，长序列的重叠窗口采用加权融合。
随后用二维观测、MixSTE 深度先验和时间平滑约束拟合固定长度的三维球杆段。
若没有提供实测标记间距离，长度来自可靠帧的模型预测中位数，属于推断值。
P2、P3、P5、P6、P8、P9、P10 再通过手臂/球杆方向及相邻阶段的顺序约束选出。
最终保存十个源视频帧、十个三维姿态图和显式十帧坐标包；人工提供的
`--positions-json` 优先于自动选择。缺少可靠点或关键姿态时会报告不完整结果。

### score：从十个三维姿态到分数

评分读取视频的 `identified_frames.npz/json` 与参考的 `joints.npz:reference_3d`，
按 P1 对 P1、P2 对 P2，直到 P10 对 P10 逐一匹配。分别以髋中心平移、以躯干长度
归一化后，计算既有的 14 项姿态角度，将用户与参考的角度差转换为 0-100 相似度，
再对可用指标等权平均。三维图的朝向对齐用于展示，不替代实际角度计算。

这一阶段使用 NumPy 等数值运算，不重新做姿态推理、不重新寻找关键帧，也不比较 JPG 像素。
评分没有新增 heel/toe 指标，且不代表球技、击球质量或经标定的三维重建准确率。
详细公式见 [SCORING_METHODOLOGY.md](SCORING_METHODOLOGY.md)。

实现入口见 [cli.py](golf_workflows/cli.py)、
[standard_frames.py](golf_workflows/standard_frames.py)、
[vitpose_tracking.py](golf_workflows/vitpose_tracking.py)、
[golf_pose_3d.py](golf_workflows/golf_pose_3d.py) 和
[compare_standard_frames.py](golf_workflows/compare_standard_frames.py)。

## 1. Build ten references

`references` reads the ten JPG sheets in `pro_refs/Standard_10_frames`, crops
face-on/down-the-line/rear views, runs body and club inference, lifts the views
with MixSTE and applies the existing alignment/fusion. Face-on supplies the
reference space; the aligned down-the-line pose supplies its side component.
The rear view is retained for inspection.

Output: `runs/workflows/references/`, with `workflow.json` and ten position
folders. Each contains `reference_3d.jpg`, `joints.npz` (numerical coordinates),
`joints.json` and camera images/poses. Fresh inference also writes `run_summary.json`.
Completion requires one finite reference and readable image per P1-P10 position.

Club landmarks keep the official GolfPose channel order: `shaft`, `hosel`,
`heel`, `toe_down`, `toe_up`. The workflow does not attach the shaft to the
wrists or force a shaft/head size. When one view has all five reliable
landmarks, it may also write `reference_3d_club_overlay.jpg` and
`joints.npz:reference_3d_club_overlay` by copying that view's already aligned
MixSTE club coordinates. This is a labeled visualization overlay only.
`joints.npz:reference_3d` is unchanged and remains the sole scoring input;
`joints.json:reference_club_overlay` records the source view and confidence.
The base `reference_3d.jpg` is body-only so imputed or out-of-frame club points
cannot be mistaken for observations.

To reproduce the paper's separate-club route, select the restored two-class
GolfPose YOLOX-s detector as the club crop source. Class `1` supplies the
per-frame club box, and the existing club-only ViTPose-H model supplies the
five landmarks. This is per-frame detection and pose estimation, not a learned
temporal tracker:

```powershell
python workflow.py references --club-crop-backend official-yolox-2cls
python workflow.py video <video.mp4> --club-crop-backend official-yolox-2cls
```

The default `segmentation` crop backend remains available for A/B comparison.

Reuse completed references without new inference:

```powershell
python workflow.py references --from-run runs/standard_frames_vitpose_huge_17plus5_20261027
```

## 2. Identify ten video frames

`video` performs body inference, club segmentation, five-landmark club pose,
phase identification, MixSTE lifting and P1-P10 selection. The launcher defaults
to ViTPose-Huge for the body, `golfpose_club_ViTPose_huge.pth` for
`shaft/hosel/heel/toe_down/toe_up`, GolfPose 17+5, and club refinement.
Segmentation selects the club and supplies its top-down crop; it is no longer the
source of the five 2D coordinates in the default workflow.
Refinement uses angular gap filling, blended windows and a constrained shaft
segment. Its length is inferred unless `--club-marker-length-m` supplies a
measured marker distance. See [the improvement log](SHAFT_TRACKING_IMPROVEMENT_LOG.md).

Output: `runs/workflows/videos/<video-stem>/`:

```text
workflow.json
identified_frames.json           P labels and original zero-based frame indices
identified_frames.npz            exactly ten poses, joint names and P labels
P1.jpg ... P10.jpg               selected source pictures with overlays
P1_3d.jpg ... P10_3d.jpg          selected 3D pose images
<stem>_3d.npz                    full sequence and pre-refinement poses
<stem>_3d.json                   models, phases and refinement details
<stem>_3d.avi                    2D + 3D video
<stem>_3d.jpg                    impact preview
```

Automatic phase detection remains heuristic. A run cannot publish as complete
unless it supplies ten distinct, increasing indices with finite 3D poses.
With refinement enabled, timing combines refined club geometry with wrist
rise/return in body-relative coordinates. A finish requires sustained low wrist
speed with raised hands; without that evidence the interval extends to clip end.
Ball correction is bounded near the wrist return and runs before lifting.
`phase_detection` in the run JSON records the evidence and any fallback.
Reviewed indices can be supplied when needed:

```powershell
python workflow.py video pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4 --positions-json runs/golf_pose_3d_vitpose_huge_17plus5/b85ed0b4db566e63bce190824c2f2196/selected_frames.json
```

Use `--no-club-refinement` or `--pose-backend yolo26n-pose` to change those
stages. `--club-pose-backend segmentation --lifter 17+2` restores the earlier
two-proxy club input. A measured marker length requires refinement.

Export ten poses from a completed run without running models again:

```powershell
python workflow.py video --from-run runs/golf_pose_3d_club_refined_20261027/b85ed0b4db566e63bce190824c2f2196
```

Cached import preserves saved numerical results and model choices. It does not
apply new refinement or selection overrides. It exports the ten-pose bundle
and 3D images from the saved coordinates, preserving the original run.

## 3. Score the selected poses

```powershell
python workflow.py score
# Or choose both inputs explicitly:
python workflow.py score runs/workflows/videos/b85ed0b4db566e63bce190824c2f2196 --references runs/workflows/references
```

The scorer reads `identified_frames.npz/json`, matches P1-P10 to the reference
`joints.npz:reference_3d` coordinates and keeps the original video frame indices.
JPGs visualize these poses; their pixels are not numerical scoring inputs.
The scorer does not identify frames again.

The existing 14 posture criteria, normalization, circular angle differences
and equal weighting are retained. See [SCORING_METHODOLOGY.md](SCORING_METHODOLOGY.md).
The three extra 17+5 head points do not add scoring criteria.

Output: `runs/workflows/scores/<video-run>/`, containing `score.json`, provenance
in `workflow.json`, ten user pose images and ten reference comparison images.
The score reports position/parameter coverage. It does not establish calibrated
physical 3D accuracy or coaching validity.

## Organization and preservation

`workflow.py` is the public launcher. Eight shared modules moved into
`golf_workflows/`: `standard_frames.py`, `golf_pose_3d.py`,
`compare_standard_frames.py`, `score_swing.py`, `trace_swing_plane.py`,
`swing_phases.py`, `vitpose_tracking.py` and `club_refinement.py`.
`cli.py` connects the three stages and validates their saved data contracts.
Detector helpers, training scripts, models, datasets and previous runs stay in place.

Low-level commands remain available as modules, for example
`python -m golf_workflows.golf_pose_3d --help`. Historical code snapshots remain
unchanged; their old imports may need adjustment if executed directly.

A nonempty explicit output folder is rejected. Default outputs receive a
timestamp suffix on reruns. Latest-run pointers under `runs/workflows/latest_*.json`
update only after successful completion. Failed runs retain diagnostics but
cannot replace a completed active result.

## Reorganization validation (2026-10-27)

All three public commands were exercised with the completed ViTPose 17+5
references and refined video via `--from-run`, followed by `score`. This validates
staging, export and scoring without repeating expensive ViTPose inference.
Ten reference and video poses were exported, with all 140 comparisons available.
The result was **93.8/100**. A fresh reference build was also completed using
the real models: all 30 camera poses and ten references succeeded, under
`runs/workflows/references_fresh_validation_20261027`. The older 92.9 result
used different inputs and is preserved.

Tests cover the existing lifting/refinement behavior, the ten-pose contract,
original source indices, explicit reference/output routing, identical-pose
scoring, incomplete references, inconsistent manifests and output preservation.
All 33 tests passed with `python -m unittest discover -s tests -v`.
