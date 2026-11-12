# 高尔夫姿态工作流：技术说明与修改记录

本文分两部分：第一部分是当前框架（代码现在实际执行的流程），第二部分是最近的修改与尝试记录（最新在前）。已不再使用的后端和早期实验的说明已删除，原文保存在 `archive/docs_zh_20261106/WORKFLOW.md`（中文）和 `archive/docs_en/`（英文）。

P1–P10 帧识别的逐条规则见 [FRAME_RECOGNITION.md](FRAME_RECOGNITION.md)，评分公式见 [SCORING_METHODOLOGY.md](SCORING_METHODOLOGY.md)。

---

# 第一部分：当前框架

## 1. 两个工作流

| | `workflow.py`（YOLO26n 工作流） | `python -m golfpose_workflow`（GolfPose 联合工作流） |
| --- | --- | --- |
| 检测器 | 双类 YOLOX-s `golfpose_detector_2cls_yolox_s.pth`（person、club） | 单类 YOLOX-s `golfpose_detector_1cls_yolox_s.pth`（golfer） |
| 人体 2D | 人物框扩展 10% 后送入 YOLO26n `yolo26n-pose.pt`，17 个 COCO 关节映射为 GolfPose 人体通道 | 球手框（1.25 倍 UDP 扩展）送入 22 点 `golfpose_golfer_ViTPose_huge.pth`；置信度低的人体关节由 YOLO26n 补充 |
| 球杆 2D | 与球手/双手关联的 YOLOX 球杆框送入仅球杆 ViTPose-H `golfpose_club_ViTPose_huge.pth`；没有可关联的球杆框时以球手框细化 | 同一个 22 点模型的 `shaft`、`hosel` 通道 |
| 3D 提升 | MixSTE2 17+2 `golfpose_17+2_33.6_30.5_59.5.bin` | 同左 |
| 输出根目录 | `runs/workflows/` | `runs/golfpose_workflow/` |

两者都只把 17 个人体关节和握把 `shaft`、杆颈 `hosel` 送入 3D 提升；杆跟/杆趾三点在提升前丢弃，记录在 `dropped_2d_landmarks`（手机视频上这三点的中位置信度只有约 0.6，与杆颈的距离误差达到杆头宽度的十倍）。`--lifter 17+5` 仍可保留全部五点，但不是默认。

另有一个独立的仅人体工作流 `python -m blazepose_workflow`（MediaPipe BlazePose GHUM，独立环境 `.venv_blazepose`，输出在 `runs/blazepose_workflow/`），不做球杆，3D 直接取自 GHUM 世界坐标而不用 MixSTE 提升，说明见 [blazepose_workflow/README.md](blazepose_workflow/README.md)。

可选模型：
- YOLOE `yoloe-26n-seg.pt`（类别 `golf ball`）：找球，用于确定 P7。文件存在即启用。
- YOLO 球杆分割 `MODEL_PATH`：杆身角度的补充测量（`--no-club-mask` 关闭）。

## 2. 视频流程（`golf_workflows/golf_pose_3d.py`：`main` → `detect_2d` → `process`）

1. **默认球手 ROI**（`golfer_roi.py`）：YOLO26n 在 12 个均匀采样帧上取评分 ≥ 0.4 的最大人物，保留与中位框 IoU ≥ 0.3 的框（至少 3 个），取并集后为球杆扩展（左右各 0.15 框高，上 0.25，下 0.05）。YOLOX 漏检时使用这个 ROI，检测到时始终以检测为准。`--golfer-box` 手动覆盖，`--no-auto-golfer-box` 关闭。
   - 人物选择（`workflow.py`，`detector_pose.YOLOXBodyPose`）：第一帧以 ROI 作为“上一帧框”，只在与之重叠（IoU ≥ 0.1）的人物中取最高分；之后只跟随与上一帧框重叠的人物，没有重叠的人物视为漏检，退回 ROI 裁剪，而不是跳到旁观者身上。
2. **逐帧 2D 检测与自动纠正**：
   - 人体框漂移纠正：框内骨架不完整（置信度 ≥ 0.25 的关键点少于 12 个，或缺任一肩，或两腕都缺）时，在上一帧完整骨架范围的 1.6 倍区域和球手 ROI 中重试，保留关节最多的结果（`skeleton_recovery`）。
   - 预测杆身裁剪：球杆置信度 < 0.3 时，按跟踪器预测的杆身位置重新裁剪推理一次，置信度提高才采用（`predicted_shaft_crops`）。
   - 片尾裁掉（`trailing_absence`）：在一段“有人物检测（分数 ≥ 0.25）→ 无检测”的交界处，若从这里到视频结束有检测的帧不超过 10%，且这段长于骨架缺口上限 max(3 帧, 0.3 秒)（片尾卡片、镜头切走），只使用之前的帧，记录在 `clip_trim`。
   - 视频从上杆顶点之后才开始（双手在开头 0.1 秒内最高，之后只下降，降幅 ≥ 双手高度范围的一半）时，P1–P3 不在视频中，打印 `ERROR:` 并抛出 `ClipCoverageError`（`phase_detection.clip_starts_after_top`）。`--positions-json` 可绕过。
3. **视角与左右手习惯**（`position_selection.infer_view` / `infer_handedness`）：默认自动，无法判断时回退到 `down_the_line` / `right`。
4. **球杆跟踪**（`club_tracking.track_club`）：以双手为锚点的刚性杆，恒角速度卡尔曼滤波加 RTS 平滑。几何确定握把/杆颈方向，置信度 ≥ 0.5 的检测始终接受，低置信度离群检测被剔除。检测不可靠的帧上补充运动模糊条纹和分割掩膜的杆身角度（`club_measurements.py`）。
5. **阶段识别**（`swing_phases.detect_refined_phases`，2D）：P1（杆头静止优先，否则手部启动）、P4（球杆转向优先，否则手腕峰值）、送杆结束。
6. **P7 = 击球瞬间**：在手腕回落点前 0.1 秒到后 0.3 秒的窗口内，依次采用：球离开原位 → 杆头到达球（检测或跟踪的杆颈）→ 杆头回到准备位置 → 手腕回落点。依据记录在 `phase_detection.impact_source` 和 `impact_contact`。
7. **骨架质量检查**（`check_skeleton`）：P1 到送杆结束内，9 个核心关节有效少于 6 个或两腕都缺即为丢失帧；连续丢失超过 max(3 帧, 0.3 秒) 或丢失比例超过 15% 时打印 `ERROR:` 并抛出 `SkeletonQualityError`。总会写出 `skeleton_diagnostics.json`。`--allow-skeleton-gaps` 改为警告。
8. **3D 提升**：只提升 P1 到送杆结束的窗口。坐标缩放到虚拟 1280×720 画布并归一化，MixSTE2 以 243 帧窗口运行（不足时补帧，较长时重叠窗口加权融合），原图与镜像预测取平均。随后以 2D 观测、MixSTE 深度先验和时间平滑拟合固定长度的 3D 球杆；杆长除非用 `--club-marker-length-m` 提供，否则是可靠帧上模型预测的中位数（推断值）。
9. **其余位置**（`position_selection.select_positions`，3D）：正面视角先做地面校平，再在 (P1, P4)、(P4, P7)、P7 之后三段内用动态规划联合选取 P2/P3、P5/P6、P8/P9/P10，输出每个位置的置信度与亚帧时间。旧的逐个贪心规则结果保留在 `rule_based_positions` 中作对照。
10. **导出**：十个源帧 `P*.jpg`、十张 3D 图 `P*_3d.jpg`、`identified_frames.npz/json`。

`--positions-json` 提供人工复核的帧号时跳过第 5、6、9 步。`--no-club-refinement` 时退回旧的杆头/手腕速度规则和贪心选取。

## 3. 参考构建（`golf_workflows/standard_frames.py`）

`references` 读取 `pro_refs/Standard_10_frames/` 中的十张参考图，每张裁出正面、后方、背面三个视角，各自做 2D 检测和 3D 提升（单张图重复补足 243 帧窗口），再在正面视角空间中对齐。

- **人体参考**：只在 `face_on` 与 `down_the_line` 中选，比较前 17 个人体关节的平均置信度（缺失关节计 0），平分时选 `face_on`。所选视角的完整对齐姿态写入 `joints.npz:reference_3d`，不混用另一视角的分量。`rear` 只供检查。
- **球杆叠加图**：同样只在两个视角中选。所有球杆通道都有效且握把到杆颈 ≥ 12 像素的视角优先，分数 = min(置信度) + 0.25 × mean(置信度) − 0.05（裁剪回退时）。都不完整时，允许杆身置信度 ≥ 0.10 的视角作为“仅杆身”回退，缺失点标注为推断。叠加图只用于展示，不改变 `reference_3d`。
- 输出：`reference_3d.jpg`、`reference_3d_club_overlay.jpg`、`joints.npz/json`，以及 `reference_3d_preview.png` 等拼图。`--from-run` 复用已完成的参考，不重新推理。

参考不做阶段识别，也不运行球杆跟踪或球检测。

## 4. 评分

评分读取视频的 `identified_frames.npz/json` 和参考的 `joints.npz:reference_3d`，按 P1 对 P1……P10 对 P10 匹配。以髋中心平移、躯干长度归一化后计算 14 项姿态角度，把角度差换算为 0–100 相似度，再对可用指标等权平均。不重新推理、不重新找帧、不比较图片像素。分数是与参考的相似度，不代表击球质量或标定过的 3D 精度。

## 5. 输出与数据约定

视频运行目录：

```text
workflow.json                    工作流清单（视角、球杆跟踪摘要等）
identified_frames.json/.npz      P1–P10 帧号（从 0 开始）与十个 3D 姿态
P1.jpg ... P10.jpg               源帧叠加图
P7_contact.jpg                   击球前后两帧（最后停留帧 | 消失帧）的球位裁剪
P1_3d.jpg ... P10_3d.jpg         3D 姿态图
<stem>_3d.npz                    完整序列：2D 坐标/有效性/置信度、3D、club_track_* 数组
<stem>_3d.json                   模型、阶段、识别依据、球杆跟踪、骨架检查
skeleton_diagnostics.json        每帧核心关节数与丢失帧段
<stem>_3d.avi / <stem>_3d.jpg    2D+3D 视频与预览
```

`<stem>_3d.json` 的主要字段：`positions`、`phases`、`phase_detection`、`ball`、`impact_contact`、`checkpoint_selection`、`view_detection`、`handedness_detection`、`club_tracking`、`shaft_angle_measurements`、`predicted_shaft_crops`、`body_fusion`、`skeleton_check`、`detector_crops`、`pose_backend`。含义见 [FRAME_RECOGNITION.md](FRAME_RECOGNITION.md) 第 10 节。

约定：
- 只有十个帧号互不相同、递增且 3D 姿态全为有限值时才算完成。
- 默认输出在重复运行时加时间戳后缀，不覆盖已有结果；显式指定的非空输出文件夹会被拒绝。
- `runs/<工作流>/latest_*.json` 指针只在成功完成后更新；失败的运行保留诊断文件（如 `pose_inference_diagnostics.json`、`skeleton_diagnostics.json`），但不替换当前结果。
- 输出目录和模型文件不纳入 git（见 `.gitignore`）。

代码布局：`workflow.py` 是 YOLO26n 工作流的启动器，`golfpose_workflow/` 是联合工作流，共用模块在 `golf_workflows/`（`golf_pose_3d.py`、`swing_phases.py`、`position_selection.py`、`club_tracking.py`、`club_measurements.py`、`club_refinement.py`、`detector_pose.py`、`golfer_vitpose.py`、`golfer_roi.py`、`standard_frames.py`、`score_swing.py`、`benchmark.py` 等）。不再使用的脚本移到了 `archive/`（见 `archive/README.md`）。

## 6. Python 环境

系统 Python 3.12 中的 `torch` 混入了 CUDA 库，导入会报 WinError 126，因此项目使用独立的 `.venv`：

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = "$PWD\.tmp"; $env:TMP = "$PWD\.tmp"; $env:PIP_CACHE_DIR = "$PWD\.tmp\pip-cache"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install --no-compile -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` 固定了 NumPy、SciPy、OpenCV、Matplotlib、Ultralytics（含 CLIP 分词器的源码提交）、timm 0.9.16、einops、local-attention 的版本。推理全部在 CPU 上运行，不需要 MMEngine/MMPose/MMDetection。ViTPose 与 MixSTE 的实现内置在 `golf_workflows/vitpose_tracking.py` 和 `third_party/golfpose/`。

测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

用 `.tmp` 下的脚本回放时，需要先设置 `$env:PYTHONPATH='J:\Golf_Posture'`。

---

# 第二部分：最近修改与尝试记录

## P7：未拍到触球时，杆头仍在球旁则取消失帧（2026-11-10）

**反馈**（`runs/workflows/videos/008_20261110_111204_195895`）：`P7_contact.jpg` 中更认可右边的 215（球已消失）而不是 P7 所取的 214（球在原位）。

**逐帧检查**：
- 214：杆头在球后约 0.8 躯干长度，正在接近。
- 215：球已消失，模糊的杆头正好扫过球位。
- 杆颈检测在这两帧都离球 450–500 px，接近速度无法外推，于是按旧规则回退到“最后停留帧”。
- 010（167/168）的情况相同：168 球刚飞出，杆头就在球位旁。
- w03（38/39）不同：39 杆头已摆到髋部高度，38 更接近触球。2026-11-06 的记录里“取消失帧为默认”因 w03 被否决，所以不能一律改取消失帧。

**修改**（`golf_pose_3d.py`）：
- 新增 `clubhead_motion`，只在“杆头未确认且无法外推”时使用。它在球位附近的地面带内用帧差找运动的杆头，找的是同时与顶点帧和另一帧都不同的像素，这样能排除脚和阴影。结果记录在 `clubhead_motion_torso`。
- 消失帧上杆头距球位 ≤ 0.5 躯干长度时，P7 取消失帧（`clubhead_motion_choice`），否则仍取最后停留帧。
- `P7_contact.jpg` 中被选为 P7 的一侧标注 `= P7`。

| 视频 | 停留帧 / 消失帧 | 杆头到球位（躯干长度） | P7 |
| --- | --- | --- | --- |
| 008 | 214 / 215 | 0.25 / 0.09 | 214 → 215 |
| 010 | 167 / 168 | 0.09 / 0.09 | 167 → 168 |
| w03 | 38 / 39 | 0.32 / 0.75 | 38 不变 |

重跑（`008_20261110_114648_534888`、`010_20261110_124645_499963`）中 P8 随之后移一帧（008：222 → 223；010：173 → 174），其他位置不变。w09、005、w07 拍到了触球，w01 由接近速度外推，都不经过这条规则。新增单元测试 `test_clubhead_motion_measures_both_contact_frames`。

## 新增：独立的 BlazePose GHUM 人体工作流（2026-11-10）

**目的**：参照 tfjs-models `pose-detection` 的 BlazePose 与 BlazePose GHUM 3D 博客，建立一个只做人体（不做球杆、球、面部）的独立工作流，与上面两个工作流互不依赖。代码、依赖和说明都在 `blazepose_workflow/`（详见 [blazepose_workflow/README.md](blazepose_workflow/README.md)）。

**实现**：
- 运行时是 Python `mediapipe==1.2.2` Tasks `PoseLandmarker`（VIDEO 模式），模型与 tfjs 相同的 GHUM lite/full/heavy，首次使用时下载。独立环境 `.venv_blazepose`（已加入 `.gitignore`）：mediapipe 自带的 `opencv-contrib-python` 与主环境的 `opencv-python` 冲突。
- 33 点去掉面部 1–10，保留鼻子和 11–32，共 23 点；每点有 2D 像素坐标和 3D 世界坐标（米，髋中心原点）。
- 两遍推理：全画面 `num_poses=3` 选球手（连续性锚点，否则居中且最大），再在球手 ROI 上放大推理。
- 清理：按 presence 丢点、0.3 秒内插值、Savitzky–Golay 平滑、3D 骨长归一化。
- P1–P10 只用双手高度和前臂 3D 抬起角度；视频不含完整挥杆时抛出 `SwingCoverageError`。
- 命令：`.\.venv_blazepose\Scripts\python.exe -m blazepose_workflow video <视频> --model heavy`，输出在 `runs/blazepose_workflow/videos/`。13 项单元测试（合成挥杆）通过；主项目 104 项测试不受影响。

**随机测试：`pro_refs/005.mp4`**（后方视角、右手，29.95 fps，573 帧；`runs/blazepose_workflow/videos/005_20261110_105239`）：
- 球手在 487/573 帧中检出；500 帧以后是没有人的片尾。
- 自动判断为 `down_the_line`、`right`，正确。
- 结果 P1 109、P2 126、P3 144、P4 176、P5 201、P6 205、P7 221、P8 242、P9 254、P10 276。主工作流对应为 P2 122、P3 150、P4 177、P5 198、P6 216、P7 225、P8 ≈ 241、P9 253、P10 304。

**调试中修正的规则**：
- **P7 早了 7 帧（214）**：原规则是“双手回到准备位置”，但 005 的双手在击球前就已回到准备高度。改为顶点后双手最低点。
- **P2 晚**：原规则取最后一次穿过角度，改为第一次。
- **视角误判为正面**：斜后方机位的 2D 肩宽 0.356 像正面，但 3D 肩线主要沿深度方向（左右占比 0.199）。改为先看 3D。
- **P7 变成 233**：缺失帧的双手高度被填成最小值。改为插值。
- **送杆双手整段缺失（233–260）**：后方视角前侧手腕被身体挡住，visibility ≈ 0。2D 改为只按 presence 丢点，遮挡关节保留模型预测值，并单独记录遮挡数。
- **P8 紧贴击球（230）**：GHUM 的前臂在击球时已有约 21°，固定 45° 在击球后两帧就达到；肘部在送杆中最多读到 149°，“双臂伸直”条件也帮不上忙。P2/P6/P8 改为“准备或击球时角度与 90° 的中点”，P8 取击球后第一次上升穿过该角度（242），P6 取下降中最后一次穿过以避开单帧抖动。

**3D 坐标轴**：博客中 tfjs `keypoints3D` 的 z 是靠近相机为正，而 Python `pose_world_landmarks` 是远离相机为正（005 上离相机近的右肩 z = −0.18，左肩 +0.04），y 向下。保存的数据使用 MediaPipe 约定，未转换。

## 修复：009 骨架跟到旁观者、片尾卡片、视频从顶点开始（2026-11-09）

**现象**（`runs/workflows/videos/009`，`workflow.py`，29.94 fps 慢动作）：骨架检查中止，P1–送杆结束（97–508）丢失 65 帧（16%），最长连续 24 帧，丢失段全在 420–508。

**原因**：
1. **骨架跟到了旁观者**。画面左边缘站着一名保安。第一帧 YOLOX 给他 0.950 分，给球手 0.939 分。人物选择没有先验，取了最高分的保安，之后按与上一帧框重叠一直跟着他。保安几乎不动，双手高度在 0–416 帧一直约 0.52 躯干长度，所以：
   - 阶段识别找不到“手抬起再落下”，回退到只用杆头的规则（`no_corroborated_wrist_rise_and_return`），给出准备 97、顶点 106、击球 351，都不对。
   - 左右手由第 507 帧（片尾）判断，结果为 `left`。
   - 球杆点在真实球手身上，因为球杆裁剪不依赖人物框。
   - 自动球手 ROI [159, 48, 705, 1006] 是正确的，但只在 YOLOX 漏检时使用，从未参与人物选择。
2. **片尾卡片**。420–508 帧是黑底片尾（没有人）。阶段识别回退时把送杆结束定在最后一帧，片尾被算进挥杆窗口。姿态模型在黑屏上还会给出 5–9 个核心关节的假骨架，所以只看骨架分不清“片尾”和“丢失球手”。用 `--allow-skeleton-gaps` 继续时，P8/P9 落在片尾（431/432），P10 缺失。

**修改**：
- `official_yolox.select_box` 新增 `require_overlap`：有上一帧框而没有重叠的人物时，视为漏检。
- `detector_pose.YOLOXBodyPose`：
  - 第一帧以球手 ROI 作为上一帧框。009 第一帧选中球手（IoU 0.31），保安与 ROI 的 IoU 只有 0.015。
  - 人物选择要求重叠，漏检时退回 ROI 裁剪。
  - ROI 或重试裁剪得到完整骨架时，用该骨架范围引导下一帧，避免沿用过期的框。
- `golf_pose_3d.trailing_absence` / `trim_swing`：
  - 在“有人物检测 → 无检测”的交界处，若此后到视频结束有检测（`selected_person_score` 或 `selected_golfer_score` ≥ 0.25）的帧不超过 10%，并且长于骨架缺口上限，只使用之前的帧。结果记录在 `clip_trim`。
  - 009 中球手分数约 0.93，片尾 89 帧中只有 7 帧 0.25–0.35 的误检，裁剪到 0–419 帧。
  - 最初用“分数 < 0.5 直到结束”，但球手收杆姿势的分数会降到 0.42，误裁了 407–419 帧，因此改为检测器本身的最低分 0.25，再容忍少量误检。

**修复后重跑仍不对**（`runs/workflows/videos/009_20261109_172715_471620`）：骨架已在球手身上，但阶段识别仍回退（P1 112、P4 155、P7 380）。原因是**视频从上杆顶点开始**：双手高度第 0 帧 1.33 躯干长度（最高），第 138 帧降到 −0.21，收杆升到 1.64。“手先抬起再落下”的抬起部分在视频开始前就已完成，准备、P2、P3 根本不在视频里，任何校准都无法识别。

- `swing_phases.detect_refined_phases`：找不到“抬起再落下”时，若双手在开头 0.1 秒内最高，之后一直下降，且降幅 ≥ 双手高度范围的一半，记录 `clip_starts_after_top`。
- `golf_pose_3d.process`：遇到这种情况打印 `ERROR:` 并抛出 `ClipCoverageError`，说明缺少 P1–P3，不再输出编造的位置。提供 `--positions-json` 时不检查。
- 在保存的 2D 上回放：只有 009 触发；w09、008、010、005、w07、w01、w03 都正常识别。

单元测试：`test_golfer_roi_beats_higher_scoring_bystander`、`test_outro_without_golfer_is_trimmed`、`test_clip_starting_at_the_top_is_flagged`，共 104 项通过。

**要让 009 跑通**：需要从准备姿势开始的片段。原视频若更长，截取包含准备姿势的部分；否则只能用 `--positions-json` 人工给出帧号。

## P7：击球瞬间落在两帧之间时如实报告（2026-11-06）

**现象**（`runs/workflows/videos/w03`，后方视角，右手）：P7 = 38，画面中杆头还没碰到球。

**逐帧检查**（原始视频，球位附近放大）：
- 38 帧：球仍在球座上，杆头在球的后侧（画面右侧），相距约 120 px（约 0.9 躯干长度），正在接近。
- 39 帧：球已飞出画面，杆头已摆到前导侧髋部高度。两帧之间杆头移动约 600 px。
- 帧时间戳：视频是可变帧率（30 fps 基准，部分位置丢帧，平均 27.37 fps），但 38→39 之间是正常的 33.3 ms，没有丢帧。
- 按杆头速度估算，触球约在 38.25 帧。**视频里不存在杆头贴着球的那一帧**；38 是离触球最近、且球仍在原位的帧，选择本身正确。

**考虑过但没有采用**：
- 调整 P7 窗口：窗口 [35, 46] 已包含 38 和 39，问题不在窗口范围。
- 外推时允许使用跟踪器的杆头：37 帧跟踪值 400.6 px、38 帧检测值 212 px，外推到达 39.1，会选 39 帧（球已飞走、杆头已远离），更差。38 帧的杆颈检测（212 px）也比目测（约 120 px）偏远，距离数据不足以做亚帧插值。
- 取消失帧为默认：010、w03 的消失帧中杆头已越过球很远，不如最后停留帧接近触球。

**修改**（`golf_pose_3d.py`），选帧规则不变，改为如实报告：
- `contact_frame` 在球离开原位时记录 `contact_between` = [最后停留帧, 消失帧]，以及 `contact_captured`：两帧中是否有杆头在球旁 0.6 躯干长度以内。
- 新增 `save_contact_bracket`：把这两帧在球位附近（±2.5 躯干长度）的裁剪并排存为 `P7_contact.jpg`，并标出球位。同时读取容器时间戳，记录两帧真实间隔 `interval_ms`、基准帧间隔 `nominal_interval_ms` 和其间丢帧数 `dropped_frames`，因为手机录屏常是可变帧率，名义 fps 会低估击球处的时间间隔。结果存于 `impact_contact.bracket`。
- `contact_captured` 为假时打印警告：触球落在哪两帧之间、相隔多少毫秒、P7 实际取了哪一帧。

**验证**（保存的 2D 回放，启用球与掩膜模型）：

| 运行 | P7 | `contact_between` | 触球被拍到 |
| --- | --- | --- | --- |
| w09 golfpose | 31 | 30–31 | 是 |
| 005 golfpose | 225 | 224–225 | 是 |
| w07 golfpose | 34 | 33–34 | 是 |
| 008 | 214 | 214–215 | 否（杆颈检测离球约 500 px） |
| 010 | 167 | 167–168 | 否（168 帧球已离开杆面） |
| w01 | 45 | 44–45 | 否（按接近速度外推取 45） |
| w03 | 38 | 38–39 | 否（触球约在 38.25） |

所有 P7 均不变，七段视频都没有丢帧。要得到真正的触球画面，需要 60 fps 以上的原始视频（不是录屏）。替换了 `tests/test_refined_phases.py` 中重名的重复测试，新增 `test_contact_between_frames_is_reported`。

## P8/P9 按定义重写，修复正面视角左右手误判（2026-11-06）

**定义**（HackMotion）：
- P8 释放：首先双臂完全伸直，其次杆身与地面平行。
- P9：P3 的反面，前导臂在前导侧与地面平行。

**008 的问题**（`runs/workflows/videos/008`）：输出 P8 = 219、P9 = 227。
- 主要原因：左右手误判为 `left`。正面视角规则取“双手最高帧”当上杆顶点，但 008 收杆时（第 296 帧）双手比上杆顶点（第 171–179 帧）更高，收杆时双手又移向前导侧，于是方向判反了。
- 后果：P3/P8/P9 用的“前导臂”其实是右臂（后侧臂）。
- 次要原因：P8 的旧规则只看伸展角（杆身只作 1/90 权重的平局决胜），而 3D 提升后的肘角在击球后一直停在 140–155°，分不出伸直与否。
- 219 帧时双手还在髋部下方，杆身朝下，不是 P8。

**修改**（`position_selection.py`）：
- **左右手**（`_high_hand_peaks`）：在双手高度上找明显的高峰，第一个是上杆顶点，最后一个是收杆，按两者之间的水平位移判断。008、010 都判为右手（010 原先是“不明确”，回退到默认值）；四个后方视角视频不变。
- **P8**：只在前导臂已从下垂抬起 ≥ 25%、且双臂伸展在 P7–P9 最大值 10° 以内的帧中选。其中可靠杆身最接近水平者优先；看不到杆身时，用前导臂代替，目标为抬起 45%（复核帧：w09 0.58、005 0.48、008 0.40）。“抬起 25%”这一门槛排除了 005 第 234 帧在手臂仍下垂时被误检为平行的杆身。
- **P9**：正面视角改用 2D 观测的前导臂穿过水平，3D 约滞后一帧。P3/P5 试过同样改用 2D，但 010 中变差（P3 109 → 113，P5 154 → 149；目视为 108–109 和 154），已撤回，仍用 3D。
- 去掉了联合选取中 `inf − inf` 引起的 `RuntimeWarning`。

**验证**（保存的 2D 回放，启用球与掩膜模型）：

| 运行 | 之前 P8 / P9 | 现在 P8 / P9 | 对照 |
| --- | --- | --- | --- |
| 008 | 219 / 227（左手误判） | 223 / 227 | 223：双臂伸向目标，手在腰部，后侧臂即将翻过；227：前导臂与肩线重合 |
| w09 golfpose | 32 / 33 | 32 / 33 | 标注 P8 32、P9 33–34 |
| 005 golfpose | 242 / 253 | 241 / 253 | 标注 P8 240–244、P9 252–254 |
| 010 | 172 / 180 | 173 / 180 | — |
| w07 golfpose | 35 / 37 | 35 / 37 | — |
| w01 | 47 / 48 | 47 / 48 | — |

008 的 P3 因改用正确的前导臂，从 148 变为 145。新增 `tests/test_position_selection.py`（4 项），共 100 项测试通过。

## 修复：w01 的 P7 早一帧（2026-11-06）

**现象**（`runs/workflows/videos/w01`，28.5 fps，后方视角）：P7 = 44，此时杆头还在球右上方约 92 px；真实击球帧是 45（杆头压在球上把球挡住），46 帧球已飞出、只剩草皮。

**原因**：击球帧 45 的 2D 恰好最差：两腕都丢失，杆颈检测（置信度 0.17）落在手的位置，跟踪器因缺少双手锚点而沿旧状态滑行。消失帧 45 与前一帧 44 的杆头都不在球旁 0.6 躯干长度（81 px）以内，于是进入“杆头无法确认”的回退，取了“球最后一次停在原位的帧”44。这个回退默认球先飞走、再被发现消失；可是触球时杆头挡住球，消失帧本身就是击球帧。

**修改**（`golf_pose_3d.approach_time`）：杆头无法确认时，用击球前最后两帧连续检测到的杆头到球距离按接近速度外推到达时刻。w01：43 帧 204 px、44 帧 92 px，每帧接近 112 px，到达时刻 44.82，取 45。只有杆头正在接近且已在 1.2 躯干长度内才外推，否则仍取最后停留帧（008、010 的杆颈检测离球几百像素，不外推，结果不变）。

**尝试过但放弃**：比较球位处图像块与球飞走后的空球位，判断“球是被杆头挡住还是已飞走”。w01 第 45 帧杆头在 16 px 图像块边缘之外（得分 0.79），草皮被打起后的变化与球本身相当；005、w07 还因此各早了一帧。已撤销。

**验证**（保存的 2D 回放，启用球与掩膜模型）：w01 44 → 45；w09 31、008 214、010 167、005 225、w07 34 不变。新增单元测试 `test_blurred_impact_uses_clubhead_approach`。

## P7 以击球瞬间为先（2026-11-06）

**要求**：P7 以击球瞬间为最优先条件，动作捕捉条件其次。职业球手击球时手腕内扣（前导腕弓起），杆身前倾、杆头落后于双手；这一动作在 2D 关键点中看不到，按手腕最低点找 P7 会偏早。

**之前**：P7 = 手腕回落点（`wrist_impact`），在 ±0.12 秒内用杆头回到准备位置修正，再在 ±0.2 秒内用杆颈离球最近的帧修正。窗口对称，且只认检测到的杆颈；击球瞬间杆头最模糊，常常检测错误或缺失。

**现在**（`swing_phases.impact_window`、`golf_pose_3d.ball_presence`、`contact_frame`）：
- 窗口改为 [`wrist_impact` − 0.1 秒, `wrist_impact` + 0.3 秒]（至少 −2/+3 帧），给杆头落后于双手留出余量。
- 判定顺序：
  1. 球离开原位（`ball_departure`）：消失帧与其前一帧中，杆头离球更近的那帧；杆头无法确认时，按击球前的接近速度外推，无法外推才取球最后一次停在原位的帧（见上一条修复）。
  2. 杆头到达球：检测到的杆颈（`clubhead_at_ball`），否则用跟踪器推算的杆颈并加 0.1 躯干长度惩罚（`tracked_clubhead_at_ball`）。
  3. 杆头回到准备位置（`clubhead_return_to_address`）。
  4. 手腕回落点（`wrist_return`）。
- 球位检测改用 1280 像素输入：640 像素时，010 远景中的球在整个下杆段都检测不到；1280 像素时置信度约 0.6–0.8。
- 准备到击球之间相机会平移/缩放（010 中球位移动约 46 像素），因此在 0.5 躯干长度内重新确定球位并逐帧跟随，判定“在原位”的半径为 0.12 躯干长度。
- 新增 `<stem>_3d.json` 字段 `impact_contact` 和 `phase_detection.impact_source` / `impact_window`；运行时打印 `P7 source`。

**验证**：用保存的 2D 回放 `process`，启用球和掩膜模型（`.tmp/replay_p7contact.py`）：

| 运行 | 之前 P7 | 现在 P7 | 依据 | 目视/标注 |
| --- | --- | --- | --- | --- |
| w09 `workflow.py` | 29 | 31 | 球在第 31 帧后消失，杆颈距球 25 px | 标注 31 |
| w09 golfpose | 30 | 31 | 同上，39 px | 标注 31 |
| 008 `workflow.py` | 211 | 214 | 球停在原位到 214，215 消失；杆颈检测距球 450–500 px，无法确认 | 214 帧杆身扫过球 |
| 010 `workflow.py` | 165 | 167 | 球停在原位到 167，168 消失（1280 输入、跟随球位） | 166 帧球仍在球座上，168 已飞出 |
| 005 golfpose | 226 | 225 | 225 消失，杆颈距球 19.9 px | — |
| w07 `workflow.py` | 34 | 34 | 35 消失，杆颈距球 21.4 px | — |

008 和 010 的 P7 都比手腕回落点晚 2–4 帧，与杆身前倾的预期一致。新增 4 个单元测试（窗口不对称、球离开优先于最近杆头、杆头无法确认时的回退、只有跟踪杆头时的判定），共 95 项测试通过。

局限：球被遮挡或出画时退回动作捕捉规则；同一帧出现两个球时可能跟错，可从 `impact_contact.ball_presence` 检查。

## 人体裁剪框漂移的自动纠正与骨架异常中止（2026-11-06）

**问题**（`runs/workflows/videos/008_20261105_174213_259058`）：YOLOX 人物框在快速挥杆时漂移，YOLO26n 只在框内找人，于是丢失骨架。
- 第 147–202 帧：框滑到球手右半身和观众上，只剩腿部关节。
- 第 232 帧起：球手的 YOLOX 分数降到 0.39，检测器改选画面左侧的钟和广告牌，0 个关节。
- 缺失帧被插值后，P7 之后的位置都落在虚构的姿态上。

**纠正**：
- `workflow.py`（`detector_pose.YOLOXBodyPose`）：骨架不完整时，在上一帧完整骨架范围的 1.6 倍区域和自动球手 ROI 中重试，保留最好的骨架，并用它引导下一帧的选框。
- golfpose（`golfer_vitpose.GolferViTPose.recover_body`）：同样的条件和重试位置，关节更多才替换。
- 每帧的 `skeleton_usable`、`skeleton_recovery` 记录在 `detector_crops` 中。

**中止**：见第一部分第 2 节第 7 步。阈值取 0.3 秒，是因为上杆顶点时手臂被头部遮挡，会短暂丢失几帧，这不应中止运行。

**验证（008）**：

| | 挥杆窗口（87–321）内丢失帧 | 最长连续丢失 | 检查结果 |
| --- | --- | --- | --- |
| 修复前（已保存的 2D） | 158（67%） | 90 帧（232–321） | 中止 |
| 修复后（重跑人体检测） | 13（5.5%），在上杆顶点 161–174 及第 197 帧 | 5 帧 | 通过 |

修复后 322 帧中 59 帧改用自动球手 ROI，36 帧改用上一帧骨架范围。新增 `tests/test_skeleton_check.py`。

## 修复：球杆跟踪器误翻转导致 P7 提前（2026-11-05）

两个工作流在 w09 上的 P7 分别提前到第 29、30 帧，杆头距球 264–327 像素。原因是新的刚性杆跟踪器把高置信度且方向正确的检测判为握把/杆颈互换：击球附近杆身每帧转动上百度，恒角速度预测与真实方向相差很大，反向方向反而通过了门控；`workflow.py` 还因初始状态取自方向错误的检测，连准备站位的帧也整段翻转。

修改：
- 两端到手的距离差 ≥ 杆长 25%（`ORIENTED_FRACTION`）时，方向由几何确定，连续性不得反转。
- 置信度 ≥ 0.5（`REINIT_CONFIDENCE`）的检测始终接受，与预测不符时以它重新初始化。
- 初始状态只从方向明确且置信度可靠的检测开始。
- 新增 `tests/test_club_tracking.py`。

验证：w09 两个工作流的 P7 均为 31，与人工确认的击球帧一致（杆颈距球 25.3 / 39.2 px）。

## 稳健化第一阶段与第二阶段前三步（2026-11-05）

一次实现的一组改动，目的是让 P 位置识别不再依赖单帧阈值，并提高球杆在模糊帧上的可用性。

**第一阶段：位置选择**
- **基准与回放**（`golf_workflows/benchmark.py`、`benchmarks/p_position_labels.json`）：人工标注的帧范围（w09、005、008、010 的部分位置）；`evaluate` 计算偏差，`replay` 用已保存的 2D 重跑识别。
- **联合动态规划选取**（`position_selection.select_positions`）：取代逐个贪心选取，按段内总代价最小且严格有序选出 P2–P10；每个位置的置信度 = 1 − exp(−margin/2)。
- **亚帧时间**：穿越点线性插值、极值点抛物线拟合，保存为 `subframe_frames` / `subframe_ms`。
- **遮挡处理**：前导腕被遮挡（置信度 < 0.3）时按置信度混合后侧手腕；golfpose 中置信度 < 0.3 的 ViTPose 人体关节由置信度 > 0.5 的 YOLO26n 关节补上（`body_fusion`）。
- **自动视角、左右手习惯与地面校平**：见 [FRAME_RECOGNITION.md](FRAME_RECOGNITION.md) 第 3 节。

**第二阶段（前三步）：球杆**
- **刚性杆卡尔曼/RTS 跟踪**（`club_tracking.py`）：门控、剔除、方向纠正与不确定度。
- **条纹与掩膜测量**（`club_measurements.py`）：手部周围运动模糊条纹的加权 PCA 直线拟合，以及分割掩膜主轴。起初从手部出发取射线，偏差约 12°，改成直线拟合后中位误差约 1°。与可靠检测相比的中位误差：

  | 视频 | 条纹 | 掩膜 |
  | --- | --- | --- |
  | w09 | 1.7° | 1.4° |
  | 005 | 0.9° | 2.4° |
  | 008 | 1.2° | 0.4° |

- **预测杆身裁剪**（`club_tracking.ShaftPredictor`）：两个工作流在球杆置信度 < 0.3 时，沿预测杆身重新裁剪一次。

尚未完成：在全部标注视频上的完整基准回放。

## 自动默认球手 ROI（2026-11-05）

起因：
- golfpose 的单类 YOLOX 在 w09（竖屏后方视角手机录像）任何一帧上评分最高只有 0.04，没有 `--golfer-box` 就会失败。
- `workflow.py` 的整帧回退会选中画面中最大的人。

改为先用 YOLO26n 推导球手 ROI（第一部分第 2 节第 1 步），每个片段开销 1–5 秒。w09 的自动 ROI `100.5 307.0 487.7 991.2` 与手动选定的 `120 330 470 1000` 接近（12/12 帧一致）。ROI 记录在 `pose_backend.initial_golfer_box_xyxy`、`initial_roi_source`、`initial_roi_details`。

## 结合球杆的 P1、P4 与可靠球杆帧（2026-11-04）

- **可靠球杆帧**：握把和杆颈都被检测到，且两者中较低的置信度 ≥ 0.3（模糊杆身通常约 0.2）。
- **P1**：由“双手开始移动”改为“上杆顶点前杆头最后一次静止”。原因：
  - 准备阶段双手会缓慢漂移。005 中双手从第 84 帧开始漂移，杆头静止到约第 104 帧，P1 从 84 改为 108。
  - 008 的 P1 从 99 改为 87。
  - 跳变过滤剔除了 008 第 90–104 帧错误的杆颈检测。
- **P4**：由手腕峰值改为球杆停止向后转动的帧（两侧都要观察到转向）。上杆顶点处球杆常模糊或出画，实际多数视频仍退回手腕峰值。
- **P2/P6**：只用可靠球杆帧上接近水平的杆身，否则用前导臂从下垂到水平转过 30% 处。30% 来自 005（25%）和 008 复查帧（28%）。现在这两条作为联合选取的代价项。

## 基于手臂的 P3/P5/P8/P9/P10 与挥杆窗口终点（2026-11-04）

- P3/P5/P9 改为前导手腕穿过肩高度的时刻。原来“与水平偏差 0.25 以内”约为 14°，低帧率下会接纳远低于平行的手臂（w09 第 33 帧 −0.22，第 34 帧 +0.16）。
- P8 改为击球后双手开始上升、双臂最伸直的帧。原因：
  - 击球后杆身最模糊（005 第 236–246 帧杆颈置信度约 0.2）。
  - 击球时手臂同样伸直，需要“双手已上升”来区分击球与释放。
- P10 限定在双手最高点附近，避免选到放杆时双手经过面前的帧（w09 原为 45，现为 40）。
- 挥杆窗口一直延伸到抬起的双手重新放下，或到片段结束。005 原先在第 285 帧截断，P10 只能落在 P9 之后一帧。
- 这些规则现在都是联合选取中的代价项，见 [FRAME_RECOGNITION.md](FRAME_RECOGNITION.md) 第 7 节。

## 两个工作流统一为 17+2 3D 提升（2026-11-04）

两个工作流的 3D 提升都由 17+5 改为 17+2，只保留握把与杆颈。证据：010 上三个杆头点的中位置信度最低（0.58–0.63），与杆颈距离的第 95 百分位约 250 px，而一号木杆头宽度只有约 25 px。阶段识别、球杆精修和评分本来就只用握把和杆颈。之前用 17+5 构建的参考与评分需要重建才能同口径比较。
