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

可选模型：
- YOLOE `yoloe-26n-seg.pt`（类别 `golf ball`）：找球，用于确定 P7。文件存在即启用。
- YOLO 球杆分割 `MODEL_PATH`：杆身角度的补充测量（`--no-club-mask` 关闭）。

## 2. 视频流程（`golf_workflows/golf_pose_3d.py`：`main` → `detect_2d` → `process`）

1. **默认球手 ROI**（`golfer_roi.py`）：YOLO26n 在 12 个均匀采样帧上取评分 ≥ 0.4 的最大人物，保留与中位框 IoU ≥ 0.3 的框（至少 3 个），取并集后为球杆扩展（左右各 0.15 框高，上 0.25，下 0.05）。YOLOX 漏检时使用这个 ROI，检测到时始终以检测为准。`--golfer-box` 手动覆盖，`--no-auto-golfer-box` 关闭。
2. **逐帧 2D 检测与自动纠正**：
   - 人体框漂移纠正：框内骨架不完整（置信度 ≥ 0.25 的关键点少于 12 个，或缺任一肩，或两腕都缺）时，在上一帧完整骨架范围的 1.6 倍区域和球手 ROI 中重试，保留关节最多的结果（`skeleton_recovery`）。
   - 预测杆身裁剪：球杆置信度 < 0.3 时，按跟踪器预测的杆身位置重新裁剪推理一次，置信度提高才采用（`predicted_shaft_crops`）。
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
