# 高尔夫姿态工作流

## 人体裁剪框漂移的自动纠正与骨架异常中止（2026-11-06）

**问题**（`runs/workflows/videos/008_20261105_174213_259058`）：YOLOX 人物框在快速挥杆时漂移，YOLO26n 只在框内找人，于是丢失骨架。
- 第 147–202 帧：框滑到球手右半身和观众上，上半身被裁掉，只剩腿部关节。
- 第 232 帧起：球手的 YOLOX 分数在收杆时降到 0.39，检测器改选画面左侧的钟和广告牌，0 个关节。
- 缺失帧被插值后，P7 之后的位置都落在虚构的姿态上。

**纠正**：
- `workflow.py`（`detector_pose.YOLOXBodyPose`）：若框内骨架不完整（置信度 ≥ 0.25 的 COCO 关键点少于 12 个，或缺任一肩，或两腕都缺），依次在两个位置重试：上一帧完整骨架范围的 1.6 倍区域，以及自动球手 ROI。保留最好的骨架，并用它的范围引导下一帧的 YOLOX 选框。
- golfpose（`golfer_vitpose.GolferViTPose.recover_body`）：用同样的条件和重试位置，ViTPose 重新推理一次，关节更多才替换。
- 每帧的 `skeleton_usable`、`skeleton_recovery` 记录在 `detector_crops` 中。

**中止**（`golf_pose_3d.check_skeleton`，两个工作流共用，在 3D 提升之前执行）：
- 核心关节为根节点、双髋、双肩、双肘、双腕。少于 6 个，或两腕都缺，即判为“丢失帧”。
- 在 P1 到送杆结束的窗口内，若连续丢失超过 max(3 帧, 0.3 秒)，或丢失比例超过 15%，就打印 `ERROR:` 并抛出 `SkeletonQualityError`，列出丢失帧段。命令以错误退出，不再生成错误的 P1–P10。
- 无论是否通过，都会写入 `skeleton_diagnostics.json`（每帧核心关节数、丢失帧段、丢失帧上的裁剪来源）。
- `--allow-skeleton-gaps` 可以把中止改为警告，三个入口都支持。
- 阈值取 0.3 秒，是因为上杆顶点时手臂被头部遮挡，会短暂丢失几帧，这不应中止运行。

**验证（008）**：

| | 挥杆窗口（87–321）内丢失帧 | 最长连续丢失 | 检查结果 |
| --- | --- | --- | --- |
| 修复前（已保存的 2D） | 158（67%） | 90 帧（232–321） | 中止 |
| 修复后（重跑人体检测） | 13（5.5%），均在上杆顶点 161–174 及第 197 帧 | 5 帧 | 通过 |

修复后，322 帧中 59 帧改用自动球手 ROI，36 帧改用上一帧骨架范围。新增 `tests/test_skeleton_check.py`。

## 修复：球杆跟踪器误翻转导致 P7 提前（2026-11-05）

两个工作流在 w09 上的 P7 分别提前到第 29 帧（`workflow.py`）和第 30 帧（golfpose），此时杆头距球 264–327 像素，球杆尚未触球。
原因是新的刚性杆跟踪器（`golf_workflows/club_tracking.py`）把高置信度且方向正确的检测判为握把/杆颈互换。
例如 golfpose 第 31 帧置信度 0.86，握把距手 89 像素，杆颈距手 306 像素（位于球旁），却被交换。
击球附近杆身每帧转动上百度，恒角速度预测与真实方向相差很大，反向方向反而通过了门控。
`workflow.py` 还因初始状态取自一个方向错误的检测，连准备站位的帧也被整段翻转。
P7 依据杆头与球的距离修正，杆头被挪到手的位置后，击球帧就落到了更早的帧。

修改：
- 握把一定在手的一端。两端到手的距离差 ≥ 杆长 25%（`ORIENTED_FRACTION`）时，方向由几何确定，连续性不得反转；只有两端距离相近（透视缩短）时才允许按连续性判定互换。
- 置信度 ≥ 0.5（`REINIT_CONFIDENCE`）的检测始终接受；与预测不符时以该检测重新初始化状态。门控只剔除低置信度误检。
- 初始状态只从方向明确且置信度可靠的检测开始。
- 新增 `tests/test_club_tracking.py`：快速下杆的高置信度检测不被反转、端点颠倒由几何纠正、低置信度离群检测被剔除。

验证：以两次 w09 运行保存的 2D（撤销旧的错误交换）回放 `process`，并启用球检测：

| 工作流 | 修复前 P7（杆头距球） | 修复后 P1–P10 | 修复后 P7 杆头距球 |
| --- | --- | --- | --- |
| `workflow.py` | 29（327.6 px） | `8,13,19,23,25,28,31,32,33,39` | 25.3 px |
| golfpose_workflow | 30（264.5 px） | `7,13,18,24,26,28,31,32,33,39` | 39.2 px |

P7 = 31 与人工确认的击球帧一致。修复后 golfpose 不再有交换帧；`workflow.py` 剩余 8 帧交换均为两端距离差明显、低置信度且端点标反的检测。

## 自动默认球手 ROI（2026-11-05）

两个视频工作流现在都会在 2D 推理之前先推导出球手（golfer）ROI，因此对于 GolfPose 检测器（detector）始终未触发的片段，
也不再需要 `--golfer-box`
（`golf_workflows/golfer_roi.py`，接入于 `golf_pose_3d.main`）。

1. 用 YOLO26n（`yolo26n-pose.pt`）读取在片段中均匀分布的 12 帧，
   并在每帧中取评分 >= 0.4 的最大人物。
2. 保留与其中位检测框 IoU >= 0.3 的检测框；至少需要 3 个。这样可以剔除
   最大检测结果是观众或根本没有检测到人的帧。
3. 取这些检测框的并集，并为球杆向外扩展：左右各扩展其高度的 0.15，上方扩展 0.25
   （P4/P10 时杆身高过头顶），下方扩展 0.05。再将其裁剪到帧范围内。

各工作流如何使用它（仅在其 GolfPose 检测器漏检时使用；只要有检测结果，始终以检测结果为准）：

| 工作流 | 它所备援的检测器 | 之前 | 现在 |
| --- | --- | --- | --- |
| golfpose_workflow | 用于 22 关节 ViTPose 裁剪框的单类 YOLOX 球手检测框 | 首次检测之前的漏检无法被察觉，且没有任何检测结果的片段会以 "supply --golfer-box" 报错失败 | 由 ROI 初始化裁剪框（`auto-golfer-roi`）；出现检测之后，与之前一样保留最近一次检测到的检测框 |
| `workflow.py` | 用于 YOLO26n 身体裁剪框的双类 YOLOX 人物检测框 | 使用整帧，由最大的人物胜出 | 使用 ROI 裁剪框（`auto-golfer-roi`）；仅在没有 ROI 时才使用整帧 |

`--golfer-box X1 Y1 X2 Y2` 现在在两个工作流中都可用（`workflow.py` 使用默认的
YOLO26n + `official-yolox-2cls`），并会覆盖自动 ROI。使用 `--no-auto-golfer-box`
可关闭自动 ROI。所选 ROI 会记录在 `<stem>_3d.json` 中：
`pose_backend.initial_golfer_box_xyxy`、`initial_roi_source`（`manual` 或 `auto-yolo26n`）以及
`initial_roi_details`（采样帧、检测到人物的帧和一致的帧，以及边距）。逐帧的裁剪框
来源也会出现在摘要中。

原因：

- **w09 在 golfpose_workflow 中失败。** 单类 YOLOX 在 `pro_refs/w09.mp4`（一段从后方拍摄的竖屏手机录像）
  的任何一帧上的评分最高只有 0.04。YOLO26n
  在每个采样帧中都以约 0.89 的评分找到了球手。
- **我为 w09 手动选定的检测框与这一流程给出的结果很接近。**
  手动检测框为 `120 330 470 1000`；自动 ROI 为 `100.5 307.0 487.7 991.2`
  （12/12 帧一致）。
- **`workflow.py` 的整帧回退（fallback）会选中最大的人物。** 在 w09 中，
  双类 YOLOX 在 46 帧中有 44 帧漏检了该人物。与球手尺寸相当的裁剪框还能让
  YOLO26n 在球手身上获得约 1.8 倍的像素。
- **开销：** 每个片段 1-5 s。

自动 ROI：w09 `[100.5, 307.0, 487.7, 991.2]`，w07 `[25.0, 240.4, 391.0, 949.8]`，
005 `[110.7, 84.6, 553.8, 876.6]`（10/12 帧一致），010 `[34.1, 204.6, 518.6, 931.7]`，
008 `[62.3, 160.9, 591.0, 1018.5]`。

验证（86 项测试通过；新增用例覆盖 ROI 并集、离群值剔除与裁剪、
自动 ROI 引导关节序列，以及 YOLOX 漏检时身体裁剪框使用
ROI）。全新运行 `pro_refs/w09.mp4` 的结果：

| 工作流 | 之前 | 现在（不使用 `--golfer-box`） |
| --- | --- | --- |
| golfpose_workflow | 不使用 `--golfer-box` 时失败；使用时为 `6,13,18,23,26,28,31,32,33,40` | `7,13,18,24,26,28,31,32,33,40`（`runs/golfpose_workflow/videos/w09_autoroi_20261105/`） |
| `workflow.py` | `8,12,18,23,26,28,31,32,34,40`（整帧身体裁剪框） | `8,13,19,23,25,28,31,32,33,40`，46 帧中有 44 帧使用 `auto-golfer-roi`（`runs/workflows/videos/w09_autoroi_20261105/`） |

`workflow.py` 的 P9 变化（从 34 变为 33）源于不同的 2D 输入。在第 33 帧，
从后方视角（down-the-line）看，前导臂（lead arm）被身体遮挡，两种裁剪框都把它的位置估错了。
在 28 fps 下，手臂在第 33 帧与第 34 帧之间越过水平位置。YOLOX
在每一帧都能检测到球手的片段不受影响。

## 结合球杆信息的 P1、P4 与 P2/P6（2026-11-04）

两个工作流（`workflow.py` 和 `python -m golfpose_workflow`）共用
`golf_workflows/swing_phases.py`，因此只要开启球杆精修（默认开启），
以下规则对两者都适用。在此改动之前，球杆只用于确定 P7（杆头（clubhead）
最接近其准备站位位置或球的时刻）和 P2/P6（杆身最接近水平的时刻，且信任
每一个 3D 提升得到的杆身）。现在，只有当杆身（shaft）和杆颈（hosel）都被检测到，
且两者中较弱标记点的置信度 >= 0.3 时，该球杆帧才算"可靠"
（`RELIABLE_CLUB_CONFIDENCE`；模糊的杆身评分通常约为 0.2）。

| 位置 | 之前的规则 | 现在的规则 |
| --- | --- | --- |
| P1 准备站位 | 双手开始移动（相对身体的手腕离开起始位置），再减去 0.1 s | 上杆顶点之前、杆头仍处于其准备站位位置（在其上杆偏移量的十分之一以内，且至少 0.1 个躯干长度）的最后一个可靠帧，加 1 帧，再减去 0.1 s。需要在 0.3 s 内有两个可靠的静止帧。相对上一个被接受的杆头检测结果，每帧跳变超过 0.6 个躯干长度的杆头检测会被忽略。否则回退到基于双手的规则 |
| P4 上杆顶点 | 手腕高度峰值 | 2D 杆身（握把到杆颈）停止向后旋转的帧，在手腕峰值前后 0.15 s 内搜索，使用投影长度可读（>= 0.4 个躯干长度）的可靠帧。转折必须位于窗口内，且两侧都至少有 3 度的旋转。否则回退到手腕峰值 |
| P2 / P6 | 3D 提升得到的杆身最接近水平的帧（P1-P3 / P5-P7） | 相同，但只用可靠球杆帧，且仅当该杆身与水平的偏差在 0.35 以内（约 20 度）时才采用。否则改用前导臂：从下垂（P1 / P7）到与地面平行（P3 / P5）的旋转进度达到 30% 处 |
| P8 平局裁决 | 在伸展程度几乎相同的帧中取杆身最接近水平的帧 | 相同，仅使用可靠球杆帧 |

`<stem>_3d.json` 会记录相关证据：
- `phase_detection`：`address_source`（`club_rest` 或 `hand_motion_onset`）、`hand_address`、`club_rest_end`、`top_source`（`club_reversal` 或 `wrist_peak`）、`wrist_top`、`club_top`。
- `checkpoint_selection`：`position_sources`（P2/P6 为 `observed_shaft` 或 `lead_arm_progress`）、`reliable_club_confidence`、`reliable_club_frames`。

原因：

- **P1：杆头最先移动且移动幅度最大。** 基于双手的规则会被准备阶段双手的缓慢漂移
  触发。在 005 中，双手从第 84 帧开始漂移，而杆头一直停在
  球上（杆颈位于像素 528, 795），直到约第 104 帧，因此 P1 从 84 移到 108。
  在 008（正面视角（face-on））中，杆头在约 93-97 帧离开球，而双手
  约从第 100 帧开始移动。旧的 P1（99）已经显示球杆在移动；现在为 87。
  在 w09 中，旧的 P1（第 8 帧）杆头已经在离开球；现在为 6，此时球杆仍静止。
  "上杆顶点之前的最后一次静止"会跳过预备摆动（waggle）。跳变过滤器剔除了 008 在第 90-104 帧
  错误的杆颈检测（400 px 的跳变，置信度 0.27-0.44）。
- **P4：HackMotion 的上杆顶点是球杆改变方向的位置。** 双手到达峰值后，球杆
  仍可能继续蓄力。实际上，在上杆顶点处 2D 球杆很少可读。
  005、w07 以及 w09 的 YOLO26n 运行在那里都没有可靠检测。
  在 010 中握把和杆颈互换（130-180 度翻转），而在 008 中检测
  在第 154 帧中断，此时球杆仍在转动。因此通常结果仍然是手腕峰值。
  只有在两侧都观察到转向时，球杆才会覆盖它；在本次
  验证中只发生了一次（使用关节模型的 w09，与手腕峰值为同一帧）。
- **P2/P6：在推断出的杆身上做"杆身平行"判定只是猜测。** 在 008 中，
  整个起杆（takeaway）过程（第 100-128 帧）都看不到球杆。旧的 P2
  （100）和一个未经过滤的门限（89）都落在了错误的杆身上。要求
  观测到接近水平的杆身，可以避免挑出"最不差"的帧。
  回退使用前导臂而不是双手高度：3D 提升后的前导手腕在准备站位时就可能
  已经与骨盆齐平（w09：0.02 个躯干长度）。30% 这一比例
  来自 005（观测杆身得到的 P2（117）位于手臂旋转进度的 25% 处），
  以及 008 中复查过的帧（杆身约在第 119 帧平行，对应 28%）。

验证（81 项测试通过；新增用例覆盖球杆转向、带有误检的
球杆静止，以及 P2 回退）。回放复用已保存的 2D 结果，完整执行
`process`（3D 提升、精修与提取；不使用球模型，因此 P7 可能相差一帧）：

| 运行 | 之前（P1-P10） | 之后（P1-P10） | 来源 |
| --- | --- | --- | --- |
| 005 golfpose_workflow | `84,133,150,177,198,214,226,242,253,306` | `108,117,150,177,198,214,226,242,253,306` | 球杆静止；P2/P6 观测杆身 |
| 010 golfpose_workflow（正面视角） | `60,72,108,139,155,165,167,175,180,208` | `56,80,108,139,155,165,167,175,180,208` | 球杆静止；P2 前导臂，P6 观测杆身 |
| 008 `workflow.py`（正面视角） | `99,100,146,153,165,183,210,223,228,236` | `87,120,146,153,164,185,210,223,228,258` | 球杆静止；P2 前导臂，P6 观测杆身 |
| w07 `workflow.py` | `9,15,22,27,30,32,34,35,36,44` | `6,15,22,27,30,31,34,35,36,44` | 球杆静止；P2 观测杆身，P6 前导臂 |
| w09 `workflow.py` | `8,9,18,23,26,28,31,32,34,40` | `8,13,19,23,26,28,30,32,33,40` | 双手启动；P2/P6 前导臂 |
| w09 golfpose_workflow（`--golfer-box`） | `8,15,18,23,26,29,31,32,33,40` | `6,13,18,23,26,28,31,32,33,40` | 球杆静止，上杆顶点来自球杆转向；P2/P6 前导臂 |
目视检查：
- **w09：** 第 6 帧的 P1 球杆静止（第 8 帧时球杆已在离开球）。第 13 帧的 P2 双手位于髋部高度，杆身朝相机方向透视缩短（与目标线平行）；第 9 帧过早。
- **010：** 杆身约在第 80-84 帧处于水平；第 72 帧时仍指向下方。
- **008：** P2（120）位于复查过的 116-122 范围内。

全新端到端运行 `pro_refs/w09.mp4`（使用球）：

| 工作流 | P1-P10 | 来源 | 输出 |
| --- | --- | --- | --- |
| `workflow.py`（YOLO26n + 球杆 ViTPose，17+2） | `8,12,18,23,26,28,31,32,34,40` | 双手启动（静止时球杆不可靠）；上杆顶点来自手腕峰值；P2/P6 前导臂 | `runs/workflows/videos/w09_club_20261104/` |
| golfpose_workflow（`--golfer-box 120 330 470 1000`） | `6,13,18,23,26,28,31,32,33,40` | 球杆静止；上杆顶点来自球杆转向（与手腕峰值一致）；P2/P6 前导臂 | `runs/golfpose_workflow/videos/w09_club_20261104/` |

尚存的局限：
- 关节模型在 w09 第 33 帧把前导手腕的位置估错了，因此该工作流的 P9 为 33 而不是 34。
- 在 `workflow.py` 中，w09 上杆（backswing）阶段的球杆 ViTPose 只有 3 个可靠球杆帧（共 16 帧），因此那里的 P1 回退到双手启动。
- 之前构建的参考（reference）和评分（score）保留旧的 P1/P2/P4/P6 帧；需重新构建它们才能做同口径比较。

## 基于手臂的 P8/P9/P10 识别（2026-11-04，w09）

这是对下面那次更新的细化（同一文件，`find_checkpoint_frames`）。在 `pro_refs/w09.mp4`（28 fps）中，P7-P9 原为
第 31/32/33 帧，P10 原为第 45 帧。P9 时双手仍处于髋部到胸部之间的高度，
手臂低于与地面平行的位置。P10 出现在球手已经放下球杆之后。

| 位置 | 之前的规则 | 现在的规则 |
| --- | --- | --- |
| P3 | P4 之前前导臂与水平偏差在 0.25 以内的第一帧 | P4 之前前导手腕向上穿过前导肩高度的时刻；在穿越点前后的两帧中，取更接近水平的那一帧。否则回退到 0.25 区间 |
| P5 | P4 之后前导臂第一次与地面平行（0.25 区间） | P4 之后前导手腕向下穿过肩高度的时刻，采用相同的穿越规则 |
| P8 | P7 与 P9 之间杆身最接近水平的帧 | 双臂完全伸展：P7 与 P9 之间平均肘角（肩-肘-手腕，双臂）最大的帧。只计入双手已从击球时的高度上升到 P9 高度四分之一的帧。与最佳值相差 5 度以内的帧视为平局；由杆身最接近水平者裁决 |
| P9 | P7 之后前导臂第一次与地面平行（0.25 区间） | 双臂上升：P7 之后前导手腕向上穿过肩高度的时刻，采用相同的穿越规则。回退规则不变 |
| P10 | P9 之后前导手腕最接近头部且手腕高于肩部的帧 | 相同的判定，但限定在收杆阶段：P9 之后双手最高点附近、始终保持在其 5% 以内（相对于击球后的上升幅度）的帧 |

原因：

- **0.25 区间会接纳远低于平行位置的手臂。** 0.25 的竖直分量
  约为 14 度。在低帧率下，进入该区间的第一帧
  双手仍可能低于肩部。在 w09 中，第 33 帧手臂为 -0.22（低于），
  第 34 帧为 +0.16（高于），因此 P9 现在为 34。同样的判定也应用于与 P9"镜像"对应的 P3
  和 P5。在 005 中，P3 从 141（手臂约低于水平 14 度）
  移到 150（水平）。
- **P8 关乎手臂，而不只是杆身。** 击球后，杆身是画面中最模糊的
  部分（005 在第 236-246 帧杆颈置信度约为 0.2）。仅靠伸展程度
  也不够：击球时手臂同样伸直（005：第 226-231 帧为 162-165
  度），与释放时（第 242 帧为 165）一样。"双手已开始上升"这一门限
  可以将两者区分开。
- **P10 选中了球杆正在放下的帧。** 当双手下落到面部前方时，
  手腕离头部比真正收杆时更近。在 w09 中，最接近的
  帧为 45（双手位于胸部）；在 w07 中为 52，即片段结尾。收杆现在
  与双手的最高点绑定。

验证（78 项测试通过；新增合成用例，覆盖 P8 伸展、P9 穿越以及 P10
在放下球杆之前）：

| 运行 | 之前（P1-P10） | 之后（P1-P10） |
| --- | --- | --- |
| w09，`workflow.py` 全新运行 | `8,9,17,23,25,26,31,32,33,45` | `8,9,18,23,26,28,31,32,34,40` |
| 005，golfpose_workflow 回放 | `84,133,141,177,195,214,226,248,250,346` | `84,133,150,177,198,214,226,242,253,306` |
| 010，golfpose_workflow 回放 | `60,72,103,139,153,165,167,175,178,219` | `60,72,108,139,155,165,167,175,180,208` |
| w07，`workflow.py` 已保存的 3D | `9,15,20,27,29,32,34,35,36,52` | `9,15,22,27,30,32,34,35,36,44` |

目视检查：

- w09 的 P8（32）显示双臂在刚越过球处伸直，P9（34）显示双手位于肩部高度并
  正在上升，P10（40）显示双手高举、球杆位于头后。从第 41 帧起球杆被放下。
- 005 的 P8（242）显示手臂在髋部高度伸展，P9（253）显示双手位于胸部高度且球杆向上。

输出：`runs/workflows/videos/w09_p8p9_20261104/`。用旧规则构建的参考和评分
保留旧的 P3/P5/P8/P9/P10 帧；需重新构建它们才能做同口径比较。

## P 位置提取与挥杆窗口更新（2026-11-04）

参考：[HackMotion, Golf Swing Positions: the P Classification System](https://hackmotion.com/golf-swing-positions/#Understanding_Golf_Swing_Positions_in_the_P_Classification_System)。
这些规则由 `workflow.py` 和 `python -m golfpose_workflow` 共用
（`golf_workflows/swing_phases.py`）。

| 位置 | HackMotion 定义 | 之前的规则 | 现在的规则 |
| --- | --- | --- | --- |
| P1 | 挥杆开始前的准备站位 | 相对身体的双手运动开始时刻减去 0.1 s | 不变 |
| P2 | 起杆结束，球杆与地面平行 | P1 与 P3 之间杆身最接近水平的帧 | 不变 |
| P3 | 前导臂与地面平行 | P4 之前前导臂第一次与地面平行 | 不变 |
| P4 | 上杆的最高点 | 经交叉印证的手腕高度峰值 | 不变 |
| P5 | 下杆初段，杆身到达平行位置之前 | P4 之后杆身与水平偏差在 0.35 以内且手腕高于髋部的第一帧 | P4 之后前导臂第一次与地面平行（P3 的镜像） |
| P6 | 击球前杆身与地面平行 | P5 与 P7 之间杆身最接近水平的帧 | 不变（现在位于修正后的 P5 之后） |
| P7 | 击球 | 手腕回位，并结合球杆/球进行精修 | 不变 |
| P8 | 释放：击球后杆身平行，手臂伸展 | P7 与 P9 之间杆身最接近水平的帧 | 不变 |
| P9 | 前导臂在前导侧与地面平行；"P3 的反面" | P7 之后任意位置前导臂最接近水平的帧 | P7 之后前导臂第一次与地面平行；回退：双手最高点之前最接近水平的帧 |
| P10 | 挥杆结束：稳定、面向目标、双手举到前导侧耳朵附近 | P9 之后前导手腕最接近头部、且位于头部前导侧（3D 点积）的帧 | P9 之后、前导手腕高于前导肩时前导手腕最接近头部的帧 |

挥杆窗口终点：之前，窗口在首次出现 0.25 s 双手抬起且静止之后 0.15 s 关闭。
现在，窗口从首次静止保持开始一直保持打开，直到抬起的
双手重新落下（低于击球高度与上杆顶点高度的中点并持续
0.1 s），或直到片段结束。`<stem>_3d.json` 中的 `phase_detection` 记录了
`finish_hold_start`、`hands_lowered_at` 和 `finish_reason`
（`raised_hands_lowered`、`clip_end_hands_still_raised` 或 `clip_end_no_sustained_hold`）。

原因：

- **005（golfpose_workflow）结束得太早。** 第一次静止时刻（约第 275 帧）
  是双手停止上升、开始绕到头部后方的停顿。
  窗口在第 285 帧关闭。保持住的收杆（前导手腕最接近头部，几乎
  静止）约在第 320-360 帧，而双手直到约第 405 帧才落下。
  P10 只能落在 P9 之后一帧（253 之后是 254）。
- **P10 的前导侧判定排除了收杆。** 3D 点积包含了
  高度：当双手高于头部而前导肩低于头部时，符号
  会翻转。即使只看水平方向，当双手绕到头部后方时，符号也会在零附近
  徘徊。"双手抬起"（前导手腕高于前导肩）与 HackMotion 的
  描述一致，而且很稳定。
- **P5 的杆身判定在上杆顶点就被触发。** HackMotion 的 P4 中，全挥杆的上杆顶点处球杆
  与地面平行，因此"P4 之后杆身接近水平"
  会立即满足。010 的 P4/P5/P6 = 139/140/141。P5 是杆身平行（P6）之前的过渡
  检查点，因此改用前导臂（P3 的镜像）。
- **P9 可能匹配到后来的一次经过。** 窗口变长后，"击球后任意位置前导臂最接近
  水平"可能选中球杆放下时手臂再次经过平行位置的
  帧。HackMotion 将 P9 定义为 P3 的反面，因此它是
  击球后第一次平行。

`workflow.py` 现在使用 YOLO26n 身体姿态（pose）配合 17+2 提升模型（lifter），与关节工作流一致。
其默认 `--lifter` 为 `17+2`（这也是独立的
`golf_pose_3d` 和 `standard_frames` 模块的默认值）。仅用于球杆的 ViTPose 仍然
预测五个标记点；heel/toe_down/toe_up 及其置信度会在 3D 提升之前被丢弃
（`dropped_2d_landmarks`），原因见下方的 17+2
一节。`--lifter 17+5` 保留全部五个；旧的"club ViTPose requires 17+5"报错
已被移除。

验证（77 项测试通过，包括新增的窗口/P5/P9/P10 用例）：

| 运行 | 之前 | 之后 |
| --- | --- | --- |
| 005，golfpose_workflow（已保存的 2D 经新的阶段划分、17+2 3D 提升与提取回放） | 窗口 84-285；P1-P10 `84,133,141,177,188,214,226,248,253,254` | 窗口 84-412（`raised_hands_lowered`）；`84,133,141,177,195,214,226,248,250,346` |
| 010，golfpose_workflow（回放，正面视角） | 窗口 60-215；`60,72,103,139,140,141,167,175,180,197` | 窗口 60-251（片段结束，双手仍抬起）；`60,72,103,139,153,165,167,175,178,219` |
| w07，`workflow.py` 全新 YOLO26n + 球杆 ViTPose + 17+2 | 未运行 | 窗口 9-52；`9,15,20,27,29,32,34,35,36,52`；19 关节有限值导出 |

使用 17+2 的 `workflow.py references` 从 30 个有限值的
相机姿态生成了 10/10 个参考（8 个球杆叠加图）。`workflow.py score` 评估了 140/140 个参数
（92.0/100，这是相似度评分）。输出：`runs/workflows/videos/w07_17plus2_20261104/`、
`runs/workflows/references_17plus2_20261104/`、`runs/workflows/scores/w07_17plus2_20261104/`。
目视检查：005 的 P10 显示面向目标的保持收杆，010 的 P5 显示下杆初段前导臂
与地面平行。尚存的局限：在 005 中，击球后球杆模糊
（第 236-246 帧杆颈置信度约为 0.2），因此 P8（248）紧挨着 P9（250）。
回放复用已保存的 2D 推理结果；已发布的 `runs/golfpose_workflow/videos/005/`
是用旧规则生成的。重新运行即可刷新：
`.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/005.mp4`。

## 关节 ViTPose 工作流改为 17+2 3D 提升（2026-11-04）

`python -m golfpose_workflow` 仍然只运行一个 2D 模型
`golfpose_golfer_ViTPose_huge.pth`，但现在使用 MixSTE2 17+2
（`golfpose_17+2_33.6_30.5_59.5.bin`）进行 3D 提升。保留 17 个身体关节以及 `shaft` 和
`hosel`；`heel`、`toe_down` 和 `toe_up` 及其置信度在
3D 提升之前被丢弃，并在 `<stem>_3d.json` 中以 `dropped_2d_landmarks` 列出。
已保存的视频/参考数组为 19 关节。`golf_pose_3d` 和 `standard_frames`
现在接受 `--pose-backend golfpose-golfer-vitpose-huge --lifter 17+2`；17+5
在这两处仍然可用。

`010.mp4` 上的证据：三个杆头点的中位置信度最低
（0.58-0.63），且与杆颈距离的第 95 百分位约为 250 px，而
一号木杆头宽度仅约 25 px；球手裁剪框中约 13 px 对应一个热图（heatmap）单元。
阶段划分、球杆精修和评分只使用杆身和杆颈。GolfPose 的
双类 YOLOX 球杆检测框 + 仅用于球杆的 ViTPose 在十帧上做了测试，
在 3/10 帧中找到了球杆（评分 0.01-0.08，其中一个是错误物体）；其球手检测框
回退会使杆身/杆颈坍缩在一起。对比：`.tmp/club_compare_010/`。

验证：74 项测试通过。完整的 `010.mp4` 运行
（`runs/golfpose_workflow/videos/010_17plus2_20261104/`）保持了完全相同的杆身/杆颈
2D 坐标，导出的 P1-P10 为
`60, 72, 103, 139, 140, 141, 167, 175, 180, 197`，且全部十个位置都有完整的
球杆证据。P4-P6 仍落在连续的上杆顶点帧上；这来自 P5/P6 的阶段
规则，而非被丢弃的点（已由上面的 P 位置更新修复）。17+2 参考
（`runs/golfpose_workflow/references/17plus2_20261104/`）生成了 10/10 个参考
和球杆叠加图；评分评估了 140/140 个参数（89.2/100，这是相似度
评分，并非准确度度量）。

## 显式 ROI 初始化与根节点诊断（2026-11-04）

`pro_refs/010.mp4` 包含 252 个已解码帧，分辨率为 592x1280。单类 YOLOX
检测器在任何一帧上都没有返回达到 0.25 的检测框。此前，视频推理
会对每一帧都跳过 ViTPose，最终抛出具有误导性的消息
`No reliable GolfPose root found`。

关节视频路线现在接受以原始帧像素为单位的 `--golfer-box X1 Y1 X2 Y2`。
它会显式初始化姿态 ROI；YOLOX 仍在每一帧上运行，
并且在检测可用时可以接管。手动 ROI 的使用会被记录
为 `manual-golfer-roi`，不会伪造检测器评分。形状、坐标为有限值、
尺寸为正以及位于已解码帧范围内都会经过校验。视频不存在
整帧姿态回退，也不会放宽根节点阈值。

```powershell
.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/010.mp4 --golfer-box 140 220 500 900 --view face_on
```

失败消息现在会区分缺少 ROI/姿态推理次数为零，与根节点确实
置信度低或位于帧外这两种情况。失败的运行会保存
`pose_inference_diagnostics.json` 和 `.npz`，其中包含已解码/已推理帧数、
原始坐标/有效性/置信度、最大根节点置信度以及逐帧裁剪框。
帧提取和评分继续沿用其现有规则。

验证：72 项测试通过。在实际的第 0、126 和 200 帧上，所提供的 ROI
得到了被观测到的根关节，置信度分别约为 0.918、0.967 和 0.911。
示例诊断：`runs/diagnose_010_20261104/manual_roi_samples.json`。

完整的 CLI 运行完成了全部 252 帧，每一帧都观测到了根关节
（置信度范围约 0.854-0.971）。所有裁剪框都被显式记录为
`manual-golfer-roi`，没有伪造任何 YOLOX 检测。自动提取
在索引
`60, 72, 103, 139, 140, 141, 167, 175, 180, 196` 处生成了十个有限值的 22 关节姿态，并将完成的视频
运行结果发布到 `runs/golfpose_workflow/videos/010_roi_20261104/`。
验证报告：`runs/diagnose_010_20261104/validation.json`。这验证的是
执行流程和数据契约，而不是关节或阶段相对真值的准确度。

## YOLOX 检测器路由修订（2026-11-03）

| 工作流 | 检测器权重 | 姿态输入 |
| --- | --- | --- |
| `workflow.py`（YOLO26n） | `golfpose_checkpoints/golfpose_detector_2cls_yolox_s.pth` | 人物裁剪框送入 YOLO26n 人体姿态模型；关联的球杆裁剪框送入仅球杆 ViTPose |
| `python -m golfpose_workflow` | `golfpose_checkpoints/golfpose_detector_1cls_yolox_s.pth` | 球手裁剪框送入单一的原生 22 关节球手 ViTPose 模型 |

保存的权重元数据确认：双类 YOLOX 的类别为 `person, club`，
单类 YOLOX 的类别为 `golfer`。两者均可按与其匹配的分类头尺寸严格加载。
原生适配器遵循已保存的配置：BGR 输入、640x640 缩放并在左上对齐填充、
类别得分乘以目标性（objectness）得分，以及 0.65 的按类别 NMS。
实现已对照 MMDetection 官方的
[YOLOX 主干网络（backbone）](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/csp_darknet.py)、
[颈部（neck）](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/yolox_pafpn.py)
和[检测头（head）](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/yolox_head.py)进行核对。

YOLO26n 工作流在参考图与视频中（包括独立模块）默认使用
`--club-crop-backend official-yolox-2cls`。双类检测器只需运行一次，即可同时提供人物与球杆候选框。
所选人物框的得分必须
>= 0.25；将人物框扩展 10% 后的裁剪图送入 YOLO26n，并把坐标
变换回原始帧。若没有被接受的人物框，YOLO26n
会自行在全帧上进行人体检测，并在日志中记录该回退（fallback）。球杆候选框
会与球手/双手进行关联，以排除无关的横幅等物体。若没有剩余候选框，
仅球杆 ViTPose 会基于球手检测框进行细化。默认的球杆分割已被
双类检测器取代；显式指定的旧版裁剪选项仍可使用。
YOLO 原始的 COCO 置信度值现在会随还原后的坐标一并保存，
供现有的基于置信度的人体参考排序使用。

联合 ViTPose 工作流使用单类检测器，得分阈值为 >= 0.25。它在
原图上对检测到的球手 ROI 进行解码，沿用现有的 1.25 倍 UDP
扩展。视频中出现检测缺口时，保留最后一次检测到的 ROI，并在日志中记为
`tracked-detector-roi`；若尚无初始 ROI，则相应关节视为未观测。视频不进行
全帧回退。仅对静态参考图而言，检测器漏检时可使用整个
面板，并记录 `reference-panel-fallback`。原始姿态置信度/有效性
被保留；跟踪得到的检测框与检测器实际观测结果会被区分。

在这些输入上，检测并非始终可靠：阈值为 0.25 时，单类模型
在 30 个参考面板中检出 1 个，在 53 帧 `w07` 中检出 7 帧；双类模型的人物输出
在面板中检出 1/30，在帧中检出 3/53。这些是检测框可用数量的统计，而非准确率评分。
在此样本上，上述回退是必要的。详细信息保存在
`runs/validation_detector_routing_20261103/detector_audit.json`。

参考 JSON 保存所选裁剪框的详细信息。视频 JSON 保存 `detector_crops`，
包括从零开始的帧索引、所选检测框/得分及回退来源。
MixSTE2 17+5、整体视角选择、叠加图、阶段/提取条件以及
14 项指标评分均保持现有实现。检测器的改动需要重新推理才能生效；
缓存的数值运行结果保留其旧元数据。

验证：全部 67 项测试通过。两条流水线均生成了数值有限的 P1 参考，并
处理了 `w07.mp4` 的全部 53 帧，同时保存了检测器/裁剪诊断信息。YOLO26n
自动导出了十个数值有限的 22 关节姿态。裁剪后的联合 ViTPose 运行
识别出准备站位/上杆顶点/击球分别位于第 9/27/34 帧，但 P5 未通过保持不变的杆身竖直
分量阈值：其符合条件的最小值约为 0.357，高于 0.35。
P6 依赖于 P5，因此该次运行未发布完整的十帧结果包。
其 53 个姿态裁剪框中，7 个来自检测，46 个来自保留的检测器 ROI，没有进行
全帧视频姿态推理。这一局限被保留以供审查，而不是
用任意帧填补缺失的阶段。已验证已发布的最新结果指针
未被改动。报告：
`runs/validation_detector_routing_20261103/pipeline_validation.json`。

## 独立的联合 GolfPose 工作流（2026-11-03）

另一个启动器、代码目录和输出集合可独立于
YOLO26n 工作流使用。参见 [golfpose_workflow/README.md](golfpose_workflow/README.md)。

```powershell
.\.venv\Scripts\python.exe -m golfpose_workflow --reference
.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/w07.mp4
.\.venv\Scripts\python.exe -m golfpose_workflow --score
```

每条参考/视频命令都会重新推理原生 22 点二维坐标，人体与球杆姿态
仅使用 `golfpose_golfer_ViTPose_huge.pth`，随后用
MixSTE2 17+5 进行 3D 提升。结果与最新结果指针位于 `runs/golfpose_workflow/` 下。
参考选择、球杆叠加图、阶段/提取条件、十帧
验证以及 14 项指标评分的实现均与原
工作流共用。自动视频提取会导出十个源帧、十张三维图像、
对应的置信度/坐标数组以及原始帧索引。经人工审核的索引
仍可通过 `--positions-json` 提供；自动结果不充分时，不会
发布完整的视频指针。评分读取本工作流自身的最新运行结果。

`workflow.py` 仍默认使用 YOLO26n 进行人体推理。其输出集合与
最新结果指针保持独立。新启动器不提供导入缓存推理结果的模式。

检测器修订前的验证：60 项测试通过；重新推理生成了全部十个参考，并
从完整的 53 帧 `w07.mp4` 视频中自动导出了十帧。所有
选中的 22 关节三维姿态均为有限值。评分使用共享规则评估了全部 140 个参数。
已验证原有的最新结果指针未被改动。
为现有的可选击球时刻（球）检测安装了缺失的 Ultralytics CLIP 分词器，
在 `requirements.txt` 中固定了其源码提交，并验证 `pip check` 无问题。
执行详情：`runs/golfpose_workflow/validation.json`。

## 当前默认：YOLO26n 人体姿态（2026-11-03）

参考生成与视频推理现在默认使用 `--pose-backend yolo26n-pose`。
默认不再加载联合 `golfpose_golfer_ViTPose_huge.pth` 权重。
YOLO26n 实现已对照仓库 HEAD
`40f50ebeef4e29139275db968c250938204e238c`（位于
https://github.com/StevenZhou795/golf-posture）进行核对。

- 人体：`yolo26n-pose.pt`，17 个 COCO 关节映射到 17 个 GolfPose 人体通道；
  根节点/脊柱/胸部/颈部/头部沿用现有的 COCO 转换规则。
- 球杆：`golfpose_checkpoints/golfpose_club_ViTPose_huge.pth`，五个球杆标记点。
  在参考图与视频中由双类 YOLOX 提供人物/球杆检测框；由仅球杆
  ViTPose 提供五个标记点坐标。当没有可关联的球杆检测结果时，
  回退为基于球手检测框的细化。
- 3D 提升：MixSTE2 `17+5`。整体视角人体选择、球杆供体选择、
  仅杆身回退、预览、阶段识别、细化与导出均保留。

公共启动器与两个独立推理模块默认使用 YOLO26n 人体姿态
和仅球杆 ViTPose。其他姿态后端仍可作为显式选项使用。
需要重新推理：`--from-run` 与参考的 `--rebuild` 会复用已保存的姿态。
此次改动不会重新生成已有的完整结果与最新结果指针。

检测器修订前的验证：全部 56 项测试通过。一张真实的 P1 图为全部
三个视角生成了有限的三维姿态，并生成了有限的选定参考。一个十帧采样视频完成了
推理、3D 提升、细化、渲染与导出，选定姿态为有限值，形状为 `(10, 22, 3)`。
该测试使用了显式指定的测试索引，因此不能验证自动
阶段识别的准确性。结果：`runs/validation_yolo_body_20261103/validation.json`。

```powershell
.\.venv\Scripts\python.exe workflow.py references
.\.venv\Scripts\python.exe workflow.py video pro_refs/w07.mp4
```

下文关于联合模型的章节描述的是早期实验，仅在
显式指定 `--pose-backend golfpose-golfer-vitpose-huge` 时适用。

## 历史方案：联合球手与球杆推理（2026-11-03）

早期的联合后端实验加载一个二维姿态权重：
`golfpose_checkpoints/golfpose_golfer_ViTPose_huge.pth`。其保存的配置
与严格参数加载确认共有 **22 个 GolfPose 原生关节**：17 个人体关节，
其后依次为 `shaft`、`hosel`、`heel`、`toe_down` 和 `toe_up`。

```powershell
.\.venv\Scripts\python.exe workflow.py references
.\.venv\Scripts\python.exe workflow.py video pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4
```

该实验性后端为 `golfpose-golfer-vitpose-huge`。它在
整个单人球手参考面板或视频帧上运行一个姿态模型，沿用现有的
192x256 UDP 裁剪、DARK 热图解码以及原图/镜像平均。
人体通道使用 GolfPose 原生的左右交换；球杆通道不
交换。无需 COCO 关节转换，也无需合成人体关节。此默认二维路径
不加载 YOLO 人物检测器、球杆分割、COCO 人体权重或仅球杆
ViTPose 权重。可选的视频球检测模型仍然可用，
但仅用于现有的阶段时刻修正。

原始的 22 点坐标与置信度会被保留。仅当某点得分至少为 0.10，
且坐标为有限值并位于原始图像内时，该点才有效。缺失的输入在
MixSTE2 17+5 3D 提升前由现有例程填补。归一化使用可靠的预测人体
关节的边界。参考对齐、人体/球杆叠加图渲染、视频
阶段选择、球杆细化、十帧导出以及评分均保留其
现有算法与文件约定。

参考结果还会在 `joints.npz` 中额外保存 `<view>_2d_confidence`（22 个值）；
视频数组保存 `keypoints_2d_confidence`（帧数 x 22）。姿态/球杆后端
元数据指向同一个已加载的模型实例。当时必须使用 `--lifter 17+5`
（自 2026-11-04 起也接受 17+2）；
选择联合后端时，旧版的 `--club-pose-backend` 与 `--club-crop-backend` 参数
会被忽略。输入中必须只有一名球手；在宽画面中
较小的球手可能会因固定的网络输入分辨率而丢失细节。

若要保留早期的双姿态模型路线，请显式选择
`--pose-backend vitpose-huge --club-pose-backend vitpose-huge`；参考图使用
`--club-crop-backend golfer-box`，视频则可使用 `segmentation` 或
`official-yolox-2cls` 裁剪。DEKR 仍处于移除状态。`--from-run` 会复用旧模型的
结果，因此使用新权重生成输出时请省略该参数。此次实现更新不会改动已有的
完整运行结果与最新结果指针。

2026-11-03 的验证：该权重以 22 个输出通道严格加载，
并声明使用 `golfswing_golfer` 数据集。全部 41 项测试通过，
包括原生通道/翻转顺序、画面外点的有效性、参考与视频共用的
路由，以及 GolfPose 手腕索引。一张真实的 P1 图为全部三个视角生成了有限的
三维结果以及一个融合参考。一个十帧视频测试
完成了推理、MixSTE2 3D 提升、球杆细化、渲染与提取，
得到的 `identified_frames.npz:keypoints_3d` 为有限值，形状为 `(10, 22, 3)`。
该视频测试使用了采样帧与显式测试索引，不能
衡量自动阶段选择的准确性。产出文件与模型元数据
保存在 `runs/validation_golfer_vitpose_20261103/validation.json`。

## Python 环境与依赖

环境配置日志（2026-11-02）：系统 Python 3.12 环境显示已安装
`torch==2.6.0+cpu`，但导入时因
`torch/lib/caffe2_nvrtc.dll` 报 WinError 126 而失败。其包目录中残留了 CUDA
库文件。为避免这一混杂安装，项目使用单独的 `.venv`。
系统环境未作改动。

在 PowerShell 中从 `J:\Golf_Posture` 复现该环境配置：

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = "$PWD\.tmp"
$env:TMP = "$PWD\.tmp"
$env:PIP_CACHE_DIR = "$PWD\.tmp\pip-cache"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install --no-compile -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` 固定了工作流的各项依赖版本，包括 NumPy、SciPy、
OpenCV、Matplotlib、Ultralytics、timm、einops 和 local-attention。此处 PyTorch 与
torchvision 使用 CPU 版 wheel；当前的 ViTPose/MixSTE 推理代码在
CPU 上运行。本地推理实现不需要 MMEngine、MMPose 和 MMDetection。
MediaPipe 是可选的视频手腕回退方案，
参考生成不需要它。

已安装并验证的版本：

| 包 | 版本 |
| --- | --- |
| torch | 2.6.0+cpu |
| torchvision | 0.21.0+cpu |
| numpy | 2.1.1 |
| scipy | 1.14.1 |
| opencv-python | 4.11.0.86 |
| matplotlib | 3.11.2 |
| ultralytics | 8.4.173 |
| timm | 0.9.16 |
| einops | 0.8.2 |
| local-attention | 1.9.0 |

2026-11-02 的验证：`pip check` 未报告任何损坏的依赖；PyTorch
导入与张量转 NumPy 均通过；全部 35 项单元测试通过。真实的
`golfpose_club_ViTPose_huge.pth` 以五个输出通道加载，
MixSTE2 17+5 权重以 22 个关节加载。这些检查验证了包的
兼容性与权重加载；参考图像未重新生成。

请显式使用项目解释器，以避开已损坏的系统安装：

```powershell
New-Item -ItemType Directory -Force .tmp\ultralytics\Ultralytics, .tmp\matplotlib | Out-Null
$env:YOLO_CONFIG_DIR = "$PWD\.tmp\ultralytics"
$env:MPLCONFIGDIR = "$PWD\.tmp\matplotlib"
.\.venv\Scripts\python.exe workflow.py references
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

在 `J:\Golf_Posture` 中每个阶段只需一条命令：

```powershell
python workflow.py references
python workflow.py video pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4
python workflow.py score
```

评分使用最近一次成功完成的参考与视频工作流结果。
每条命令都接受 `--output-dir`。运行 `python workflow.py <stage> --help`
查看选项。`python -m golf_workflows` 是另一种启动方式。

## 推理思路与模型调用说明

项目采用“先识别二维关键点，再提升为三维姿态，最后按姿态指标评分”的流程。
人体推理使用 YOLO26n-pose，球杆标记点使用仅球杆的 ViTPose，MixSTE2 负责三维转换。
当前 `workflow.py` 的默认设置为 `yolo26n-pose`、仅球杆的 ViTPose 和 `17+5`。
视频开启球杆 refinement，CPU 线程数为 8；底层模块的独立命令可能有不同默认值。

### 默认调用哪些模型

| 模型 | 权重文件 | 输入与职责 | 调用阶段 |
| --- | --- | --- | --- |
| YOLO26n-pose | `yolo26n-pose.pt` | 17 个 COCO 人体关节，映射到 GolfPose 人体通道 | references、video |
| GolfPose club ViTPose-Huge | `golfpose_checkpoints/golfpose_club_ViTPose_huge.pth` | 五个球杆标记点 | references、video |
| GolfPose MixSTE2 17+5 | `golfpose_checkpoints/golfpose_17+5_39.2_32.3_62.8.bin` | 将二维人体与球杆关键点序列提升为三维坐标 | references、video |
| YOLOE 分割模型，可选 | `yoloe-26n-seg.pt` | 在文件存在时以 `golf ball` 类别检测球，辅助修正击球时刻 | 仅 video |

`--pose-backend yolo26n-pose` 使用 YOLO 人体关键点；五个球杆标记点仍由仅球杆的 ViTPose 提供。
使用 `--pose-backend yolo26n-pose --club-pose-backend segmentation --club-crop-backend segmentation --lifter 17+2` 时，改用
`golfpose_checkpoints/golfpose_17+2_33.6_30.5_59.5.bin`，输入为 17 个人体点和 2 个球杆点。
`--from-run` 复用已完成结果，不重新调用上述神经网络；`score` 也不调用这些模型。

### references：从参考图片到十个三维标准姿态

1. **拆分视角。** 每张 P1-P10 标准图包含正面、沿目标线和背面三个视角，
   先裁剪成单人图片。十张标准图正常对应 30 次单视角姿态处理。
2. **联合二维姿态。** 整个单人视角由 `golfpose_golfer_ViTPose_huge.pth` 一次推理，输出 22 个 GolfPose 原生关节；无需 COCO 转换或独立球杆模型。
3. **热图解码。** 使用原图与水平翻转输入，按人体左右对称关系交换通道后平均，再用 UDP/DARK 解码回原图坐标。
4. **组织 17+5 输入。** 保留全部原始坐标与置信度。低于 0.10 或位于画面外的点标记为无效，仅在提升模型输入中补值，再传给 MixSTE2 17+5。
5. **二维提升为三维。** 根据人体框将坐标缩放、平移到虚拟 1280×720 画布并归一化。
   单张图片没有真实时间序列，因此将同一帧重复补足 MixSTE 的 243 帧输入窗口，
   再进行原始与镜像输入的三维预测平均，得到以根关节为原点的单视角三维姿态。
   MixSTE 利用空间注意力学习人体与球杆的关联，利用时间注意力处理轨迹；
   对重复静态帧而言，不存在额外的真实运动信息。
6. **选择整体参考。** 将各视角姿态归一化并对齐，在 face_on 与 down_the_line 之间按 17 个人体关节平均置信度选择较高者，完整保留其对齐后的骨架，不再替换另一视角的空间分量。保存 `joints.npz:reference_3d` 并渲染 `reference_3d.jpg`。背面仅供检查。

这种单视角选择与对齐不是基于已标定相机的多视角三角测量，输出的长度与深度不能直接视为精确物理测量。
`references` 不进行挥杆起止时间识别，也不运行视频的球杆轨迹 refinement 或球检测。

### video：从视频到 P1-P10

视频帧使用 YOLO26n 人体关节和仅球杆的 ViTPose 标记点。默认的 refinement 在选择阶段时间窗口之前运行。
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

## 1. 构建十个参考姿态

`references` 读取 `pro_refs/Standard_10_frames` 中的十张 JPG 参考图（reference sheet），
裁剪出正面视角（face-on）、后方视角（down-the-line）和背后视角，运行身体与球杆推理，
用 MixSTE 对各视角进行 3D 提升，并在正面视角空间中对齐姿态。置信度较高的
正面视角或后方视角身体姿态将提供完整的数值参考。
背后视角仅保留用于检查。

`python workflow.py references` 和 `python -m golf_workflows.standard_frames`
默认均使用 `yolo26n-pose` 推理身体、使用仅球杆的 ViTPose 推理球杆，并采用 MixSTE2
17+5。映射得到的 17 个身体通道和五个球杆通道将用于 3D 提升和整体姿态选择。
如需显式选择可选的联合模型实验：

```powershell
python workflow.py references --pose-backend golfpose-golfer-vitpose-huge --lifter 17+5
```

输出：`runs/workflows/references/`，包含 `workflow.json` 和十个位置
文件夹。每个文件夹包含 `reference_3d.jpg`、`joints.npz`（数值坐标）、
`joints.json` 以及各相机的图像/姿态。全新推理还会写入 `run_summary.json`。
完成的条件是 P1-P10 每个位置都有一个有限值参考和一张可读图像。

每个参考文件夹还包含按 P1-P10 数字顺序排列的 PNG 拼图：
`reference_3d_preview.png`（十个身体）、
`reference_3d_club_overlay_preview.png`（十个球杆叠加图）以及
`reference_all_preview.png`（两组合并）。叠加图会标注来源视角和
推断点；缺失的叠加图使用明确的占位图。这些
预览在全新推理、缓存参考导入和基于已保存姿态的重建之后，由已保存的 JPG 生成；
刷新预览无需任何模型推理。

身体参考选择规则（2026-11-03）：仅比较具有有限值 3D 姿态的 `face_on` 和
`down_the_line` 视角。每个视角的评分使用**前 17 个身体关节的平均
置信度**。对于原生 22 关节检查点，
无效/缺失的身体关节计为零，有限的置信度被裁剪
到 [0, 1]。球杆置信度不影响身体选择。对于旧版 COCO
后端，使用 COCO 平均置信度乘以映射后身体的有效关节
比例。分数相同时优先选择 `face_on`。已知分数优先于缺失
分数；若旧的已保存运行中所有分数均缺失，则优先选择 `face_on`。

将所选视角完整的归一化、已对齐姿态复制到
`joints.npz:reference_3d`。不要用另一台相机的数据替换 x/y/z 分量或单个关节。
`rear` 仍可用于检查，但绝不参与选择。
`side_axis` 现在为 null，`reference_kept` 仅包含所选视角。
`joints.json:reference_body_selection` 记录所选视角、两个分数
以及所用规则。在所检查的 P9 运行中，正面视角得分 0.9250，而后方视角为 0.7799，
因此保留其完整骨架。球杆叠加图的来源视角
按下述规则独立选择；它不会改变身体
或数值评分参考。

全新参考生成和 `standard_frames --rebuild` 都会应用此
规则。重建使用已保存的 2D 置信度/有效性和 3D 姿态，同时重新生成
身体图像和叠加图，无需神经网络推理。已有的
已完成文件夹不会被自动重写；单独使用 `--from-run` 只会复制
已存储的结果，不会应用此修订。

验证：全部 49 项测试通过。对已保存 P9 文件夹的副本进行重建，
选择了 `face_on`；其数值参考与完整的
已对齐正面视角姿态完全一致，其叠加图保留了相同的身体。原始的各视角
3D 姿态和所选的正式运行结果均被保留。重建示例
位于 `runs/validation_body_selection_20261103/`。

球杆叠加图选择规则（2026-11-03）：**仅考虑 `face_on` 和
`down_the_line`**。`rear` 绝不作为叠加图来源。不存在固定的
按位置视角覆盖设置。候选视角必须五个球杆标记点（landmark）全部
有效（包括杆头标记），具有有限值的 2D/已对齐 3D 坐标和
置信度，并且杆身（shaft）与杆颈（hosel）之间至少相距 12 像素。这些完整的
五点候选始终优先于不完整的候选。

在符合条件的候选中选择得分较高者：
`min(confidence) + 0.25 * mean(clip(confidence, 0, 1)) - fallback_penalty`，
其中使用了裁剪框回退（fallback）时惩罚为 0.05，否则为零。
此处的置信度是按顺序排列的五点球杆置信度。分数完全相同时
优先选择 `face_on`。

**缺失关节回退：**仅当两个符合条件的视角都没有全部五个
可靠的 2D 球杆标记点时，某个视角才可以**仅凭杆身**获得资格。杆身
必须具有有效、有限值的 2D 观测，且置信度至少为 0.10。其余
四个标记点可以全部缺失、无效或位于画面之外。这些部分
候选按 `shaft_confidence - fallback_penalty` 排序，分数完全相同时
正面视角胜出。使用此回退时，12 像素的杆身/杆颈长度要求
不适用。仍要求全部五个球杆关节具有有限值的已对齐 MixSTE 几何，
以便能够渲染叠加图。如果存在完整视角，则不使用此
回退。如果既没有完整候选也没有杆身置信度达标的候选，
则不生成叠加图。

对于回退叠加图，将同一所选视角的五个已对齐 MixSTE
坐标一起复制；缺失的端点/杆头标记属于**推断几何**，
而不是恢复出的 2D 观测。原始 2D 有效性标志保持不变。
`reference_club_overlay` 记录 `shaft_only_fallback`、
`observed_club_landmarks` 和 `inferred_club_landmarks`；无法获得的观测
杆身长度/方向为 null。图像标题会标识推断点。
这适用于全新推理和基于已保存姿态的重建。输出仍为
`reference_3d_club_overlay.jpg` / NPZ 文件中的 `reference_3d_club_overlay`。

验证：全部 52 项测试通过，其中包括仅有杆身观测、
其余四个 2D 关节均缺失的情况。P8 和 P10 的重建副本均生成了叠加图：
P8 选择了正面视角（推断 `toe_down`），P10 选择了后方视角
（推断 `heel`、`toe_down`、`toe_up`）。示例保存在
`runs/validation_shaft_fallback_20261103/` 下；正式输出未被覆盖。

P10 叠加图刷新（2026-11-03）：随后选定的
`runs/workflows/references/` 文件夹中仍包含回退规则之前的 P10 结果。
其叠加图已根据已保存的对齐坐标刷新，选择了
`down_the_line`，其杆身置信度为 0.9449，而正面视角为 0.6468。全部五个
球杆坐标均从该来源视角一并复制；缺失的杆头标记
仍标注为推断。已验证数值身体参考保持不变，
之前的 P10 文件已备份至
`runs/workflow_snapshots/p10_overlay_before_refresh_*`。

将所选视角**完整的已对齐 MixSTE 球杆几何作为一个整体**复制，
并保持通道顺序：`shaft`、`hosel`、`heel`、`toe_down`、`toe_up`。
不要混用来自不同相机的端点或杆头标记。工作流
不会将杆身连接到手腕，也不会强制设定杆身/杆头尺寸。
此选择在参考生成期间执行，并记录在
`joints.json:reference_club_overlay` 中，包括符合条件的视角、所选视角、
分数和置信度。它可能通过复制该视角已对齐的
MixSTE 球杆坐标来写入 `reference_3d_club_overlay.jpg` 和
`joints.npz:reference_3d_club_overlay`。这只是一个带标注的可视化叠加图。
`joints.npz:reference_3d` 保持不变，仍是唯一的评分输入；
`joints.json:reference_club_overlay` 记录来源视角和置信度。
基础图像 `reference_3d.jpg` 仅包含身体，以免插补的或画面外的球杆点
被误认为是观测值。

如需复现论文中单独处理球杆的路线，请选择恢复的二类
GolfPose YOLOX-s 检测器作为球杆裁剪框来源。类别 `1` 提供
逐帧的球杆框，现有的仅球杆 ViTPose-H 模型提供
五个标记点。这是逐帧检测和姿态估计，而不是经过学习的
时序跟踪器：

```powershell
python workflow.py references --pose-backend vitpose-huge --club-crop-backend official-yolox-2cls
python workflow.py video <video.mp4> --pose-backend vitpose-huge --club-crop-backend official-yolox-2cls
```

显式的 `segmentation` 裁剪选项仍然可用。二类 YOLOX
提供默认的视频球杆裁剪框。参考裁剪框也默认使用
`official-yolox-2cls`。联合后端使用一类 YOLOX 球手裁剪框，
并采用上文所述的跟踪 ROI 和参考面板回退策略。

无需新推理即可复用已完成的参考：

```powershell
python workflow.py references --from-run runs/standard_frames_vitpose_huge_17plus5_20261027
```

## 2. 识别十个视频帧

### 视频/参考推理一致性审计（2026-11-03）

历史联合后端验证：参考面板和视频帧共用 `GolferViTPose.predict_frame`：同一个
`golfpose_golfer_ViTPose_huge.pth`、原生 22 通道关节顺序和翻转映射、
UDP/DARK 解码、置信度 >= 0.10、画面内有效性，以及基于
可靠身体关节边界的归一化。视频使用原生手腕关节 13/16，而不是 COCO
索引。这次较早的审计使用的是扩展系数为 1.25 的完整单球手图像。
上文的检测器修订现在在两条路径中都提供一类 YOLOX 裁剪框。

最新的球杆缺失规则现按视频帧逐帧应用：

- `complete`：所有球杆通道均被观测到，且杆身/杆颈间距至少
  为 12 像素。
- `shaft_fallback`：球杆不完整，但杆身被观测到且分数
  >= 0.10。将该帧的 MixSTE 球杆几何作为整体显示，并将缺失的
  端点/杆头标注为推断。绝不更改原始 2D 有效性标志。
- `none`：没有符合条件的球杆证据；隐藏该帧的 3D 球杆。

完整视频数组和十帧导出保留 `keypoints_2d_confidence`
（帧数 x 22）、`keypoints_2d_valid`、五个球杆置信度和 `club_evidence`。
每个所选帧的清单（manifest）记录其证据类别以及共享的
姿态后端元数据。没有证据数组的旧缓存运行会在导出时
根据已保存的坐标/有效性/置信度推导出该数组。源帧索引
和数值 3D 姿态不会因导出而改变。推断的 3D 球杆几何
属于可视化/模型输出，而不是新检测到的 2D 端点。

在识别时，应在整个片段中一致地使用单个观测到的运动标记点：
杆颈有足够有效样本时使用杆颈，否则在可用时使用杆身，
再否则使用基于手腕的证据。`phase_detection.club_track_source` 记录
该选择。现有的相对身体的手腕抬升/回落、收杆（finish）停留检查、
有界的球位校正以及按顺序的 P1-P10 选择仍然有效。
仅有杆身的帧不会成为用于几何拟合或
基于球杆的击球（impact）校正的有效端点对。可用的杆身本身并不能证明
所有十个挥杆阶段都能被识别。

参考中正面视角/后方视角的竞争不适用于单相机
视频：不要用一个无关但置信度更高的帧替换时序事件帧。
视频在整个 3D 提升过程中保留每帧完整的身体预测。
球杆轨迹仍需通过 `--track-club-path` 手动启用；显示杆身
回退不会自动启用轨迹。

验证：全部 56 项测试通过。将当前 `w07` 运行已保存的 22 关节
2D 观测重新经过自动识别、MixSTE2、精修、渲染
和十帧导出，得到索引 `[7, 15, 18, 27, 32, 33, 34, 35, 37, 39]`，
与其先前的选择一致。导出的 XYZ 形状为 `(10, 22, 3)`，完整
置信度形状为 `(10, 22)`。此次重放复用了已保存的 2D 推理；未更改任何最新
视频/参考指针。产物位于
`runs/validation_video_reference_parity_20261103/`。

`video` 运行 YOLO26n 身体推理加仅球杆的 ViTPose、阶段识别、
MixSTE2 17+5 3D 提升以及 P1-P10 选择。启动器和独立
模块均默认使用 `yolo26n-pose`。现有的球杆精修在启动器中
默认保持启用。
精修使用角度间隙填充、混合窗口和受约束的杆身
线段。其长度为推断值，除非通过 `--club-marker-length-m` 提供
实测的标记距离。参见[改进日志](SHAFT_TRACKING_IMPROVEMENT_LOG.md)。

输出：`runs/workflows/videos/<video-stem>/`：

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

自动阶段检测仍是启发式的。除非运行提供十个互不相同、递增且具有有限值 3D 姿态的索引，
否则不能作为完成结果发布。
启用精修时，时序判定会结合精修后的球杆几何与身体相对坐标下的手腕
抬升/回落。收杆需要在双手抬起的情况下手腕速度持续较低；若缺乏该证据，区间将延伸至片段末尾。
球位校正被限制在手腕回落附近，并在 3D 提升之前执行。
运行 JSON 中的 `phase_detection` 记录相关证据及任何回退。
必要时可提供经人工审核的索引：

```powershell
python workflow.py video pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4 --positions-json runs/golf_pose_3d_vitpose_huge_17plus5/b85ed0b4db566e63bce190824c2f2196/selected_frames.json
```

使用 `--no-club-refinement` 可禁用球杆精修。`--pose-backend yolo26n-pose --club-pose-backend segmentation --club-crop-backend segmentation --lifter 17+2` 可恢复早期的
双代理点球杆输入。实测标记长度需要启用精修。

无需再次运行模型即可从已完成的运行中导出十个姿态：

```powershell
python workflow.py video --from-run runs/golf_pose_3d_club_refined_20261027/b85ed0b4db566e63bce190824c2f2196
```

缓存导入会保留已保存的数值结果和模型选择。它不会
应用新的精修或选择覆盖设置。它根据已保存的坐标导出十姿态数据包
和 3D 图像，并保留原始运行。

## 3. 对所选姿态评分

```powershell
python workflow.py score
# Or choose both inputs explicitly:
python workflow.py score runs/workflows/videos/b85ed0b4db566e63bce190824c2f2196 --references runs/workflows/references
```

评分器读取 `identified_frames.npz/json`，将 P1-P10 与参考
`joints.npz:reference_3d` 坐标进行匹配，并保留原始视频帧索引。
JPG 仅用于可视化这些姿态；其像素不是数值评分输入。
评分器不会再次识别帧。

现有的 14 项姿势标准、归一化、圆周角度差
和等权重均保持不变。参见 [SCORING_METHODOLOGY.md](SCORING_METHODOLOGY.md)。
17+5 中额外的三个杆头点不会增加评分标准。

输出：`runs/workflows/scores/<video-run>/`，包含 `score.json`、记录来源信息的
`workflow.json`、十张用户姿态图像和十张参考对比图像。
评分会报告位置/参数覆盖率。它并不能证明经过标定的
物理 3D 精度或教学有效性。

## 组织结构与保留

`workflow.py` 是公开的启动器。八个共享模块已移入
`golf_workflows/`：`standard_frames.py`、`golf_pose_3d.py`、
`compare_standard_frames.py`、`score_swing.py`、`trace_swing_plane.py`、
`swing_phases.py`、`vitpose_tracking.py` 和 `club_refinement.py`。
`cli.py` 连接三个阶段并验证它们已保存的数据契约。
检测器辅助程序、训练脚本、模型、数据集和以往的运行均保持原位。

底层命令仍可作为模块使用，例如
`python -m golf_workflows.golf_pose_3d --help`。历史代码快照保持
不变；若直接执行，其旧的导入语句可能需要调整。

显式指定的非空输出文件夹会被拒绝。默认输出在重复运行时会添加
时间戳后缀。`runs/workflows/latest_*.json` 下的最新运行指针
仅在成功完成后才会更新。失败的运行会保留诊断信息，但
不能替换已完成的当前结果。

## 重组验证（2026-10-27）

三个公开命令均已使用已完成的 ViTPose 17+5
参考和经过精修的视频通过 `--from-run` 进行测试，随后运行 `score`。这验证了
暂存、导出和评分流程，而无需重复耗时的 ViTPose 推理。
共导出十个参考姿态和十个视频姿态，全部 140 项比较均可用。
结果为 **93.8/100**。此外还使用真实模型完成了一次全新的参考构建：
全部 30 个相机姿态和十个参考均成功，位于
`runs/workflows/references_fresh_validation_20261027`。较早的 92.9 结果
使用的是不同的输入，已予以保留。

测试涵盖现有的 3D 提升/精修行为、十姿态契约、
原始源索引、显式的参考/输出路由、相同姿态的
评分、不完整的参考、不一致的清单以及输出保留。
使用 `python -m unittest discover -s tests -v` 运行时，全部 33 项测试均通过。
