# 高尔夫姿态工作流

从一段挥杆视频中找出 P1–P10 十个关键帧，并导出二维骨架和三维姿态。项目提供四个工作流：YOLO26n、GolfPose 联合、BlazePose GHUM，以及 RTMW 2D + SwingTrace。前两个支持职业参考姿态比较评分；BlazePose 使用独立的人体动作规则，RTMW 复用原有选帧与 MixSTE 流程，目前这两个入口提供视频识别和导出。

在项目文件夹中运行。YOLO26n / GolfPose 使用主项目 `.venv`（环境配置见 [WORKFLOW.md](WORKFLOW.md) 第 6 节）；BlazePose 和 RTMW 使用各自的独立环境，命令见对应章节：

```powershell
cd J:\Golf_Posture
```

## 1. YOLO26n 工作流（`workflow.py`）

```powershell
.\.venv\Scripts\python.exe workflow.py references
.\.venv\Scripts\python.exe workflow.py video pro_refs/w07.mp4
.\.venv\Scripts\python.exe workflow.py score
```

- 人体：双类 YOLOX（`golfpose_detector_2cls_yolox_s.pth`，类别 person/club）给出人物框，框内运行 YOLO26n 人体姿态（`yolo26n-pose.pt`）。
- 球杆：YOLOX 的球杆框内运行仅球杆的 ViTPose-H（`golfpose_club_ViTPose_huge.pth`），只用握把 `shaft` 与杆颈 `hosel` 两点。
- 输出：`runs/workflows/`。

## 2. GolfPose 联合工作流（`python -m golfpose_workflow`）

```powershell
.\.venv\Scripts\python.exe -m golfpose_workflow --reference
.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/w07.mp4
.\.venv\Scripts\python.exe -m golfpose_workflow --score
```

- 单类 YOLOX（`golfpose_detector_1cls_yolox_s.pth`，类别 golfer）给出球手框，框内运行 22 点联合 ViTPose-H（`golfpose_golfer_ViTPose_huge.pth`），一次得到人体和球杆。
- 低置信度人体关节由 YOLO26n 补充（`--no-body-fusion` 关闭）。杆跟/杆趾三点被丢弃。
- 输出：`runs/golfpose_workflow/`。详见 [golfpose_workflow/README.md](golfpose_workflow/README.md)。

## 3. BlazePose GHUM 工作流（`python -m blazepose_workflow`）

用于只凭人体动作寻找十帧，采用 MediaPipe `PoseLandmarker` 的 VIDEO 跟踪模式。人体二维像素坐标与 GHUM 三维世界坐标由同一模型输出，三维以髋中心为原点；不需要 MixSTE。

```powershell
# 独立环境，避免 MediaPipe 的 OpenCV 依赖与主环境冲突
py -3.12 -m venv .venv_blazepose
.\.venv_blazepose\Scripts\python.exe -m pip install -r blazepose_workflow/requirements.txt
.\.venv_blazepose\Scripts\python.exe -m blazepose_workflow video pro_refs/005.mp4 --model heavy
```

流程：

1. **全画面检测与球手跟踪**：逐帧读取实际时间戳，在最多三个人中按髋中心连续性、人物大小和居中程度选择球手；可用 `--golfer-box` 指定球手区域。
2. **球手 ROI 第二遍推理**：默认根据第一遍结果裁剪并放大球手区域，映射回原图；第二遍漏检时使用第一遍结果。`--no-crop` 可跳过这一遍。
3. **清理与三维尺度处理**：33 点中保留 23 点，含鼻子参考和人体、手足点；按 presence 过滤，保留遮挡点的预测，插值短缺口并平滑，再归一化三维髋宽和骨长。
4. **人体十帧识别**：自动判断视角和惯用手，使用双手高度、运动速度与前导前臂角度寻找 P1–P10。P2/P6/P8 使用前臂代理，P7 使用顶点后的双手最低点，因此与有球杆、球证据的选帧结果可能不同。
5. **覆盖检查与导出**：检测到缺少上杆或送杆等不完整挥杆时，保留诊断并报错；导出原始/清理后坐标、二维叠加视频及十帧图片。

模型支持 `lite` / `full` / `heavy`，默认 `heavy`；首次使用会下载对应 `.task` 文件到 `blazepose_workflow/models/`。默认输出为 `runs/blazepose_workflow/videos/<视频名>_<时间戳>/`，可用 `--out` 指定。

详细坐标约定、十帧代理规则及输出字段见 [blazepose_workflow/README.md](blazepose_workflow/README.md)。

## 4. RTMW 2D + SwingTrace 工作流（`python -m rtmw_workflow`）

人体使用 **RTMW-X 2D**，输入高 256、宽 192，输出 133 个全身二维点；球杆使用 **SwingTrace YOLO11m** 的杆身、杆头、双手／握把三类检测框。

```powershell
# 安装独立环境；已有环境可直接运行 self-test
powershell -ExecutionPolicy Bypass -File rtmw_workflow/scripts/setup.ps1
.\rtmw_workflow\.venv\Scripts\python.exe -m rtmw_workflow self-test
.\rtmw_workflow\.venv\Scripts\python.exe -m rtmw_workflow video pro_refs/005.mp4
.\rtmw_workflow\.venv\Scripts\python.exe -m rtmw_workflow batch --input-dir pro_refs
```

流程：

1. **球手定位**：YOLO26n 选择初始 ROI，YOLOX 提供逐帧人物框，沿用连续跟踪和骨架恢复裁剪。YOLO26n 不补充 RTMW 的人体关节。
2. **二维人体推理**：严格加载 `models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth`，使用官方配置和坐标还原；保存全部 133 点及置信度，提取前 17 个 COCO 点适配原流程。
3. **球杆检测与关联**：每帧运行一次 `models/swingtrace-yolo11m.pt`，保存三类原始框，并用当前球手手腕和握把位置关联球杆。杆身框中心、杆头框中心作为旧接口两列的代理，漏检保留缺失；这些框中心并非精确的 GolfPose 球杆标记。
4. **球与十帧依据**：YOLOE 寻找球的放置位置，也可用 `--ball-position X Y` 指定。复用球杆清理、跟踪、骨架质量检查、MixSTE 17+2 和 P1–P10 顺序规则；保留实际回退来源，单凭球的位置不能确认触球。
5. **渲染与导出**：输出 133 点骨架视频，逐帧绘制 `shaft`、`head`、`hands/grip` 框及置信度；十帧图片和三维对照视频原画面侧也显示检测框。保存原始检测、选帧依据、时间戳、模型信息和批次成败汇总。

当前入口使用 RTMW 2D 与 SwingTrace，不加载 RTMW3D、球杆 ViTPose 或球杆分割模型。三维选帧由 MixSTE 提供。默认 CPU；安装 CUDA 环境时使用 `setup.ps1 -Cuda`，运行时可指定 `--pose-device cuda --club-device cuda`，以 FP32 分阶段使用 GPU。

默认输出为 `rtmw_workflow/outputs/<批次>/<视频名>/`，批处理只读取输入根目录视频，单段失败后继续其余视频。`self-test` 只使用合成数据，不读取视频。

详细运行参数见 [rtmw_workflow/README.md](rtmw_workflow/README.md)，三类框适配与十帧依据见 [rtmw_workflow/FRAME_RULES.md](rtmw_workflow/FRAME_RULES.md)。

## 5. YOLO26n / GolfPose 共用框架

YOLO26n 和 GolfPose 联合工作流的视频命令都走 `golf_workflows/golf_pose_3d.py`；RTMW 也调用这一流程，但使用上节说明的独立人体/球杆适配，不启用 ViTPose 重裁剪或分割掩膜角度补充。BlazePose 使用自己的清理和选帧实现。

原有两个工作流的完整步骤为：

1. **球手 ROI**：YOLO26n 在 12 个采样帧上找出球手，作为 YOLOX 漏检时的默认裁剪区域（`--golfer-box` 可手动指定，`--no-auto-golfer-box` 关闭）。
2. **逐帧 2D 检测**，并自动纠正漂移的人体框：骨架不完整时，在上一帧骨架范围和球手 ROI 中重试。球杆置信度低时沿预测杆身重新裁剪一次。
3. **视角与左右手习惯**：默认自动判断（`--view`、`--handedness` 可手动指定）。
4. **球杆跟踪**：以双手为锚点的刚性杆卡尔曼/RTS 跟踪，纠正握把/杆颈互换、剔除误检；检测不可靠时补充运动模糊条纹和分割掩膜的杆身角度（`--no-club-mask` 关闭掩膜）。
5. **阶段识别（2D）**：P1、P4、送杆结束；P7 以击球瞬间为准——球离开原位、杆头到达球优先，动作捕捉（杆头回位、手腕回落）其次。
6. **骨架质量检查**：挥杆窗口内骨架丢失过久时打印 `ERROR:` 并中止（`--allow-skeleton-gaps` 改为警告）。
7. **3D 提升**：MixSTE2 17+2（`golfpose_17+2_33.6_30.5_59.5.bin`），只提升 P1 到送杆结束的窗口，并拟合刚性球杆。
8. **其余位置（3D）**：P2、P3、P5、P6、P8、P9、P10 分段用动态规划联合选取，附带每个位置的置信度和亚帧时间。
9. **导出与评分**：十个源帧、十张 3D 图和坐标包；评分按 P1 对 P1……P10 对 P10 计算 14 项姿态角度的相似度。

帧识别规则的完整说明见 [FRAME_RECOGNITION.md](FRAME_RECOGNITION.md)，评分公式见 [SCORING_METHODOLOGY.md](SCORING_METHODOLOGY.md)，流程细节与修改记录见 [WORKFLOW.md](WORKFLOW.md)。

## 6. 输出

| 阶段 | YOLO26n 工作流 | GolfPose 联合工作流 |
| --- | --- | --- |
| 参考 | `runs/workflows/references/` | `runs/golfpose_workflow/references/` |
| 视频 | `runs/workflows/videos/<视频名>_<时间戳>/` | `runs/golfpose_workflow/videos/<视频名>_<时间戳>/` |
| 评分 | `runs/workflows/scores/<视频运行名>/` | `runs/golfpose_workflow/scores/<视频运行名>/` |

BlazePose 和 RTMW 的输出：

| 工作流 | 默认目录 | 主要文件 |
| --- | --- | --- |
| BlazePose | `runs/blazepose_workflow/videos/<视频名>_<时间戳>/` | `<stem>_pose.npz/json`、`<stem>_blazepose.mp4`、`identified_frames.json`、P1–P10 源帧与三维图 |
| RTMW | `rtmw_workflow/outputs/<批次>/<视频名>/` | `wholebody.npz/json`、`swingtrace_detections.json`、`frame_evidence.json`、`rtmw_2d.mp4`、`identified_frames.npz/json`、P1–P10 图片及总览；批次根目录另有 `batch_summary.json`、`validation.json` |

YOLO26n / GolfPose 完成的视频运行包含 `P1.jpg`–`P10.jpg`、`P1_3d.jpg`–`P10_3d.jpg`、击球前后两帧对照 `P7_contact.jpg`、`identified_frames.npz/json`、完整序列 `<stem>_3d.npz`、识别依据 `<stem>_3d.json`、`skeleton_diagnostics.json` 和渲染视频 `<stem>_3d.avi`。只有十个帧号互不相同、递增且 3D 姿态全为有限值时才算完成，并更新该工作流的最新结果指针。

对指定的运行评分：

```powershell
.\.venv\Scripts\python.exe workflow.py score runs/workflows/videos/<视频运行名> --references runs/workflows/references
.\.venv\Scripts\python.exe -m golfpose_workflow --score --run runs/golfpose_workflow/videos/<视频运行名> --references runs/golfpose_workflow/references
```

## 7. 常用选项

本节选项适用于 YOLO26n 和 GolfPose 联合工作流；BlazePose、RTMW 的参数见各自 README。

| 选项 | 作用 |
| --- | --- |
| `--view auto\|face_on\|down_the_line` | 拍摄视角，默认自动 |
| `--handedness auto\|right\|left` | 左右手习惯，默认自动 |
| `--golfer-box X1 Y1 X2 Y2`、`--no-auto-golfer-box` | 手动球手 ROI / 关闭自动 ROI |
| `--positions-json <文件>` | 使用人工复核的 P1–P10 帧号 |
| `--allow-skeleton-gaps` | 骨架丢失时只警告、不中止 |
| `--no-club-mask` | 不用分割掩膜测杆身角度 |
| `--no-club-refinement` | 关闭球杆精修，退回旧的速度规则 |
| `--track-club-path` | 渲染杆头轨迹 |
| `--club-marker-length-m <米>` | 提供实测的握把到杆颈距离 |
| `--pose-cpu-threads <n>` | CPU 线程数 |
| `--output-dir <空文件夹>` | 指定输出目录 |

`workflow.py video` 还支持 `--from-run <运行目录>`（不重跑模型，直接从已有运行导出十个姿态）。查看全部选项：

```powershell
.\.venv\Scripts\python.exe workflow.py video --help
.\.venv\Scripts\python.exe -m golfpose_workflow --help
```

## 8. 测试与基准

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m golf_workflows.benchmark evaluate <视频运行目录>
.\.venv_blazepose\Scripts\python.exe -m unittest discover -s blazepose_workflow/tests
.\rtmw_workflow\.venv\Scripts\python.exe -m rtmw_workflow self-test
```

人工标注的帧范围在 `benchmarks/p_position_labels.json`。

早期实验（COCO 单图测试、分割掩膜杆头轨迹、GolfPosePro 速度分段、17+5 与 COCO ViTPose 后端等）的说明已移出本文，原文保存在 `archive/docs_zh_20261106/` 和 `archive/docs_en/`。
