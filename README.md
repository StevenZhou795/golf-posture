# 高尔夫姿态工作流

从一段挥杆视频中找出 P1–P10 十个关键帧，提升为 3D 姿态，并与十个职业参考姿态逐一比较评分。项目有两个相互独立的工作流，2D 检测方式不同，从 2D 关键点开始的帧识别、3D 提升和评分完全共用。

在项目文件夹中，用项目虚拟环境运行（环境配置见 [WORKFLOW.md](WORKFLOW.md) 第 6 节）：

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

## 3. 共用框架

两个工作流的视频命令都走 `golf_workflows/golf_pose_3d.py`：

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

## 4. 输出

| 阶段 | YOLO26n 工作流 | GolfPose 联合工作流 |
| --- | --- | --- |
| 参考 | `runs/workflows/references/` | `runs/golfpose_workflow/references/` |
| 视频 | `runs/workflows/videos/<视频名>_<时间戳>/` | `runs/golfpose_workflow/videos/<视频名>_<时间戳>/` |
| 评分 | `runs/workflows/scores/<视频运行名>/` | `runs/golfpose_workflow/scores/<视频运行名>/` |

每个完成的视频运行包含 `P1.jpg`–`P10.jpg`、`P1_3d.jpg`–`P10_3d.jpg`、击球前后两帧对照 `P7_contact.jpg`、`identified_frames.npz/json`、完整序列 `<stem>_3d.npz`、识别依据 `<stem>_3d.json`、`skeleton_diagnostics.json` 和渲染视频 `<stem>_3d.avi`。只有十个帧号互不相同、递增且 3D 姿态全为有限值时才算完成，并更新该工作流的最新结果指针。

对指定的运行评分：

```powershell
.\.venv\Scripts\python.exe workflow.py score runs/workflows/videos/<视频运行名> --references runs/workflows/references
.\.venv\Scripts\python.exe -m golfpose_workflow --score --run runs/golfpose_workflow/videos/<视频运行名> --references runs/golfpose_workflow/references
```

## 5. 常用选项

两个视频命令都支持：

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

## 6. 测试与基准

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m golf_workflows.benchmark evaluate <视频运行目录>
```

人工标注的帧范围在 `benchmarks/p_position_labels.json`。

早期实验（COCO 单图测试、分割掩膜杆头轨迹、GolfPosePro 速度分段、17+5 与 COCO ViTPose 后端等）的说明已移出本文，原文保存在 `archive/docs_zh_20261106/` 和 `archive/docs_en/`。
