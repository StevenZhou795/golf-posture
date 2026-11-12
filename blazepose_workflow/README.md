# BlazePose GHUM 高尔夫姿态工作流（仅人体）

独立于 `workflow.py` / `golfpose_workflow` 的第三个工作流：用 MediaPipe BlazePose GHUM 对挥杆视频做逐帧 2D + 3D 人体姿态推理，再只凭人体关键点识别 P1–P10。不做球杆、球和面部关键点。

- 模型：与 tfjs-models [`pose-detection`](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection) 中 BlazePose 相同的 GHUM `lite` / `full` / `heavy` 三档。浏览器 API 在 Python 中的对应物是 `mediapipe` Tasks 的 `PoseLandmarker`（`RunningMode.VIDEO`，带跟踪）。
- 3D：每个关键点同时输出 [BlazePose GHUM 3D](https://blog.tensorflow.org/2021/08/3d-pose-detection-with-mediapipe-blazepose-ghum-tfjs.html) 的世界坐标（米，髋中心为原点）。
- 代码只依赖本文件夹；与主项目 `.venv` 分开安装（mediapipe 自带 `opencv-contrib-python`，与主项目固定的 `opencv-python` 冲突）。

## 安装

```powershell
py -3.12 -m venv .venv_blazepose
.\.venv_blazepose\Scripts\python.exe -m pip install -r blazepose_workflow\requirements.txt
.\.venv_blazepose\Scripts\python.exe -m unittest discover -s blazepose_workflow\tests
```

模型文件 `pose_landmarker_<lite|full|heavy>.task` 在第一次使用时从 `storage.googleapis.com/mediapipe-models` 下载到 `blazepose_workflow/models/`（heavy 约 30 MB，lite 约 6 MB），不纳入 git。

## 用法

```powershell
.\.venv_blazepose\Scripts\python.exe -m blazepose_workflow video pro_refs\005.mp4 --model heavy
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model` | `heavy` | `lite` / `full` / `heavy`，精度递增、速度递减 |
| `--view` | `auto` | `face_on` / `down_the_line` |
| `--handedness` | `auto` | `right` / `left` |
| `--golfer-box X1 Y1 X2 Y2` | 无 | 开头几帧中包含球手髋部的像素框，画面里有多人且自动选择错误时使用 |
| `--no-crop` | 关 | 不做第二遍 ROI 推理 |
| `--out DIR` | `runs/blazepose_workflow/videos/<stem>_<时间>` | 输出文件夹 |

CPU 上 heavy 模型两遍推理约 2 分钟 / 570 帧（005）。

## 流程

1. **解码与时间戳**（`inference.py`）：OpenCV 逐帧读取，时间取 `CAP_PROP_POS_MSEC`（手机慢动作视频常为可变帧率），转为严格递增的整数毫秒送入 `detect_for_video`。
2. **第一遍：全画面**，`num_poses=3`。球手的选择：
   - 有锚点（上一帧球手的髋中心，500 ms 内有效）时，取距离最近且位移不超过 0.5 个身高的人；
   - 没有锚点时，取“身高 × (0.5 + 0.5 × 居中程度)”最大的人；给了 `--golfer-box` 时只在框内选。
3. **第二遍：球手 ROI**（默认）：由第一遍球手框的 2%/98% 分位范围外扩（左右各 15%，上 25%），裁剪后放大到长边 ≥ 960 px 再推理，坐标映射回原图。ROI 超过画面 80% 时跳过。第二遍漏检的帧用第一遍结果补上。
4. **清理**（`smoothing.py`，`pipeline.py`）：
   - 2D 和 3D 都只丢弃 presence < 0.5（BlazePose 认为在画面外）的点。被身体遮挡的关节（visibility < 0.5）保留预测值：后方视角送杆时前臂手腕在身体后面，005 中 233–260 帧前腕 visibility ≈ 0，若按 visibility 丢弃，双手信号会整段缺失。每帧被遮挡关节数记录在 `occluded_2d_landmarks_per_frame`。
   - 不超过 0.3 秒的缺口按时间线性插值，再用约 0.1 秒窗口的 Savitzky–Golay（二阶）平滑。
   - 3D 额外做骨长归一化：GHUM 世界坐标的尺度逐帧变化，先把髋宽缩放到全片中位数（保持髋中心原点），再从髋部沿骨架逐段把骨长设为中位数，方向不变。
5. **视角、左右手、P1–P10**（`positions.py`，见下节）。
6. **导出**（`render.py`，`pipeline.py`）。

## 关键点（23 个）

BlazePose 33 点中去掉 1–10（眼、耳、嘴），保留鼻子（只作头部参考和朝向判断）与 11–32：

| 输出序号 | BlazePose 序号 | 名称 |
| --- | --- | --- |
| 0 | 0 | `nose` |
| 1–2 | 11–12 | `left_shoulder`, `right_shoulder` |
| 3–4 | 13–14 | `left_elbow`, `right_elbow` |
| 5–6 | 15–16 | `left_wrist`, `right_wrist` |
| 7–8 | 17–18 | `left_pinky`, `right_pinky` |
| 9–10 | 19–20 | `left_index`, `right_index` |
| 11–12 | 21–22 | `left_thumb`, `right_thumb` |
| 13–14 | 23–24 | `left_hip`, `right_hip` |
| 15–16 | 25–26 | `left_knee`, `right_knee` |
| 17–18 | 27–28 | `left_ankle`, `right_ankle` |
| 19–20 | 29–30 | `left_heel`, `right_heel` |
| 21–22 | 31–32 | `left_foot_index`, `right_foot_index` |

左右是球手自身的左右。“双手”= 两侧手腕、食指、拇指的平均。

## 坐标约定

博客对 `keypoints3D` 的定义：

> The x, y, z are in meter units. The person is modeled as if they were in a 2m x 2m x 2m cubic space. The range for each axis goes from -1 to 1 (therefore 2m total delta). The origin of this 3D space is the hip center (0, 0, 0). From the origin, z is positive if moving closer to the camera, and negative if moving away from the camera.

Python `PoseLandmarker` 的 `pose_world_landmarks` 是同一个模型输出，单位与原点相同，但**轴向不同**：x 向右，y 向下（与图像一致），z **远离**相机为正（MediaPipe 约定；tfjs 封装把 z 取反）。在 005（后方视角、右手）的准备姿势上实测：离相机近的右肩 z = −0.18，远的左肩 z = +0.04。本工作流保存的 3D 就是这个 MediaPipe 约定，未做转换。

- `image_2d` / `image_raw`：原图像素，y 向下；`image_raw[..., 2]` 是 MediaPipe 的相对深度 z 乘以裁剪宽度，`[..., 3]` visibility，`[..., 4]` presence。
- `world_3d` / `world_raw`：米，髋中心原点，x 右、y 下、z 远离相机。
- 3D 图 `P*_3d.jpg` 画成（x，z，−y），即竖直轴向上。

## P1–P10 规则（只用人体）

定义沿用 HackMotion（见 `FRAME_RECOGNITION.md` 第 1 节）。没有球杆时，以前臂（右手球手为左臂）与竖直向下方向的 3D 夹角作代理：0° 自然下垂，90° 与地面平行。

| 位置 | 规则 |
| --- | --- |
| P4 顶点 | 双手（相对髋部、以躯干长度计）高度峰值，显著度 ≥ max(0.5, 0.5 × 高度范围)，并且之后有一次下降（双手回升到下降深度一半前为止）不小于最大下降的一半；取第一个满足的峰 |
| P1 准备 | 顶点前最低点附近最后一段静止（< 0.6 躯干/秒，持续 0.2 秒）；否则取顶点前双手最低处 |
| P2 | 前臂第一次上升穿过“准备时角度与 90° 的中点”（005：53°） |
| P3 | 前臂第一次上升穿过 90° |
| P5 | 顶点后前臂第一次下降穿过 90° |
| P6 | 击球前前臂最后一次下降穿过“击球时角度与 90° 的中点”（005：56°）；取最后一次以避开下杆中 3D 手臂的单帧抖动 |
| P7 击球 | 顶点之后双手最低点（看不到球，不是触球瞬间） |
| P8 | 击球后前臂第一次上升穿过 P6 的角度 |
| P9 | 击球后前臂第一次上升穿过 90° |
| P10 | P9 之后双手高且静止的开始；否则 P9 之后双手最高处 |

P2、P3、P5、P6、P8、P9 给出插值的亚帧时间（`details.<p>.time_ms`）。中点角度而不是固定 45°：GHUM 的前臂在准备和击球时并不竖直（005 分别约 16° 和 21°），固定 45° 会让 P8 落在击球后两三帧。

视角：先看 3D 肩线在左右方向上的占比（≥ 0.7 为正面，≤ 0.35 为后方），不确定时再看 2D 肩宽 / 躯干长（≥ 0.35 正面，< 0.25 后方），仍不确定时默认后方。左右手（阈值 0.2 躯干长）：正面视角看从准备到顶点双手移向画面哪侧（右手球手向左）；后方视角看准备时双手在髋部画面的哪一侧（右手球手在右）；都不确定时后方视角看鼻子相对髋部的方向，最后默认右手。

**中止条件**（`SwingCoverageError`，先写出所有输出再报错，命令行返回码 2）：
- 视频从顶点之后开始（双手在开头 0.1 秒内最高，之后只下降）；
- 顶点前没有足够的上升（< 高度范围的一半）；
- 击球后前臂没有抬到 90°（视频在送杆前结束）。

## 输出

```text
<stem>_pose.npz            joints、times_ms、image_raw/world_raw（模型原始输出）、image_2d/world_3d（清理后）、found、candidates
<stem>_pose.json           检测报告、清理参数、每帧缺失/遮挡关节数、positions、details、视角与左右手判断依据、error
identified_frames.json     P1–P10 帧号（从 0 开始）与时间（毫秒）
<stem>_blazepose.mp4       骨架叠加视频（左侧绿、右侧橙、中线蓝）
P1.jpg ... P10.jpg         源帧 2D 叠加
P1_3d.jpg ... P10_3d.jpg   3D 骨架（相机视角、侧视、斜视）
```

## 已知限制

- 没有球杆和球：P2/P6/P8 是前臂代理，P7 是双手最低点；与主工作流的帧号可差几帧（005：P6 205 对 216，P7 221 对 225，P10 276 对 304）。
- GHUM 3D 是单目先验，肘部在送杆中很少显示为伸直（005 最大约 149°），手臂角度在快速下杆中有单帧抖动。
- 后方视角时前侧肢体长期被遮挡，这些点是模型猜测值（见 `occluded_2d_landmarks_per_frame`）。
- 画面中有多人时，自动选择依赖“居中且最大”，必要时用 `--golfer-box`。

## 测试

`tests/` 用合成挥杆（正面视角，右手和左右镜像的左手）检查关键点子集、骨长归一化、球手选择、P1–P10 顺序、左右手判断和两种中止条件：

```powershell
.\.venv_blazepose\Scripts\python.exe -m unittest discover -s blazepose_workflow\tests
```
