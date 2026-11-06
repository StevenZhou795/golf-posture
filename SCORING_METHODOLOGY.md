# 标准参考姿态评分方法

本文档说明 [compare_standard_frames.py](golf_workflows/compare_standard_frames.py) 中实现的评分方法；`python workflow.py score` 使用该方法将保存的 P1–P10 姿态与十个参考姿态进行比较。新的参考位于 `runs/workflows/references/` 下；较早的参考集仍可通过显式路径使用。本文描述截至 2026-10-27 的当前实现。

结果是一个 0–100 的相似度评分，基于每个可用位置上的 14 个姿势角度。评分越高，表示测得的角度越接近参考。该评分没有经过校准的合格/不合格阈值。

## 1. 输入与位置匹配

公开的评分工作流读取由
`python workflow.py video` 生成的显式 P1-P10 数据包，以及一组经过验证的十个参考文件夹。

| 输入 | 使用的数据 |
| --- | --- |
| 视频 `identified_frames.npz` | 十个 `keypoints_3d` 姿态、GolfPose 关节名称、位置标签和原始帧索引 |
| 视频 `identified_frames.json` | 对应的位置/帧记录、惯用手和前导侧 |
| 参考 `P<number>-*/joints.npz` | `reference_3d`：19 或 22 个关节 × 3 个坐标 |
| 对应的 `reference_3d.jpg` | 用于对比图的参考渲染图 |

评分使用保存的 3D 关节坐标。源帧 JPG 和 `reference_3d.jpg`
用于可视化这些姿态；其像素不是数值评分输入。数据包
必须按顺序包含 P1-P10、有限坐标、预期的关节顺序，以及
十个互不相同且递增的从零开始的源索引。其 JSON 与 NPZ 中的选择
必须一致。评分不会重新识别帧，也不会挑选评分更高的
相邻帧。video 命令可以接受 `--positions-json` 以使用经人工审核的索引。

默认情况下，评分器使用最近一次完成的参考工作流和视频工作流，
它们记录在 `runs/workflows/latest_references.json` 和 `latest_video.json` 中。
也可以显式提供文件夹。每个 P 位置必须恰好有一个完整的参考。
使用 22 个关节时，同样的 14 项评分标准使用原有的身体及
杆身/杆颈索引；额外的三个头部点不增加任何评分标准。

较底层的比较器仍支持较旧的 `*_3d.npz/json` 运行结果。该旧版
路径使用保存的 `positions` 字典，并对 P1、P4 和
P7 采用阶段回退。公开的评分命令则改用显式的十姿态数据包。

## 2. 参考骨架的构建方式

[standard_frames.py](golf_workflows/standard_frames.py) 将每张参考图表裁剪为
正面（face-on）、沿目标线（down-the-line）和后方三个视图，并分别用 MixSTE2 将其提升为 3D。静态
图像会被重复以填满时序模型的输入窗口。各视图经过归一化，
并利用现有的镜像/非镜像偏航角搜索在正面视图空间中对齐。

数值化的 `reference_3d` 在符合条件的正面或沿目标线身体中选择置信度较高者，
并保留其完整的对齐骨架。原生 GolfPose
使用 17 个身体置信度分数的均值，无效关节按零计入；
基于 COCO 的旧版推理使用其现有的置信度/有效比例评分。
平局时优先选择正面视图。后方视图仅用于诊断。不再融合来自
不同身体的空间分量。

球杆叠加层独立地从正面或沿目标线视图中选择一套完整的对齐球杆几何，
优先使用完整的观测；如果所有符合条件的观测都不完整，则回退到置信度足够的
仅杆身观测。该叠加层
是一项渲染功能：评分仍读取 `reference_3d`，而不是叠加图像
或另一套球杆骨架。

`python workflow.py score` 和 `python -m golfpose_workflow --score` 都调用
同一个比较器，并使用下文相同的 14 项评分标准和公式。独立的
关节模型工作流在 `runs/golfpose_workflow/` 下保存其自己的最新参考/视频/评分指针；
它不会回退到 `runs/workflows/`。

## 3. 归一化、坐标与惯用手

每个用户姿态和参考姿态都独立进行归一化：

```text
centered_joint = joint - root
scale = norm(thorax - root)
normalized_joint = centered_joint / scale
```

`root` 是髋部中心关节。尺度取根节点到胸部（thorax）的长度，而不是名为 `spine` 的关节的长度。任何包含非有限坐标或尺度小于 `1e-6` 的姿态都会被排除。归一化消除了平移和整体尺寸差异；角度计算本身对整体缩放已不敏感。

计算使用保存的相机坐标，向上方向为 `U = (0, -1, 0)`，水平面为 XZ。渲染时可能为显示而重新排列坐标轴；评分使用保存的坐标。

当 `lead_side` 为 `left` 或 `right` 时，前导侧取自该值。否则，右手球手的前导侧为左侧，左手球手的前导侧为右侧。相反的一侧为后侧。用户的前导侧选择同时应用于用户和参考的计算；评分器不会自动镜像惯用手相反的参考。

## 4. 角度定义

在每个归一化骨架上使用以下记号：

```text
d(A, B) = (B - A) / norm(B - A)
angle(v, w) = degrees(acos(clip(dot(v, w), -1, 1)))
joint_angle(A, B, C) = angle(d(B, A), d(B, C))
horizontal(v) = normalize((v.x, 0, v.z))
signed_up(v, axis) = degrees(atan2(dot(v, axis), dot(v, U)))
```

长度非有限或接近零的方向不可用；向量长度的截断阈值为 `1e-8`。

共用的方向为：

```text
S = d(root, thorax)
T = normalize(d(left_hip, left_knee) + d(right_hip, right_knee))
L = horizontal(lead_shoulder - trail_shoulder)
F = normalize(cross(U, L))
```

`T` 是两条大腿单位方向的平均。`L` 水平指向前导肩。`F` 是由 `U` 和 `L` 构造的前向轴。

下列全部 14 项测量会在每个位置上尝试计算；本评分器没有针对特定位置的权重，也没有不同的评分标准列表。

| `score.json` 中的参数 | 精确测量方式 |
| --- | --- |
| Alignment | `degrees(acos(abs(dot(horizontal(left_shoulder - right_shoulder), horizontal(left_hip - right_hip)))))`，并对余弦值进行截断。点积取绝对值，因此将两条线视为无方向。 |
| Hip Bend | `angle(S, T)` |
| Hip Depth | `angle(T, U)` |
| Knee Bend | 左右两侧 `joint_angle(hip, knee, foot)` 的均值；两侧都必须可用。 |
| Hands | `angle(S, d(thorax, midpoint(left_wrist, right_wrist)))` |
| Stance Width | `joint_angle(left_foot, root, right_foot)` |
| Flying Elbow | `angle(d(trail_shoulder, trail_elbow), d(trail_shoulder, trail_hip))` |
| Trail Knee | `joint_angle(trail_hip, trail_knee, trail_foot)` |
| Bend | `signed_up(S, F)` |
| Shaft Lean | `signed_up(d(shaft, hosel), L)` |
| Spine Tilt | `signed_up(S, -L)` |
| Lead Shoulder | `angle(d(thorax, lead_shoulder), d(lead_shoulder, lead_elbow))` |
| Lead Hip | `angle(d(lead_hip, root), d(lead_hip, lead_knee))` |
| Lead Knee | `joint_angle(lead_hip, lead_knee, lead_foot)` |

这些名称只是上述几何定义的标签。例如，Hip Depth 是一个角度，而不是测得的位移；Stance Width 是一个角度，而不是距离；Knee Bend 是膝关节内角，而不是 `180 - internal_angle`。

## 5. 将角度差转换为评分

对于两个骨架中都可用的每个参数，计算最短的环形角度差：

```text
gap = abs(user_angle - reference_angle) % 360
if gap > 180:
    gap = 360 - gap

parameter_score = round(100 * (1 - gap / 180), 1)
```

| 角度差 | 参数评分 |
| --- | ---: |
| 0 度 | 100.0 |
| 10 度 | 94.4 |
| 30 度 | 83.3 |
| 90 度 | 50.0 |
| 180 度 | 0.0 |

所有参数都使用同一个线性 180 度标尺，包括自然变化范围较小的参数。该标尺对中等程度的差异相对宽容：高评分并不意味着所有角度都几乎相同。

## 6. 权重、取整与缺失数据

每个可用的参数比较具有相同的权重：

```text
position_score = round(mean(available parameter scores at that position), 1)
overall_pose_score = round(mean(all available parameter scores across P1–P10), 1)
```

参数评分在汇总前先取整到一位小数。总评分直接由这些参数评分计算，而不是对显示出来的、已取整的位置评分求平均。

当十个位置上全部 14 个参数都可用时，总评分是 140 个值的平均。此时在显示取整之前，每个位置的贡献相同。存在缺失参数时，各位置按其可用比较的数量成比例地贡献。

缺少参考、缺少用户帧选择、姿态非有限或归一化失败，都会导致整个位置被记录为 `null`。在其他方面有效的姿态中，未定义的参数会同时从参数字典和分母中省略。没有任何可用比较的位置为 `null`；没有任何比较的总结果同样为 `null`。缺失值不按零分计。保存的帧索引必须位于用户姿态数组的范围之内。

评分器不应用检测器置信度权重，也不检查一个有限值关节最初是被检测到的、插值得到的，还是由其他关节填充的。因此，填充的关节也可能对评分产生贡献。

## 7. 显示对齐与评分不变性

为生成对比图，`best_yaw()` 以 15 度为步长绕竖直轴旋转归一化后的用户骨架，并选择与参考之间平均骨骼方向余弦相似度最高的旋转。

保存的 `yaw_deg` 描述的就是这一显示对齐。参数评分使用的是该显示旋转之前的原始归一化姿态。由于这些参数基于内角、水平方向的相对对齐以及相对于肩部的坐标轴来定义，它们对骨架整体的共同偏航旋转是不变的。但这并不能消除相机俯仰/横滚、3D 重建过程中引入的误差，或参考融合中的误差。

## 8. 运行与保存的结果

```powershell
python workflow.py score
# Or choose both inputs explicitly:
python workflow.py score runs/workflows/videos/b85ed0b4db566e63bce190824c2f2196 --references runs/workflows/references
```

结果写入单独的 `runs/workflows/scores/<video-run>/` 文件夹。后续运行会使用
带版本号的名称，从而保留已有文件夹。

```text
score.json
workflow.json
P1_user_3d.jpg ... P10_user_3d.jpg
P1_vs_reference.jpg ... P10_vs_reference.jpg
```

`score.json` 记录总评分、等权重设置、原始视频帧
索引、各位置评分、14 项参数评分、输入文件夹、评分标准
名称以及覆盖情况。不保存原始的用户/参考角度。即使导出的数据包只包含十个姿态，
所选的源索引仍会被保留。

重组后的冒烟测试复用了已完成的精修视频以及 ViTPose
17+5 参考。它对全部十个位置和全部 140 项参数比较进行了评分，
得到 **93.8/100**。这与较早的 **92.9/100** 基线
使用的是不同的输入组合；两次较早的输出文件夹均已保留。

## 9. 解读与适用范围

该评分衡量的是在上述公式下与保存的参考骨架之间的一致程度。它不衡量击球质量、挥杆速度、节奏或杆面朝向，也不是一个概率或技术正确率的百分比。其准确性取决于帧选择、2D 检测、缺口填充、3D 提升和参考构建。本文所述方法没有针对教练评估进行过经验校准。

部分测量存在重叠：Trail Knee 和 Lead Knee 重复了 Knee Bend 中已体现的单侧膝关节角度。因此，等权重并不意味着各项是相互独立的生物力学证据。单一的汇总评分也可能掩盖某个参数上较大的差异；请结合参数评分和对比图一起查看。

独立的 [score_swing.py](golf_workflows/score_swing.py) 工作流使用针对特定位置的评分标准将用户与经 3D 提升的参考视频进行比较，并单独报告一个杆头路径评分。其评分标准列表和路径评分不用于本文档所述的标准图表方法。

## 实现参考

- [compare_standard_frames.py](golf_workflows/compare_standard_frames.py)：姿势定义、环形角度差、汇总、显示对齐和输出。
- [standard_frames.py](golf_workflows/standard_frames.py)：归一化和多视图参考构建。
- [score_swing.py](golf_workflows/score_swing.py)：共用的帧查找、前导侧选择、向量归一化和线性角度评分辅助函数。
- [WORKFLOW.md](WORKFLOW.md)：外围的检测、3D 提升和比较流水线。
