from pathlib import Path
p=Path('WORKFLOW.md')
s=p.read_text(encoding='utf8')
start=s.index('Reference generation uses the previous')
end=s.index('## Python environment and dependencies',start)
s=s[:start]+'''## Joint golfer-with-club inference (2026-11-03)

The default reference and video workflows now load one 2D pose checkpoint:
`golfpose_checkpoints/golfpose_golfer_ViTPose_huge.pth`. Its saved configuration
and strict parameter load confirm **22 native GolfPose joints**: 17 body joints
followed by `shaft`, `hosel`, `heel`, `toe_down`, and `toe_up`.

```powershell
.\\.venv\\Scripts\\python.exe workflow.py references
.\\.venv\\Scripts\\python.exe workflow.py video pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4
```

The shared backend is `golfpose-golfer-vitpose-huge`. It runs one pose model on
an entire single-golfer reference panel or video frame, using the existing
192x256 UDP crop, DARK heatmap decoding, and original/mirrored averaging.
Body channels use the native GolfPose left/right swaps; club channels do not
swap. No COCO joint conversion or synthesized body joints are needed. No
YOLO person detector, club segmentation, COCO body checkpoint, or club-only
ViTPose checkpoint is loaded by this default 2D path. The optional video ball
model remains available solely for the existing phase-timing correction.

Raw 22-point coordinates and confidence are retained. A point is valid only
when its score is at least 0.10 and its coordinates are finite and inside the
original image. Missing inputs are filled by the existing routine before
MixSTE2 17+5 lifting. Normalization uses the bounds of reliable predicted body
joints. Reference alignment, view fusion, body/club-overlay rendering, video
phase selection, club refinement, ten-frame export, and scoring retain their
existing algorithms and file contracts.

References additionally save `<view>_2d_confidence` (22 values) in `joints.npz`;
video arrays save `keypoints_2d_confidence` (frames x 22). The pose/club backend
metadata refer to the same loaded model instance. `--lifter 17+5` is required;
legacy `--club-pose-backend` and `--club-crop-backend` flags are ignored while
the joint backend is selected. Inputs must contain one golfer; a small golfer
in a wide frame may lose detail at the fixed network input resolution.

To retain the earlier two-pose-model route, explicitly select
`--pose-backend vitpose-huge --club-pose-backend vitpose-huge`; references use
`--club-crop-backend golfer-box`, while videos can use `segmentation` or
`official-yolox-2cls` crops. DEKR remains removed. `--from-run` reuses old model
results, so omit it when generating outputs with the new checkpoint. Existing
completed runs and latest pointers are not changed by the implementation update.

''' +s[end:]
s=s.replace('人体、球杆和三维重建由不同模型负责，阶段识别与评分则由几何规则和数值计算完成。','人体与球杆由同一个 22 点 GolfPose ViTPose 模型负责，三维重建由 MixSTE2 负责；阶段识别与评分由几何规则和数值计算完成。')
s=s.replace('以下说明对应 `python workflow.py` 的默认配置：`vitpose-huge`、`17+5`，','以下说明对应 `python workflow.py` 的默认配置：`golfpose-golfer-vitpose-huge`、`17+5`，')
a=s.index('| YOLO26n-pose |');b=s.index('| GolfPose MixSTE2',a)
s=s[:a]+'''| GolfPose golfer ViTPose-Huge | `golfpose_checkpoints/golfpose_golfer_ViTPose_huge.pth` | 同时预测 17 个人体点与 5 个球杆点；整张单人图片作为裁剪范围 | references、video |
'''+s[b:]
s=s.replace('使用 `--club-pose-backend segmentation --lifter 17+2` 时，改用','使用 `--pose-backend yolo26n-pose --club-pose-backend segmentation --club-crop-backend segmentation --lifter 17+2` 时，改用')
a=s.index('2. **人体二维姿态。**');b=s.index('5. **二维提升为三维。**',a)
s=s[:a]+'''2. **联合二维姿态。** 整个单人视角由 `golfpose_golfer_ViTPose_huge.pth` 一次推理，输出 22 个 GolfPose 原生关节；无需 COCO 转换或独立球杆模型。
3. **热图解码。** 使用原图与水平翻转输入，按人体左右对称关系交换通道后平均，再用 UDP/DARK 解码回原图坐标。
4. **组织 17+5 输入。** 保留全部原始坐标与置信度。低于 0.10 或位于画面外的点标记为无效，仅在提升模型输入中补值，再传给 MixSTE2 17+5。
'''+s[b:]
a=s.index('视频逐帧复用人物检测');b=s.index('用球杆角度/投影长度',a)
s=s[:a]+'''视频逐帧使用同一个 22 点 golfer-with-club ViTPose 模型，直接保存人体与球杆点。默认 refinement 在确定时间窗口之前，
'''+s[b:]
a=s.index('Both `python workflow.py references`');b=s.index('Output: `runs/workflows/references/`',a)
s=s[:a]+'''Both `python workflow.py references` and `python -m golf_workflows.standard_frames`
default to the joint `golfpose_golfer_ViTPose_huge.pth` checkpoint and MixSTE2
17+5. All 22 native output channels feed the existing lifting/fusion pipeline.
To select this route explicitly:

```powershell
python workflow.py references --pose-backend golfpose-golfer-vitpose-huge --lifter 17+5
```

'''+s[b:]
s=s.replace('python workflow.py references --club-crop-backend official-yolox-2cls','python workflow.py references --pose-backend vitpose-huge --club-crop-backend official-yolox-2cls')
s=s.replace('python workflow.py video <video.mp4> --club-crop-backend official-yolox-2cls','python workflow.py video <video.mp4> --pose-backend vitpose-huge --club-crop-backend official-yolox-2cls')
s=s.replace('The `segmentation` crop backend remains available as an explicit reference\noption for A/B comparison. Video generation still defaults to segmentation crops.','The `segmentation` crop backend remains available when a legacy separate-body\npose backend is explicitly selected. The joint default uses full-image crops.')
a=s.index('`video` performs body inference');b=s.index('Refinement uses angular gap filling',a)
s=s[:a]+'''`video` runs joint 22-point GolfPose ViTPose inference, phase identification,
MixSTE2 17+5 lifting, and P1-P10 selection. Both the launcher and standalone
module default to `golfpose-golfer-vitpose-huge`. The body and club coordinates
come from the same pose model; existing club refinement remains enabled by
default in the launcher.
'''+s[b:]
s=s.replace('`--club-pose-backend segmentation --lifter 17+2` restores the earlier\ntwo-proxy club input.','`--pose-backend yolo26n-pose --club-pose-backend segmentation --club-crop-backend segmentation --lifter 17+2` restores the earlier\ntwo-proxy club input.')
p.write_text(s,encoding='utf8')
