# Golf posture workflows

Run each stage with one command from `J:\Golf_Posture`:

```powershell
python workflow.py references
python workflow.py video pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4
python workflow.py score
```

These commands build ten references, identify ten video poses, then score the
saved poses against the references. Scoring defaults to the latest completed
reference/video workflows. Use `python workflow.py <stage> --help` for options.

Shared code is in `golf_workflows/`; generated files are in `runs/workflows/`.
The new entry point defaults to ViTPose-Huge, GolfPose 17+5 and optional club
refinement enabled for video. The three extra clubhead points remain unmeasured.
See [WORKFLOW.md](WORKFLOW.md) for output contracts, reviewed-frame overrides,
cached-run reuse and the exact scoring inputs. Existing model files and earlier
runs are preserved. The notes below describe earlier experiments.

---

# Attempt 1

Pose prediction on the images in `images/` using Ultralytics YOLO26n-pose (`yolo26n-pose.pt`), the COCO-keypoint model described in the [pose estimation docs](https://docs.ultralytics.com/tasks/pose).

## How to run

```powershell
pip install ultralytics
python predict_pose.py
```

The script loads every image under `images/`, draws the predicted skeleton, and writes:

- annotated images in `runs/pose/predict/`
- joint coordinates in `runs/pose/predict/keypoints.json`

Each person record has the 17 COCO joints (nose through both ankles), pixel `x`/`y`, per-joint confidence, and the person box.

## Outcome

| Item | Result |
| --- | --- |
| Images predicted | 178 |
| Inference | 22.6 ms per image at 640×640 |
| One person | 172 images |
| Three people | `pro_part_01.png` through `pro_part_04.png` |
| Eight people | `pro_sequence.jpg` and `pro_sequence.png` |
| Keypoints | 3,400 |
| Keypoints below 0.5 confidence | 716 |
| People with a wrist below 0.5 | 106 |
| Keypoint confidence | min 0.006, median 0.977, mean 0.787 |

Single-golfer swing frames are detected reliably. The pro reference sheets are collages, so the model returns a skeleton for every figure in the sheet. Median joint confidence is high, but wrists and other occluded joints drop out often enough that a swing angle built from those points would be unstable.

## Possible future improvements

- Split `pro_part_*` and `pro_sequence` into one golfer per image, or keep only the intended person (largest box, or the figure nearest the image center). The current JSON treats every detected figure as a golfer.
- Drop or interpolate joints under a confidence cutoff, especially wrists. 106 of the detected people have at least one weak wrist, which is the joint a club-shaft estimate depends on.
- Track one golfer across each swing sequence and smooth joint paths over time. Each frame is predicted on its own, so a joint can jump when the far arm is hidden behind the body.
- Compute golf measures from the keypoints that are already saved: spine tilt, shoulder tilt, hip line, knee flex, and lead-wrist position. The JSON currently stops at raw coordinates.
- Map COCO left/right to lead and trail side once the camera view (face-on or down-the-line) and the golfer’s handedness are known.
- Move from YOLO26n-pose to a larger pose checkpoint when joint placement matters more than the current 22.6 ms. The nano model is the lowest-accuracy pose variant in that family.
- Raise `imgsz` above 640, or crop to the golfer before inference, so wrists and ankles keep more pixels.
- Add club, grip, and ball points. The 17 COCO joints do not include the club, so posture relative to the shaft is not measured yet.
- Deduplicate repeated swing copies in `images/` before comparing attempts, so the same frames are not counted more than once.

# Swing plane trace

`trace_swing_plane.py` draws the smoothed clubhead path on one video, from the first accepted shaft through the last. The club mask comes from the trained YOLO26n-seg weights. YOLO26n-pose supplies the hands so the grip end of that mask can be separated from the clubhead.

The script expects `runs/segment/golf_club_train/weights/best.pt`, produced by `prepare_golf_club_seg.py` and `train_golf_club_seg.py`, and the pose checkpoint `yolo26n-pose.pt`.

## How to run

```powershell
pip install ultralytics
python -m golf_workflows.trace_swing_plane
python -m golf_workflows.trace_swing_plane user_swings\1785648828688618000_swing.mp4
python -m golf_workflows.trace_swing_plane user_swings\1785648828688618000_swing.mp4 --mediapipe-wrists
```

With no path, the script picks a random video under `user_swings/`. Accepted extensions are `.mov`, `.mp4`, `.avi`, `.mkv`, and `.webm`.

## Workflow

Each frame is handled in one pass. The path on screen is only what has been accepted up to that frame.

1. **Pose.** YOLO26n-pose reads the video at `imgsz` 640. If several people are found, the largest box is the golfer.
2. **Hands.** The anchor is the mean of the wrists whose confidence is at least 0.25. If both wrists are weak, the shoulders are used. If those are missing too, the center of the golfer’s box is used.
3. **Club.** The trained segmentation model runs on the same frame (`conf` 0.25, retina masks, at most 3 detections). Only boxes whose class name is a golf club are kept.
4. **Shaft.** The club contour’s major axis is the shaft. Contours shorter than 12 pixels are ignored. A contour whose major/minor variance ratio is below 8 is treated as a clubhead point only, with no grip, so a round false mask does not invent a shaft.
5. **Ends.** The shaft end nearer the hands is the grip. The other end is the clubhead. With no hands and no earlier clubhead, the lower end is the clubhead.
6. **Choice.** When more than one club mask is present, the winner is the longer, more elongated detection whose grip sits nearer the hands.
7. **Stability.** A clubhead that jumps more than 45% of the frame diagonal within 6 frames is dropped. On the next frame, the new shaft is blended toward the previous one (72% new) so the plane does not jitter. A gap longer than 6 frames starts a new segment and is not joined to the earlier path.
8. **Phases.** After the clip is read, `swing_phases.py` labels address, top, impact, and follow-through from the clubhead path (wrist height only if the club is missing). The path color runs through those phases instead of a single start-to-finish ramp.
9. **Draw.** The video is written on a second pass so every frame can use the phase labels. The video’s frame rate comes from the pose reader, falling back to 30.

## What is drawn

Color runs from blue at the first sample to red at the latest sample.

- A smoothed clubhead path for linked samples. Each step is drawn as eight short sections of a curve, or as straight sections when a curve cannot be fit. A ring marks the first clubhead.
- A filled dot on the current clubhead, when this frame produced a sample.
- A small legend labeled start and finish.

## Output

Files are written under `runs/swing_plane/`, using the source video’s stem:

- `*_swing_plane.avi` — the clubhead path building from the start of the clip to the end
- `*_swing_plane.jpg` — the last frame, with the full path
- `*_swing_plane.json` — phase frame indexes, plus one record per accepted sample: frame index, confidence, whether it continues the previous sample (`linked`), grip `[x, y]` or `null`, and clubhead `[x, y]`
- `*_phases.png` — smoothed clubhead height and speed, with the phase lines marked

## Example

| Item | Result |
| --- | --- |
| Source | `user_swings/1785648828688618000_swing.mp4` |
| Frames written | 66 |
| Swing-plane samples | 53 |
| Video | `runs/swing_plane/1785648828688618000_swing_swing_plane.avi` |
| Still | `runs/swing_plane/1785648828688618000_swing_swing_plane.jpg` |

# GolfPose 3D

`golf_pose_3d.py` lifts each swing video to a 3D golfer and club with GolfPose-3D, the MixSTE2 model from [GolfPose](https://github.com/MingHanLee/GolfPose) (ICPR 2024). The 2D input comes from the same Ultralytics models used above: YOLO26n-pose for the golfer and the trained YOLO26n-seg weights for the club. The GolfPose MMPose and MMDet 2D models are not used. Their environment is Linux-only and built for CUDA, while this project runs on Windows on CPU.

## Setup

```powershell
pip install ultralytics einops timm==0.9.16 local-attention==1.9.0 mediapipe scipy
```

- The model code is vendored unchanged in `third_party/golfpose/common/`, with the GolfPose `LICENSE` and the source commit in `third_party/golfpose/SOURCE.txt`. The file imports the older `timm.models.helpers` API, so timm is pinned to 0.9.16.
- Put the checkpoint at `golfpose_checkpoints/golfpose_17+2_33.6_30.5_59.5.bin`. It is the GolfPose-3D (GC) N=19 row in the GolfPose README and is about 406 MB.

## How to run

```powershell
python -m golf_workflows.golf_pose_3d
python -m golf_workflows.golf_pose_3d user_swings\1785648828688618000_swing.mp4
python -m golf_workflows.golf_pose_3d --all
python -m golf_workflows.golf_pose_3d user_swings\1785648828688618000_swing.mp4 --mediapipe-wrists
```

With no path, a random video under `user_swings/` is used. `--all` processes every video there and loads the models once.

## Workflow

1. **2D detection.** Each frame goes through YOLO26n-pose and the club segmentation model. The golfer, hands, club choice, and jump filter are the ones in `trace_swing_plane.py`, so the 2D clubhead matches the swing-path script.
2. **Keypoint conversion.** GolfPose expects 17 body joints in Human3.6M order, followed by club points. COCO joints are converted as follows:
   - Hips, knees, ankles (as feet), shoulders, elbows, and wrists map directly.
   - `root` is the midpoint of the hips, `thorax` is the midpoint of the shoulders, and `spine` is the midpoint of `root` and `thorax`.
   - `neck_base` is halfway from `thorax` to the nose. `head` is 25% beyond the ears and eyes, measured away from `thorax`.
   - The club’s grip end becomes `shaft` and its clubhead end becomes `hosel`, which gives the 17+2 layout.
   - Joints below 0.25 confidence are treated as missing.
3. **Gap filling and smoothing.** Missing joints and club points are filled by linear interpolation over time, and the first or last good value is held at the ends of the clip. Each filled joint is then smoothed with a short moving average (about 0.15 s, same idea as GolfPosePro’s wrist filter) before the 3D model sees it.
4. **Phases.** Address, top, impact, and follow-through come from `swing_phases.py`. The 3D lift uses only that window, so idle frames before address and after the finish are not padded into the model.
5. **Normalization.** GolfPose was trained on 1280x720 cameras. The swing is centered on the golfer and scaled so the median golfer box is 70% of a virtual 1280x720 canvas. It is then normalized as in GolfPose: x and y are divided by the width, multiplied by 2, and shifted by `[1, h / w]`.
6. **3D lift.** MixSTE2 runs on CPU with 243-frame windows, at about 3 seconds per window:
   - Clips shorter than 243 frames are padded by repeating the last frame.
   - Longer clips use more windows, with the last one aligned to the end of the clip.
   - Each window is averaged with its mirror (x negated and left and right swapped), as in GolfPose evaluation.
   - The output is root-relative, in meters, in camera axes.
7. **Render.** The left panel is the video frame with the converted 2D skeleton and the smoothed 2D clubhead path, colored by swing phase. The right panel is a 3D view with the skeleton, the club as a shaft-to-hosel line, and the 3D hosel path. Frames outside the phase window are left blank on the 3D side.

`--mediapipe-wrists` on either script fills a YOLO wrist below 0.25 confidence from MediaPipe Pose when that landmark is visible. It does not replace the pose model or the club segmentation.

## Output

Files are written under `runs/golf_pose_3d/`, using the source video’s stem:

- `*_3d.avi` — the side-by-side video
- `*_3d.jpg` — the last frame of that video
- `*_3d.npz` — `joints` (19 names), `keypoints_2d` and `keypoints_2d_valid` (pixels and the per-joint detected flag), `keypoints_2d_filled`, `keypoints_2d_normalized` (the model input), `keypoints_3d` (frames × 19 × 3, meters), `fps`, `frame_size`
- `*_3d.json` — the checkpoint, frame counts, normalization values, frames with a detected club, interpolated frames per joint, and the mean and standard deviation of every 3D bone length

## Example

| Item | `1785648828688618000_swing.mp4` | `1785642572528296000_swing.mov` |
| --- | --- | --- |
| Frames | 66 | 331 |
| Address / top / impact / finish | 11 / 27 / 44 / 54 | 144 / 160 / 177 / 187 |
| 3D window | frames 11–54, one padded window | frames 144–187, one padded window |
| Club detected | 53 frames | 270 frames |
| Forearm length variation in that window | about 2 cm | about 2 cm |
| Shaft-to-hosel length | 0.54 ± 0.19 m | 0.54 ± 0.19 m |

The long clip’s speed peak is frame 177, the same frame chosen as impact. Frames outside that window stay in the 2D path and are left blank in the 3D view.

## Limitations

- GolfPose-3D was trained on ground-truth 2D from its own motion-capture set. Here, `neck_base`, `head`, and `spine` are estimated from COCO joints and YOLO adds noise, so expect more error than the paper’s 33.6 mm. `neck_base` is the least stable joint when the nose is hidden or the golfer looks down at the ball.
- The 3D shaft is shorter than a real club (0.6–0.8 m against about 1.1 m for a driver). The segmentation grip end may not sit where the GolfPose `shaft` marker was placed.
- No heel or toe points are available, so the clubhead is the hosel point.
- Output is root-relative. Overall position, and the absolute scale from the camera, are not measured.
- A later upgrade is an Ultralytics pose model fine-tuned on the GolfPose 22-keypoint data, which would supply heel and toe points and allow the 17+5 checkpoint.

# Swing phases

`swing_phases.py` is the temporal layer adapted from [GolfPosePro](https://github.com/ryanboscobanze/GolfPosePro). That notebook tracks a MediaPipe wrist and cuts the swing from its height and speed. Here the same cuts use the segmented clubhead, because a wrist is only a stand-in for the club.

- Smooth the path with `scipy.ndimage.uniform_filter1d` (about 0.15 s).
- Speed is `np.gradient` of that path.
- **Address** is the frame before speed stays above 12% of the peak.
- **Top** is the clubhead farthest from the address position, before impact.
- **Impact** is the fastest frame that has returned near the address position.
- **Follow-through** is where speed falls below 35% of the impact speed.

Wrist height is used only when fewer than eight clubhead samples exist. `--mediapipe-wrists` is optional and only patches weak YOLO wrists.


## Experimental 17+5 lifter

`golfpose_17+5_39.2_32.3_62.8.bin` is a 22-joint MixSTE2 checkpoint: 17 body joints plus `shaft`, `hosel`, `heel`, `toe_down`, and `toe_up`. Its learned spatial embedding is `[1, 22, 512]`, and its temporal embedding is `[1, 243, 512]`. It cannot accept a 15+5 / 20-point tensor or the existing 19-point tensor directly.

Both lifter sizes are supported. The original 17+2 checkpoint remains the default and its file is preserved. The 17+5 mode has a separate default output directory.

```powershell
python -m golf_workflows.standard_frames --lifter 17+5 --input-dir pro_refs/Standard_10_frames --output-dir runs/standard_frames_17plus5_20261027
python -m golf_workflows.golf_pose_3d user_swings/<swing>.mp4 --lifter 17+5 --output-dir runs/golf_pose_3d_17plus5

# Original model and reference generation:
python -m golf_workflows.standard_frames --lifter 17+2
```

This is an **imputed-input experiment**, not a complete 22-landmark detector upgrade. The existing YOLO26n-pose and club segmentation supply the same 17+2 inputs. Three unavailable clubhead inputs (`heel`, `toe_down`, `toe_up`) are stored as NaN with detection flags False; only the filled model input places them at the hosel. Their predicted 3D positions are model estimates conditioned on that fill, not measured clubface geometry. Clubhead orientation inferred from them is not reliable.

A trial of the locally saved 22-keypoint DEKR model on the P1 face-on reference returned very low-confidence, out-of-golfer predictions. It was not incorporated as a source of measured landmarks. The DEKR checkpoint file remains available.

The new reference run writes all 22 predicted joints, separate camera and fused reference renderings, raw 2D points, per-joint validity, filled 2D inputs, and imputation metadata. Reference fusion uses the same face-on/down-the-line method as before, with alignment chosen from the existing body/shaft bones. `runs/standard_frames/` is preserved and remains the scoring default.

Official joint schema: https://github.com/MingHanLee/GolfPose/blob/main/configs/mmpose/_base_/datasets/golfswing_golfer.py


## ViTPose-Huge 2D backend with the 17+5 lifter

The saved COCO ViTPose-Huge checkpoint is available through `--pose-backend vitpose-huge` in both `standard_frames.py` and `golf_pose_3d.py`. The existing `yolo26n-pose` backend remains the default.

```powershell
python -m golf_workflows.standard_frames --pose-backend vitpose-huge --lifter 17+5 --input-dir pro_refs/Standard_10_frames --output-dir runs/standard_frames_vitpose_huge_17plus5_20261027
python -m golf_workflows.golf_pose_3d user_swings/<swing>.mp4 --pose-backend vitpose-huge --lifter 17+5 --output-dir runs/golf_pose_3d_vitpose_huge_17plus5
```

The preserved YOLO26n-pose model locates the golfer's box. ViTPose-Huge replaces its body keypoints using a padded UDP crop at 256x192, RGB normalization, original/mirrored heatmap averaging, and MMPose's UDP/DARK decoder. Video frames use box-IoU association to follow the same golfer, with largest-box selection when no match exists. Single reference pictures are processed independently.

`vitpose_tracking.py` implements the checkpoint's 32-layer, 1280-dimensional transformer and heatmap head with native PyTorch CPU operators. All released state tensors load strictly; training-only dropout is disabled. The unused class-token parameter is preserved, and patch positions use the checkpoint's slots 1 onward. This avoids requiring MMPreTrain or MMCV compiled operators in the current Windows environment. The ViTPose backend defaults to four PyTorch CPU threads; the video CLI can override this with `--pose-cpu-threads`. Decoder source attribution and license are in `third_party/golfpose/VITPOSE_SOURCE.txt` and `MMPOSE_LICENSE`.

The 17 COCO body landmarks are converted to the existing GolfPose body order. The current trained segmentation model supplies shaft/hosel inputs. **ViTPose does not detect clubhead heel/toe landmarks**: those three missing 2D inputs remain NaN/False in the raw records and are filled at the hosel only for the 17+5 model input. Those 3D clubhead predictions remain experimental.

Each reference view's `joints.json` records ViTPose's raw COCO coordinates, confidence responses, crop affine matrix, inference time, and the converted GolfPose joints. `joints.npz` stores 22-joint 2D/3D arrays and validity/filling information. The new run also provides comparison overviews and validation records. Existing reference folders and checkpoints are preserved; no new reference set is automatically selected for scoring.


For the reviewed `b85ed0b4db566e63bce190824c2f2196.mp4` video trial:

```powershell
python -m golf_workflows.golf_pose_3d pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4 --pose-backend vitpose-huge --lifter 17+5 --pose-cpu-threads 8 --positions-json runs/golf_pose_3d/b85ed0b4db566e63bce190824c2f2196/selected_frames.json --output-dir runs/golf_pose_3d_vitpose_huge_17plus5
```

`--positions-json` reuses the visually reviewed zero-based P1-P10 indices while calculating fresh 2D and 3D poses. It validates the source filename, coverage, ordering and frame bounds. Reviewed impact is retained rather than replaced by ball proximity. Video timestamps describe the edited slow-motion source. Omitting this option keeps automatic position selection.

`--pose-cpu-threads` defaults to four. A timing check on the current CPU selected eight for this video; the heatmap differences from four threads were below `4e-7`. Source checkpoints and prior outputs are preserved. The video trial is saved separately under `runs/golf_pose_3d_vitpose_huge_17plus5/`.
