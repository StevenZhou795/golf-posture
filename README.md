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
python trace_swing_plane.py
python trace_swing_plane.py user_swings\1785648828688618000_swing.mp4
python trace_swing_plane.py user_swings\1785648828688618000_swing.mp4 --mediapipe-wrists
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
python golf_pose_3d.py
python golf_pose_3d.py user_swings\1785648828688618000_swing.mp4
python golf_pose_3d.py --all
python golf_pose_3d.py user_swings\1785648828688618000_swing.mp4 --mediapipe-wrists
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
