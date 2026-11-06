# Joint GolfPose workflow

Run from `J:\Golf_Posture` with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m golfpose_workflow --reference
.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/w07.mp4
.\.venv\Scripts\python.exe -m golfpose_workflow --score
```

This workflow runs fresh 2D pose inference with one pose checkpoint:
`golfpose_checkpoints/golfpose_golfer_ViTPose_huge.pth`. Its native 22 channels
contain 17 body joints followed by shaft, hosel, heel, toe_down, and toe_up.
Before pose inference, `golfpose_checkpoints/golfpose_detector_1cls_yolox_s.pth`
locates the single `golfer` class. The selected box supplies the 192x256 UDP crop
with the existing 1.25 expansion. UDP/DARK decoding maps coordinates directly into
the original image; flip averaging, confidence >= 0.10 and in-frame validity remain.

Lifting uses MixSTE2 **17+2** (`golfpose_17+2_33.6_30.5_59.5.bin`): the 17 body
joints plus `shaft` (grip-side marker) and `hosel`. The heel, toe_down and toe_up
channels and their confidences are dropped before lifting; saved arrays are
19-joint, and `<stem>_3d.json` lists them under `dropped_2d_landmarks`. Reasons:

- On `010.mp4` the golfer crop maps about 13 source pixels to one heatmap cell, while
  the driver head is about 25 px wide. The three head points had the lowest median
  confidence (0.58-0.63), and their 95th-percentile distance from the hosel was
  about 250 px; they often spread across the frame in the downswing and finish.
- Phase detection, club refinement and scoring use only shaft and hosel.
- The GolfPose 17+2 lifter has lower reported error than 17+5
  (MPJPE 33.6 vs 39.2 mm; body 30.5 vs 32.3; club 59.5 vs 62.8).
- GolfPose's alternative club route (two-class YOLOX club box + club-only
  ViTPose) was tested on ten `010.mp4` frames and was worse: the club class was
  found in 3/10 frames at scores 0.01-0.08, one of them on the wrong object,
  and golfer-box fallbacks gave collapsed shaft/hosel pairs.

Shaft and hosel still come from the 22-joint model and can be wrong in blurred
downswing frames; club refinement interpolates low-confidence pairs.

For video, overlapping detections retain the golfer identity; on detector gaps,
pose inference uses the last detected ROI and records `tracked-detector-roi`.
An initial detector miss produces no observed pose. Video never falls back to a
full-frame pose crop. Still reference panels can fall back to the complete panel
when detection fails, recorded as `reference-panel-fallback`. The detector and
pose confidence values remain separate; a tracked ROI is not marked as a new detection.
There is no separate club pose model in this workflow.

By default an automatic golfer ROI is derived first: a YOLO26n pass over 12
sampled frames, union of the agreeing golfer boxes, expanded for the club.
It is used only where YOLOX misses, recorded as `auto-golfer-roi`, and disabled
with `--no-auto-golfer-box`. `pro_refs/w09.mp4`, where YOLOX never fires, now runs
without a manual box. To override it, initialize the ROI explicitly:

```powershell
.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/010.mp4 --golfer-box 140 220 500 900 --view face_on
```

The rectangle is `(x1, y1, x2, y2)` in original decoded-frame pixels. These values
were verified through the full 252-frame `010.mp4` run at 592x1280; choose a suitable box for other
videos. The same 22-joint model processes this crop while YOLOX keeps checking
for detections. Crop metadata distinguishes `manual-golfer-roi` from a detector
observation or a retained detected ROI. This option initializes the ROI without
using a full-frame video pose fallback.

On root failure, `pose_inference_diagnostics.json/.npz` are saved in the video
output folder. They distinguish zero pose inference due to missing detector
boxes from pose inference whose root failed the confidence/image-bound tests.
The full ROI-initialized `010.mp4` run produced observed roots in all 252 frames
and exported ten finite 22-joint poses. Outputs are in
`runs/golfpose_workflow/videos/010_roi_20261104/`; validation details are in
`runs/diagnose_010_20261104/validation.json`. All 72 tests pass.

The detector revision passes all 67 tests. On `w07.mp4`, one-class detection was
available in 7/53 frames; the other 46 used the retained detector ROI. Cropped
inference completed, but its lifted shaft failed the existing P5 threshold
(minimum vertical component about 0.357 versus the required <= 0.35).
P6 consequently remained unavailable, and completion validation rejected that
automatic ten-frame export. The rule is unchanged; review the predictions before
supplying `--positions-json` for this sample. Diagnostics are in
`runs/validation_detector_routing_20261103/pipeline_validation.json`.
MixSTE2 17+2 lifts those predictions into 3D. Video ball timing correction remains
available through the shared pipeline when its optional ball model is installed.
Its CLIP tokenizer dependency is installed in the project environment and pinned
in `requirements.txt` to Ultralytics CLIP commit
`7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40`.

Code and instructions are in `golfpose_workflow/`. Results are collected separately:

```text
runs/golfpose_workflow/
  latest_references.json
  latest_video.json
  latest_score.json
  references/                  ten reference folders and three combined previews
  videos/<video-stem>/          full tracking results and ten extracted frames
  scores/<video-run-name>/      score.json and comparison pictures
```

Repeated runs get new timestamped folders. Completed results are preserved.
The original `workflow.py` retains its YOLO26n default and its own `runs/workflows/`
latest pointers. `--score` here uses only this workflow's pointers by default;
`--run` and `--references` can explicitly select other completed folders.
`--output-dir` can explicitly choose an empty output folder for any stage.

References use the existing whole-skeleton confidence selection between face-on
and down-the-line, plus independent whole-club donor selection and the shaft-only
fallback when all eligible club observations are incomplete. Rear remains diagnostic.

Video uses the same phase and extraction conditions as the original workflow,
following the [P classification](https://hackmotion.com/golf-swing-positions/#Understanding_Golf_Swing_Positions_in_the_P_Classification_System)
(see `WORKFLOW.md`, 2026-11-04 P-position update):

- P1 address, P4 top, P7 impact come from body-relative wrist and observed club
  motion, with optional ball timing. P1 is the last reliable clubhead rest
  before the takeaway (fallback: hand motion onset). P4 is the 2D shaft's
  turn near the wrist peak when it is seen on both sides (fallback: wrist peak).
  Club frames count as reliable when shaft and hosel are detected with
  confidence >= 0.3. The swing window ends when the raised
  finish hands come back down (or at the clip end), not at the first pause.
- P3 is the lead wrist rising through lead-shoulder height before top. Of the two
  frames around the crossing, the one closer to horizontal is used; if there is
  no crossing, the first frame with the arm's vertical component <= 0.25 is used.
  P2 is the reliable shaft nearest horizontal between address and P3 (within
  about 20 degrees); otherwise the lead arm at 30% of its rotation from P1 to P3.
- P5 mirrors P3: the lead wrist coming down through shoulder height after the top.
  P6 is chosen like P2, between P5 and impact (arm fallback measured from P7).
- P9 mirrors P3 after impact: the lead wrist rising through shoulder height
  (fallback: arm nearest horizontal before the hands' highest point).
- P8 has both arms fully extended: the largest mean elbow angle between impact and P9.
  Only frames where the hands have risen a quarter of the way to their P9 height count;
  near-ties go to the shaft nearest horizontal.
- P10 is the frame with the lead wrist nearest the head, with the wrist above the
  lead shoulder. It is searched only while the hands stay within 5% of their
  highest point, before they are lowered. Missing evidence stays missing.

Club refinement and window blending are enabled by default. Extraction reserves
space between neighboring positions, and completion requires ten distinct,
increasing indices, finite 19-joint poses and their matching source images.
There is no replacement with ten evenly spaced frames. If automatic evidence is
insufficient, the command reports the failure without publishing a completed run;
review indices and retry with `--positions-json <file>` when needed.

Each video exports `P1.jpg` through `P10.jpg`, ten matching `P*_3d.jpg`,
`identified_frames.npz/json`, full-sequence 2D/3D arrays and confidence, and the
rendered tracking video. Source indices are zero-based. Inferred club markers
remain distinguished from observed points.

Scoring calls the exact existing comparator, with the same 14 posture criteria,
handedness, normalization, yaw alignment and angular-score rules. It uses the
saved P1-P10 coordinates without selecting frames again. See
[SCORING_METHODOLOGY.md](../SCORING_METHODOLOGY.md) for the scoring formulas.

Validation before the detector revision on 2026-11-03: all 60 tests passed. Fresh inference generated all ten
references and processed all 53 frames of `pro_refs/w07.mp4`. Automatic P1-P10
indices were `7, 15, 18, 27, 32, 33, 34, 35, 37, 39`; all ten exported 22-joint
3D poses were finite. The shared scorer evaluated all 140 posture parameters
and returned 91.4/100. This is a similarity score, not a measured inference
accuracy. Original workflow latest pointers remained byte-identical.
The execution report is `runs/golfpose_workflow/validation.json`.
