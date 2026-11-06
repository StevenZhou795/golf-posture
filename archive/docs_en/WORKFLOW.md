# Golf posture workflows

## Automatic default golfer ROI (2026-11-05)

Both video workflows now derive a golfer ROI before 2D inference, so a clip where
the GolfPose detector never fires no longer needs `--golfer-box`
(`golf_workflows/golfer_roi.py`, wired in `golf_pose_3d.main`).

1. Read 12 frames evenly spaced over the clip with YOLO26n (`yolo26n-pose.pt`) and
   take the largest person with score >= 0.4 in each.
2. Keep the boxes with IoU >= 0.3 against their median box; at least 3 are needed. This drops
   frames where a spectator or nothing was the largest detection.
3. Take the union and expand it for the club by 0.15 of its height left and right, 0.25
   above (shaft over the head at P4/P10) and 0.05 below. Clip it to the frame.

How each workflow uses it (only where its GolfPose detector misses; a detection always wins):

| Workflow | Detector it backs up | Before | Now |
| --- | --- | --- | --- |
| golfpose_workflow | One-class YOLOX golfer box for the 22-joint ViTPose crop | Misses before the first detection were unobserved, and a clip with no detection failed with "supply --golfer-box" | The ROI initializes the crop (`auto-golfer-roi`); after a detection the last detected box is retained as before |
| `workflow.py` | Two-class YOLOX person box for the YOLO26n body crop | Full frame, where the largest person wins | The ROI crop (`auto-golfer-roi`); the full frame only if there is no ROI |

`--golfer-box X1 Y1 X2 Y2` now works in both workflows (`workflow.py` with the default
YOLO26n + `official-yolox-2cls`) and overrides the automatic ROI. Use `--no-auto-golfer-box`
to turn the automatic ROI off. The chosen ROI is recorded in `<stem>_3d.json`:
`pose_backend.initial_golfer_box_xyxy`, `initial_roi_source` (`manual` or `auto-yolo26n`) and
`initial_roi_details` (sampled, person and agreeing frames, margins). Per-frame crop
sources also appear in the summary.

Reasons:

- **w09 failed in golfpose_workflow.** One-class YOLOX scored at most 0.04 on any
  frame of `pro_refs/w09.mp4`, a portrait phone recording from behind. YOLO26n
  found the golfer at about 0.89 in every sampled frame.
- **The box I picked by hand for w09 was close to what the pass gives.** The
  manual box was `120 330 470 1000`; the automatic ROI is `100.5 307.0 487.7 991.2`
  (12/12 agreeing frames).
- **`workflow.py`'s full-frame fallback picks the largest person.** In w09 the
  two-class YOLOX missed the person in 44 of 46 frames. A golfer-sized crop also gives
  YOLO26n about 1.8x the pixels on the golfer.
- **Cost:** 1-5 s per clip.

Automatic ROIs: w09 `[100.5, 307.0, 487.7, 991.2]`, w07 `[25.0, 240.4, 391.0, 949.8]`,
005 `[110.7, 84.6, 553.8, 876.6]` (10/12 agreeing), 010 `[34.1, 204.6, 518.6, 931.7]`,
008 `[62.3, 160.9, 591.0, 1018.5]`.

Validation (86 tests pass; new cases for ROI union, outlier rejection and clipping,
the automatic ROI bootstrapping the joint sequence, and the body crop using the ROI on
a YOLOX miss). Fresh `pro_refs/w09.mp4` runs:

| Workflow | Before | Now (no `--golfer-box`) |
| --- | --- | --- |
| golfpose_workflow | Failed without `--golfer-box`; with it, `6,13,18,23,26,28,31,32,33,40` | `7,13,18,24,26,28,31,32,33,40` (`runs/golfpose_workflow/videos/w09_autoroi_20261105/`) |
| `workflow.py` | `8,12,18,23,26,28,31,32,34,40` (full-frame body crop) | `8,13,19,23,25,28,31,32,33,40`, 44/46 frames on `auto-golfer-roi` (`runs/workflows/videos/w09_autoroi_20261105/`) |

The `workflow.py` P9 change (34 to 33) comes from the different 2D input. At frame 33
the lead arm is hidden behind the body from down the line, and both crops misplace it.
At 28 fps the arm passes horizontal between frames 33 and 34. Clips where YOLOX
detects the golfer in every frame are unaffected.

## Club-aware P1, P4 and P2/P6 (2026-11-04)

Both workflows (`workflow.py` and `python -m golfpose_workflow`) share
`golf_workflows/swing_phases.py`, so these apply to both whenever club
refinement is on (the default). Before this change, the club only set P7 (clubhead
nearest its address spot, or the ball) and P2/P6 (shaft nearest horizontal, trusting
every lifted shaft). A club frame now counts as "reliable" only when both shaft and
hosel are detected with the weaker landmark confidence >= 0.3
(`RELIABLE_CLUB_CONFIDENCE`; blurred shafts typically score about 0.2).

| Position | Rule before | Rule now |
| --- | --- | --- |
| P1 address | Hands start moving (body-relative wrist leaves its start position), minus 0.1 s | Last reliable frame before the top with the clubhead still at its address spot (within a tenth of its backswing excursion, at least 0.1 torso), plus 1 frame, minus 0.1 s. Needs two reliable rest frames within 0.3 s. Clubhead detections that jump more than 0.6 torso per frame from the last accepted one are ignored. Falls back to the hand rule |
| P4 top | Wrist-height peak | Frame where the 2D shaft (grip to hosel) stops rotating back, searched within 0.15 s of the wrist peak, using reliable frames with a readable projected length (>= 0.4 torso). The turn must be inside the window, with at least 3 degrees of rotation on both sides. Falls back to the wrist peak |
| P2 / P6 | Lifted 3D shaft nearest horizontal (P1-P3 / P5-P7) | Same, but only reliable club frames, and only if that shaft is within 0.35 of horizontal (about 20 degrees). Otherwise the lead arm stands in: 30% of its rotation from hanging (P1 / P7) toward parallel (P3 / P5) |
| P8 tie-break | Shaft nearest horizontal among near-equal extensions | Same, reliable club frames only |

`<stem>_3d.json` records the evidence:
- `phase_detection`: `address_source` (`club_rest` or `hand_motion_onset`), `hand_address`, `club_rest_end`, `top_source` (`club_reversal` or `wrist_peak`), `wrist_top`, `club_top`.
- `checkpoint_selection`: `position_sources` (`observed_shaft` or `lead_arm_progress` for P2/P6), `reliable_club_confidence`, `reliable_club_frames`.

Reasons:

- **P1: the clubhead moves first and furthest.** The hand rule fires on slow hand
  drift at setup. In 005 the hands drift from frame 84 while the clubhead stays on
  the ball (hosel at pixel 528, 795) until about frame 104, so P1 moves 84 -> 108.
  In 008 (face-on) the clubhead leaves the ball at about 93-97 and the hands move
  from about 100. The old P1 (99) already showed the club moving; now it's 87.
  In w09 the old P1 (frame 8) had the clubhead leaving the ball; now it's 6, with the club still.
  "Last rest before the top" skips waggles. The jump filter rejects 008's false
  hosel detections at frames 90-104 (400 px jumps, confidence 0.27-0.44).
- **P4: HackMotion's top is where the club changes direction.** The club can
  keep loading after the hands peak. In practice the 2D club is rarely readable at
  the top. 005, w07 and the YOLO26n run of w09 had no reliable detections there.
  In 010 the grip and hosel swap (130-180 degree flips), and in 008 detections stop
  at 154 while the club is still turning. So the wrist peak remains the usual result.
  The club only overrides it when the reversal is seen on both sides; in this
  validation that happened once (w09 with the joint model, same frame as the wrist peak).
- **P2/P6: a "shaft parallel" decision on an inferred shaft is a guess.** In 008
  the club is invisible through the whole takeaway (frames 100-128). The old P2
  (100) and an unfiltered gate (89) both landed on false shafts. Requiring a
  near-horizontal observed shaft avoids picking "the least bad" frame. The
  fallback uses the lead arm, not hand height: the lifted lead wrist can already
  be level with the pelvis at address (w09: 0.02 torso). The 30% fraction comes
  from 005, where the observed-shaft P2 (117) is at 25% of the arm rotation, and
  from the frames reviewed in 008, where the shaft is parallel at about 119 (28%).

Validation (81 tests pass; new cases for club reversal, club rest with a
misdetection, and P2 fallback). Replays reuse saved 2D through the full
`process` (lifting, refinement and extraction; no ball model, so P7 can differ by a frame):

| Run | Before (P1-P10) | After (P1-P10) | Sources |
| --- | --- | --- | --- |
| 005 golfpose_workflow | `84,133,150,177,198,214,226,242,253,306` | `108,117,150,177,198,214,226,242,253,306` | club rest; P2/P6 observed shaft |
| 010 golfpose_workflow (face-on) | `60,72,108,139,155,165,167,175,180,208` | `56,80,108,139,155,165,167,175,180,208` | club rest; P2 lead arm, P6 observed shaft |
| 008 `workflow.py` (face-on) | `99,100,146,153,165,183,210,223,228,236` | `87,120,146,153,164,185,210,223,228,258` | club rest; P2 lead arm, P6 observed shaft |
| w07 `workflow.py` | `9,15,22,27,30,32,34,35,36,44` | `6,15,22,27,30,31,34,35,36,44` | club rest; P2 observed shaft, P6 lead arm |
| w09 `workflow.py` | `8,9,18,23,26,28,31,32,34,40` | `8,13,19,23,26,28,30,32,33,40` | hand onset; P2/P6 lead arm |
| w09 golfpose_workflow (`--golfer-box`) | `8,15,18,23,26,29,31,32,33,40` | `6,13,18,23,26,28,31,32,33,40` | club rest, top from club reversal; P2/P6 lead arm |
Visual checks:
- **w09:** P1 at frame 6 has the club still (frame 8 had it leaving the ball). P2 at frame 13 has the hands at hip height with the shaft foreshortened toward the camera (parallel to the target line); frame 9 was too early.
- **010:** the shaft is horizontal at about 80-84; at 72 it still points down.
- **008:** P2 (120) is inside the reviewed 116-122 range.

Fresh end-to-end runs of `pro_refs/w09.mp4` (with ball):

| Workflow | P1-P10 | Sources | Output |
| --- | --- | --- | --- |
| `workflow.py` (YOLO26n + club ViTPose, 17+2) | `8,12,18,23,26,28,31,32,34,40` | hand onset (club not reliable at rest); top from wrist peak; P2/P6 lead arm | `runs/workflows/videos/w09_club_20261104/` |
| golfpose_workflow (`--golfer-box 120 330 470 1000`) | `6,13,18,23,26,28,31,32,33,40` | club rest; top from club reversal (agrees with the wrist peak); P2/P6 lead arm | `runs/golfpose_workflow/videos/w09_club_20261104/` |

Remaining limits:
- The joint model misplaces the lead wrist at w09 frame 33, so that workflow's P9 is 33 rather than 34.
- In `workflow.py` the club ViTPose has only 3 reliable club frames (of 16) in the w09 backswing, so P1 falls back to the hand onset there.
- References and scores built earlier keep their old P1/P2/P4/P6 frames; rebuild them to compare like for like.

## Arm-based P8/P9/P10 recognition (2026-11-04, w09)

This refines the update below (same file, `find_checkpoint_frames`). In `pro_refs/w09.mp4` (28 fps), P7-P9 were
frames 31/32/33 and P10 was 45. P9 still had the hands at hip-to-chest height with
the arm below parallel. P10 came after the golfer had dropped the club.

| Position | Rule before | Rule now |
| --- | --- | --- |
| P3 | First frame with the lead arm within 0.25 of horizontal before P4 | The lead wrist rising through lead-shoulder height before P4; of the two frames around the crossing, the one closer to horizontal. Falls back to the 0.25 band |
| P5 | First lead-arm parallel (0.25 band) after P4 | The lead wrist coming down through shoulder height after P4, same crossing rule |
| P8 | Shaft nearest horizontal between P7 and P9 | Arms fully extended: largest mean elbow angle (shoulder-elbow-wrist, both arms) between P7 and P9. Only frames where the hands have risen a quarter of the way from their impact height to P9 count. Frames within 5 degrees of the best are tied; the shaft nearest horizontal breaks the tie |
| P9 | First lead-arm parallel (0.25 band) after P7 | Arms rising: the lead wrist rising through shoulder height after P7, same crossing rule. Fallback is unchanged |
| P10 | Lead wrist nearest the head after P9, wrist above shoulder | Same test, limited to the finish: the frames around the hands' highest point after P9 that stay within 5% of it (relative to the rise since impact) |

Reasons:

- **The 0.25 band admits arms well below parallel.** A vertical component of 0.25
  is about 14 degrees. At low frame rates the first frame inside the band can
  still have the hands below the shoulder. In w09 frame 33 has the arm at -0.22 (below)
  and frame 34 at +0.16 (above), so P9 is now 34. The same test was applied to P3
  and P5, which "mirror" P9. In 005, P3 moves from 141 (arm about 14 degrees
  below) to 150 (horizontal).
- **P8 is about the arms, not only the shaft.** After impact the shaft is the most blurred
  part of the picture (005 hosel confidence about 0.2 for frames 236-246). Extension alone
  is not enough either: the arms are just as straight at impact (005: 162-165
  degrees at frames 226-231) as at the release (165 at 242). The "hands have
  started up" gate separates them.
- **P10 picked frames where the club was being lowered.** As the hands come down
  in front of the face, the wrist gets closer to the head than at the real finish. In w09 the nearest
  frame was 45 (hands at the chest); in w07 it was 52, the clip end. The finish is
  now tied to the hands' highest point.

Validation (78 tests pass; new synthetic case for P8 extension, P9 crossing and P10
before lowering):

| Run | Before (P1-P10) | After (P1-P10) |
| --- | --- | --- |
| w09, `workflow.py` fresh run | `8,9,17,23,25,26,31,32,33,45` | `8,9,18,23,26,28,31,32,34,40` |
| 005, golfpose_workflow replay | `84,133,141,177,195,214,226,248,250,346` | `84,133,150,177,198,214,226,242,253,306` |
| 010, golfpose_workflow replay | `60,72,103,139,153,165,167,175,178,219` | `60,72,108,139,155,165,167,175,180,208` |
| w07, `workflow.py` saved 3D | `9,15,20,27,29,32,34,35,36,52` | `9,15,22,27,30,32,34,35,36,44` |

Visual checks:

- w09 P8 (32) shows both arms straight just past the ball, P9 (34) the hands at shoulder height and
  rising, and P10 (40) the hands high with the club behind the head. Frame 41 onwards the club is dropped.
- 005 P8 (242) shows extended arms at hip height, and P9 (253) the hands at chest height with the club up.

Output: `runs/workflows/videos/w09_p8p9_20261104/`. References and scores built with the
earlier rules keep their old P3/P5/P8/P9/P10 frames; rebuild them to compare like for like.

## P-position extraction and swing window update (2026-11-04)

Reference: [HackMotion, Golf Swing Positions: the P Classification System](https://hackmotion.com/golf-swing-positions/#Understanding_Golf_Swing_Positions_in_the_P_Classification_System).
These rules are shared by `workflow.py` and `python -m golfpose_workflow`
(`golf_workflows/swing_phases.py`).

| Position | HackMotion definition | Rule before | Rule now |
| --- | --- | --- | --- |
| P1 | Setup before the swing begins | Start of body-relative hand motion minus 0.1 s | Unchanged |
| P2 | Takeaway ends with the club parallel to the ground | Shaft nearest horizontal between P1 and P3 | Unchanged |
| P3 | Lead arm parallel to the ground | First lead-arm parallel before P4 | Unchanged |
| P4 | Highest point of the backswing | Corroborated wrist-height peak | Unchanged |
| P5 | Early downswing, before the shaft reaches parallel | First frame after P4 with the shaft within 0.35 of horizontal and the wrist above the hip | First lead-arm parallel after P4 (mirror of P3) |
| P6 | Shaft parallel to the ground before impact | Shaft nearest horizontal between P5 and P7 | Unchanged (now after the corrected P5) |
| P7 | Impact | Wrist return, refined with club/ball | Unchanged |
| P8 | Release: shaft parallel after impact, arms extended | Shaft nearest horizontal between P7 and P9 | Unchanged |
| P9 | Lead arm parallel on the lead side; "opposite of P3" | Lead arm nearest horizontal anywhere after P7 | First lead-arm parallel after P7; fallback: nearest horizontal before the hands' highest point |
| P10 | End of swing: stable, facing the target, hands up by the lead ear | Lead wrist nearest the head after P9, on the lead side of the head (3D dot product) | Lead wrist nearest the head after P9 while the lead wrist is above the lead shoulder |

Swing window end: before, the window closed 0.15 s after the first 0.25 s of quiet,
raised hands. Now it stays open from that first quiet hold until the raised
hands come back down (below the midpoint between impact and top height for
0.1 s), or until the clip ends. `phase_detection` in `<stem>_3d.json` records
`finish_hold_start`, `hands_lowered_at` and `finish_reason`
(`raised_hands_lowered`, `clip_end_hands_still_raised` or `clip_end_no_sustained_hold`).

Reasons:

- **005 (golfpose_workflow) ended too early.** The first quiet moment (about frame 275)
  is the pause where the hands stop rising and start wrapping around the head.
  The window closed at 285. The held finish (lead wrist nearest the head, almost
  still) is at about frames 320-360, and the hands only drop at about 405.
  P10 could only land one frame after P9 (253 then 254).
- **The P10 lead-side test excluded the finish.** The 3D dot product included
  height: with the hands above the head and the lead shoulder below it, the sign
  flips. Even horizontally the sign hovers around zero as the hands wrap behind
  the head. "Hands up" (lead wrist above the lead shoulder) matches HackMotion's
  description and is stable.
- **The P5 shaft test fired at the top.** HackMotion's P4 has the club parallel
  to the ground at the top of a full swing, so "shaft near horizontal after P4"
  was satisfied immediately. 010 had P4/P5/P6 = 139/140/141. P5 is the transition
  checkpoint before shaft parallel (P6), so the lead arm (mirror of P3) is used.
- **P9 could match a later pass.** With the longer window, "lead arm nearest
  horizontal anywhere after impact" could pick the arm passing parallel again as
  the club is lowered. HackMotion defines P9 as the opposite of P3, so it is the
  first post-impact parallel.

`workflow.py` now uses YOLO26n body pose with the 17+2 lifter, matching the joint workflow.
Its default `--lifter` is `17+2` (also the default for the standalone
`golf_pose_3d` and `standard_frames` modules). The club-only ViTPose still
predicts five landmarks; heel/toe_down/toe_up and their confidences are dropped
before lifting (`dropped_2d_landmarks`), for the reasons in the 17+2 section
below. `--lifter 17+5` keeps all five; the old "club ViTPose requires 17+5" error
was removed.

Validation (77 tests pass, including new window/P5/P9/P10 cases):

| Run | Before | After |
| --- | --- | --- |
| 005, golfpose_workflow (saved 2D replayed through the new phases, 17+2 lifting and extraction) | window 84-285; P1-P10 `84,133,141,177,188,214,226,248,253,254` | window 84-412 (`raised_hands_lowered`); `84,133,141,177,195,214,226,248,250,346` |
| 010, golfpose_workflow (replay, face-on) | window 60-215; `60,72,103,139,140,141,167,175,180,197` | window 60-251 (clip end, hands still raised); `60,72,103,139,153,165,167,175,178,219` |
| w07, `workflow.py` fresh YOLO26n + club ViTPose + 17+2 | not run | window 9-52; `9,15,20,27,29,32,34,35,36,52`; 19-joint finite export |

`workflow.py references` with 17+2 produced 10/10 references from 30 finite
camera poses (8 club overlays). `workflow.py score` evaluated 140/140 parameters
(92.0/100, a similarity score). Outputs: `runs/workflows/videos/w07_17plus2_20261104/`,
`runs/workflows/references_17plus2_20261104/`, `runs/workflows/scores/w07_17plus2_20261104/`.
Visual checks: 005 P10 shows the held finish facing the target, and 010 P5 shows the lead arm parallel
in the early downswing. Remaining limit: in 005 the club is blurred after impact
(hosel confidence about 0.2 for frames 236-246), so P8 (248) lands next to P9 (250).
Replays reuse saved 2D inference; the published `runs/golfpose_workflow/videos/005/`
was produced with the old rules. Re-run it to refresh:
`.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/005.mp4`.

## Joint ViTPose workflow lifts 17+2 (2026-11-04)

`python -m golfpose_workflow` still runs one 2D model,
`golfpose_golfer_ViTPose_huge.pth`, but now lifts with MixSTE2 17+2
(`golfpose_17+2_33.6_30.5_59.5.bin`). The 17 body joints plus `shaft` and
`hosel` are kept; `heel`, `toe_down` and `toe_up` and their confidences are
dropped before lifting and listed as `dropped_2d_landmarks` in `<stem>_3d.json`.
Saved video/reference arrays are 19-joint. `golf_pose_3d` and `standard_frames`
now accept `--pose-backend golfpose-golfer-vitpose-huge --lifter 17+2`; 17+5
remains available there.

Evidence on `010.mp4`: the three head points had the lowest median confidence
(0.58-0.63) and a 95th-percentile distance of about 250 px from the hosel, for a
driver head about 25 px wide; the golfer crop maps about 13 px to one heatmap cell.
Phases, club refinement and scoring use only shaft and hosel. GolfPose's
two-class YOLOX club box + club-only ViTPose was tested on ten frames and found
the club in 3/10 frames (scores 0.01-0.08, one wrong object); its golfer-box
fallback collapsed shaft/hosel. Comparison: `.tmp/club_compare_010/`.

Validation: 74 tests pass. The full `010.mp4` run
(`runs/golfpose_workflow/videos/010_17plus2_20261104/`) kept identical shaft/hosel
2D coordinates and exported P1-P10 at
`60, 72, 103, 139, 140, 141, 167, 175, 180, 197` with complete club evidence in all
ten. P4-P6 still fall on consecutive top frames; that comes from the P5/P6 phase
rules, not from the dropped points (fixed by the P-position update above). 17+2 references
(`runs/golfpose_workflow/references/17plus2_20261104/`) produced 10/10 references
and club overlays; scoring evaluated 140/140 parameters (89.2/100, a similarity
score, not an accuracy measurement).

## Explicit ROI initialization and root diagnostics (2026-11-04)

`pro_refs/010.mp4` contains 252 decoded frames at 592x1280. The one-class YOLOX
detector returned no box reaching 0.25 on any frame. Previously, video inference
skipped ViTPose for every frame and eventually raised the misleading message
`No reliable GolfPose root found`.

The joint video route now accepts `--golfer-box X1 Y1 X2 Y2` in original-frame
pixels. It initializes the pose ROI explicitly; YOLOX still runs on every frame
and can take over when detection becomes available. Manual ROI usage is logged
as `manual-golfer-roi`, without fabricating a detector score. Shape, finite
coordinates, positive size and decoded-frame bounds are validated. There is no
video full-frame pose fallback or relaxation of the root threshold.

```powershell
.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/010.mp4 --golfer-box 140 220 500 900 --view face_on
```

Failure messages now distinguish missing ROI/zero pose inference from an actual
low-confidence or out-of-frame root. The failed run saves
`pose_inference_diagnostics.json` and `.npz` with decoded/inferred frame counts,
raw coordinates/validity/confidence, maximum root confidence and per-frame crops.
Frame extraction and scoring continue using their existing rules.

Validation: 72 tests pass. On actual frames 0, 126 and 200, the supplied ROI
produced observed root joints with confidence about 0.918, 0.967 and 0.911.
Sample diagnostics: `runs/diagnose_010_20261104/manual_roi_samples.json`.

The full CLI run completed all 252 frames with observed root joints on every
frame (confidence range about 0.854-0.971). All crops were explicitly logged as
`manual-golfer-roi`, with no fabricated YOLOX detections. Automatic extraction
produced ten finite 22-joint poses at indices
`60, 72, 103, 139, 140, 141, 167, 175, 180, 196` and published the completed video
run under `runs/golfpose_workflow/videos/010_roi_20261104/`.
Validation report: `runs/diagnose_010_20261104/validation.json`. This verifies
execution and data contracts, not ground-truth joint or phase accuracy.

## YOLOX detector routing revision (2026-11-03)

| Workflow | Detector checkpoint | Pose input |
| --- | --- | --- |
| `workflow.py` (YOLO26n) | `golfpose_checkpoints/golfpose_detector_2cls_yolox_s.pth` | Person crop to YOLO26n body pose; associated club crop to club-only ViTPose |
| `python -m golfpose_workflow` | `golfpose_checkpoints/golfpose_detector_1cls_yolox_s.pth` | Golfer crop to the single native 22-joint golfer ViTPose model |

Saved checkpoint metadata confirms classes `person, club` for two-class YOLOX
and `golfer` for one-class YOLOX. Both load strictly with their matching classifier
head sizes. The native adapter follows the saved BGR, 640x640 resize/top-left
padding, class-score times objectness, and 0.65 class-wise NMS configuration.
The implementation was checked against the primary MMDetection
[YOLOX backbone](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/csp_darknet.py),
[neck](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/yolox_pafpn.py),
and [head](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/yolox_head.py).

The YOLO26n workflow defaults to `--club-crop-backend official-yolox-2cls` for
references and videos, including standalone modules. A single two-class detector
pass supplies the person and club candidates. The selected person score must be
>= 0.25; a 10% expanded person crop is passed to YOLO26n and coordinates are
translated back to the original frame. If no accepted person box exists, YOLO26n
runs its own full-frame body detection and the fallback is logged. Club candidates
are associated with the golfer/hands to avoid unrelated banners. If none remains,
club-only ViTPose refines from the golfer box. Default club segmentation is replaced
by the two-class detector; explicit legacy crop options remain available.
YOLO's original COCO confidence values now accompany the restored coordinates
for the existing confidence-based body reference ranking.

The joint ViTPose workflow uses the one-class detector at score >= 0.25. It uses
original-image decoding of the detected golfer ROI, with the existing 1.25 UDP
expansion. On video gaps, the last detected ROI is retained and logged as
`tracked-detector-roi`; no initial ROI means unobserved joints. There is no video
full-frame fallback. For still references only, a detector miss can use the full
panel with `reference-panel-fallback` recorded. Raw pose confidence/validity is
preserved; tracked detector boxes are distinguished from detector observations.

Detection is not uniformly reliable on these inputs: at 0.25, the one-class model
found 1/30 reference panels and 7/53 `w07` frames; the two-class person output found
1/30 panels and 3/53 frames. These are box availability counts, not accuracy scores.
The fallbacks above are necessary on this sample. Details are saved in
`runs/validation_detector_routing_20261103/detector_audit.json`.

Reference JSON saves the selected crop details. Video JSON saves `detector_crops`
with zero-based frame indices, selected boxes/scores and fallback sources.
MixSTE2 17+5, whole-view selection, overlays, phase/extraction conditions and
14-criterion scoring retain their current implementations. Fresh inference is
required to apply detector changes; cached numerical runs retain their old metadata.

Validation: all 67 tests pass. Both pipelines produced finite P1 references and
processed all 53 frames of `w07.mp4`, saving detector/crop diagnostics. YOLO26n
automatically exported ten finite 22-joint poses. The cropped joint ViTPose run
found address/top/impact at 9/27/34, but P5 failed the unchanged shaft vertical
component threshold: its minimum eligible value was about 0.357, above 0.35.
P6 depends on P5, so that run did not publish a completed ten-frame bundle.
Its 53 pose crops were 7 detections plus 46 retained detector ROIs, with no
full-frame video pose inference. This limitation is preserved for review rather
than filling the missing phases with arbitrary frames. Published latest pointers
were verified unchanged. Report:
`runs/validation_detector_routing_20261103/pipeline_validation.json`.

## Separate joint GolfPose workflow (2026-11-03)

A second launcher, code folder and output collection are available independently
of the YOLO26n workflow. See [golfpose_workflow/README.md](golfpose_workflow/README.md).

```powershell
.\.venv\Scripts\python.exe -m golfpose_workflow --reference
.\.venv\Scripts\python.exe -m golfpose_workflow --video pro_refs/w07.mp4
.\.venv\Scripts\python.exe -m golfpose_workflow --score
```

Each reference/video command infers fresh native 22-point 2D coordinates using
only `golfpose_golfer_ViTPose_huge.pth` for body and club pose, then lifts with
MixSTE2 17+5. Results and latest pointers live under `runs/golfpose_workflow/`.
The same reference selection, club overlay, phase/extraction conditions, ten-frame
validation and 14-criterion scoring implementation are shared with the original
workflow. Automatic video extraction exports ten source frames, ten 3D images,
matching confidence/coordinate arrays and original frame indices. Reviewed indices
remain available via `--positions-json`; insufficient automatic results do not
publish a completed video pointer. Scoring reads this workflow's own latest runs.

`workflow.py` still defaults to YOLO26n body inference. Its output collection and
latest pointers stay separate. The new launcher has no cached-inference import mode.

Validation before the detector revision: 60 tests passed; fresh inference produced all ten references and
automatically exported ten frames from the full 53-frame `w07.mp4` video. All
selected 22-joint 3D poses were finite. Scoring evaluated all 140 parameters
with the shared rules. Original latest pointers were verified unchanged.
Installed the missing Ultralytics CLIP tokenizer for existing optional ball timing,
pinned its source commit in `requirements.txt`, and verified `pip check` is clean.
Execution details: `runs/golfpose_workflow/validation.json`.

## Current default: YOLO26n body pose (2026-11-03)

Reference generation and video inference now default to `--pose-backend yolo26n-pose`.
The joint `golfpose_golfer_ViTPose_huge.pth` checkpoint is no longer loaded by default.
The YOLO26n implementation was checked against repository HEAD
`40f50ebeef4e29139275db968c250938204e238c` at
https://github.com/StevenZhou795/golf-posture.

- Body: `yolo26n-pose.pt`, 17 COCO joints mapped to the 17 GolfPose body channels;
  root/spine/thorax/neck/head use the existing COCO conversion rules.
- Club: `golfpose_checkpoints/golfpose_club_ViTPose_huge.pth`, five club landmarks.
  Two-class YOLOX supplies person/club boxes in references and videos; club-only
  ViTPose supplies the five landmark coordinates. Golfer-box refinement is the
  fallback when no associated club detection is available.
- Lifting: MixSTE2 `17+5`. Whole-view body selection, club donor selection,
  shaft-only fallback, previews, phase recognition, refinement, and exports remain.

The public launcher and both standalone inference modules use YOLO26n body pose
and club-only ViTPose by default. Alternative pose backends remain explicit options.
Fresh inference is required: `--from-run` and reference `--rebuild` reuse saved poses.
Existing completed results and latest pointers are not regenerated by this change.

Validation before the detector revision: all 56 tests passed. A real P1 sheet produced finite 3D poses for all
three views and a finite selected reference. A ten-frame sampled video completed
inference, lifting, refinement, rendering, and export with finite `(10, 22, 3)`
selected poses. It used explicit test indices, so it does not validate automatic
phase accuracy. Results: `runs/validation_yolo_body_20261103/validation.json`.

```powershell
.\.venv\Scripts\python.exe workflow.py references
.\.venv\Scripts\python.exe workflow.py video pro_refs/w07.mp4
```

The joint-model sections below describe the earlier experiment and apply only when
`--pose-backend golfpose-golfer-vitpose-huge` is explicitly requested.

## Historical joint golfer-with-club inference (2026-11-03)

The earlier joint-backend experiment loaded one 2D pose checkpoint:
`golfpose_checkpoints/golfpose_golfer_ViTPose_huge.pth`. Its saved configuration
and strict parameter load confirm **22 native GolfPose joints**: 17 body joints
followed by `shaft`, `hosel`, `heel`, `toe_down`, and `toe_up`.

```powershell
.\.venv\Scripts\python.exe workflow.py references
.\.venv\Scripts\python.exe workflow.py video pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4
```

That experimental backend is `golfpose-golfer-vitpose-huge`. It runs one pose model on
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
joints. Reference alignment, body/club-overlay rendering, video
phase selection, club refinement, ten-frame export, and scoring retain their
existing algorithms and file contracts.

References additionally save `<view>_2d_confidence` (22 values) in `joints.npz`;
video arrays save `keypoints_2d_confidence` (frames x 22). The pose/club backend
metadata refer to the same loaded model instance. `--lifter 17+5` was required
at the time (17+2 is accepted since 2026-11-04);
legacy `--club-pose-backend` and `--club-crop-backend` flags are ignored while
the joint backend is selected. Inputs must contain one golfer; a small golfer
in a wide frame may lose detail at the fixed network input resolution.

To retain the earlier two-pose-model route, explicitly select
`--pose-backend vitpose-huge --club-pose-backend vitpose-huge`; references use
`--club-crop-backend golfer-box`, while videos can use `segmentation` or
`official-yolox-2cls` crops. DEKR remains removed. `--from-run` reuses old model
results, so omit it when generating outputs with the new checkpoint. Existing
completed runs and latest pointers are not changed by the implementation update.

Validation on 2026-11-03: the checkpoint loaded strictly with 22 output
channels and declared the `golfswing_golfer` dataset. All 41 tests passed,
including native channel/flip order, out-of-frame validity, shared reference
and video routing, and GolfPose wrist indexing. A real P1 sheet produced finite
3D results for all three views and a fused reference. A ten-frame video test
completed inference, MixSTE2 lifting, club refinement, rendering, and extraction
with finite `identified_frames.npz:keypoints_3d` of shape `(10, 22, 3)`.
This video test used sampled frames and explicit test indices; it does not
measure automatic phase-selection accuracy. Artifacts and model metadata are
saved in `runs/validation_golfer_vitpose_20261103/validation.json`.

## Python environment and dependencies

Setup log (2026-11-02): the system Python 3.12 environment reported
`torch==2.6.0+cpu` but failed to import it with WinError 126 for
`torch/lib/caffe2_nvrtc.dll`. Its package directory contained leftover CUDA
libraries. A separate project `.venv` is used to avoid that mixed installation.
The system environment is unchanged.

Reproduce the setup from `J:\Golf_Posture` in PowerShell:

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

`requirements.txt` pins the workflow dependencies, including NumPy, SciPy,
OpenCV, Matplotlib, Ultralytics, timm, einops, and local-attention. PyTorch and
torchvision use CPU wheels here; the current ViTPose/MixSTE inference code runs
on CPU. MMEngine, MMPose, and MMDetection are not required by the local
inference implementations. MediaPipe is an optional video wrist fallback and
is not required for reference generation.

Installed and verified versions:

| Package | Version |
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

Validation on 2026-11-02: `pip check` reported no broken requirements; PyTorch
import and tensor-to-NumPy conversion passed; all 35 unit tests passed. The real
`golfpose_club_ViTPose_huge.pth` loaded with five output channels, and the
MixSTE2 17+5 checkpoint loaded with 22 joints. These checks validate package
compatibility and checkpoint loading; reference images were not regenerated.

Use the project interpreter explicitly to avoid the broken system installation:

```powershell
New-Item -ItemType Directory -Force .tmp\ultralytics\Ultralytics, .tmp\matplotlib | Out-Null
$env:YOLO_CONFIG_DIR = "$PWD\.tmp\ultralytics"
$env:MPLCONFIGDIR = "$PWD\.tmp\matplotlib"
.\.venv\Scripts\python.exe workflow.py references
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Run each stage with one command from `J:\Golf_Posture`:

```powershell
python workflow.py references
python workflow.py video pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4
python workflow.py score
```

Scoring uses the latest successfully completed reference and video workflows.
Each command accepts `--output-dir`. Run `python workflow.py <stage> --help`
for options. `python -m golf_workflows` is an alternative launcher.

## 推理思路与模型调用说明

项目采用“先识别二维关键点，再提升为三维姿态，最后按姿态指标评分”的流程。
Body inference uses YOLO26n-pose, club landmarks use club-only ViTPose, and MixSTE2 performs 3D conversion.
The current `workflow.py` defaults are `yolo26n-pose`, club-only ViTPose, and `17+5`.
视频开启球杆 refinement，CPU 线程数为 8；底层模块的独立命令可能有不同默认值。

### 默认调用哪些模型

| 模型 | 权重文件 | 输入与职责 | 调用阶段 |
| --- | --- | --- | --- |
| YOLO26n-pose | `yolo26n-pose.pt` | 17 COCO body joints mapped to GolfPose body channels | references, video |
| GolfPose club ViTPose-Huge | `golfpose_checkpoints/golfpose_club_ViTPose_huge.pth` | Five club landmarks | references, video |
| GolfPose MixSTE2 17+5 | `golfpose_checkpoints/golfpose_17+5_39.2_32.3_62.8.bin` | 将二维人体与球杆关键点序列提升为三维坐标 | references、video |
| YOLOE 分割模型，可选 | `yoloe-26n-seg.pt` | 在文件存在时以 `golf ball` 类别检测球，辅助修正击球时刻 | 仅 video |

`--pose-backend yolo26n-pose` uses YOLO body keypoints; club-only ViTPose still supplies the five club landmarks.
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

Video frames use YOLO26n body joints and club-only ViTPose landmarks. Default refinement runs before phase-window selection.
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

## 1. Build ten references

`references` reads the ten JPG sheets in `pro_refs/Standard_10_frames`, crops
face-on/down-the-line/rear views, runs body and club inference, lifts the views
with MixSTE and aligns the poses in face-on space. The higher-confidence
face-on or down-the-line body supplies the entire numerical reference.
The rear view is retained for inspection.

Both `python workflow.py references` and `python -m golf_workflows.standard_frames`
default to `yolo26n-pose` for the body and club-only ViTPose for the club, with MixSTE2
17+5. The mapped 17 body channels and five club channels feed lifting and whole-pose selection.
To select the optional joint-model experiment explicitly:

```powershell
python workflow.py references --pose-backend golfpose-golfer-vitpose-huge --lifter 17+5
```

Output: `runs/workflows/references/`, with `workflow.json` and ten position
folders. Each contains `reference_3d.jpg`, `joints.npz` (numerical coordinates),
`joints.json` and camera images/poses. Fresh inference also writes `run_summary.json`.
Completion requires one finite reference and readable image per P1-P10 position.

Each reference folder also includes PNG contact sheets in numeric P1-P10 order:
`reference_3d_preview.png` (ten bodies),
`reference_3d_club_overlay_preview.png` (ten club overlays), and
`reference_all_preview.png` (both sets together). Overlays show donor-view and
inferred-point labels; missing overlays use an explicit placeholder. These
previews are generated from saved JPGs after fresh inference, cached reference
import, and saved-pose rebuilds; no model inference is needed to refresh them.

Body-reference selection rule (2026-11-03): compare only `face_on` and
`down_the_line` views that have a finite 3D pose. Score each using the **mean
confidence of the first 17 body joints**. For the native 22-joint checkpoint,
invalid/missing body joints contribute zero, and finite confidence is clipped
to [0, 1]. Club confidence does not affect body selection. For legacy COCO
backends, use mean COCO confidence multiplied by the mapped body's valid-joint
fraction. A tie prefers `face_on`. A known score takes priority over a missing
score; if all scores are absent in old saved runs, prefer `face_on`.

Copy the selected view's entire normalized, aligned pose into
`joints.npz:reference_3d`. Do not replace x/y/z components or individual joints
from another camera. `rear` remains available for inspection, never selection.
`side_axis` is now null, and `reference_kept` contains only the selected view.
`joints.json:reference_body_selection` records the selected view, both scores
and the rule. In the examined P9 run, face-on scores 0.9250 versus 0.7799 for
down-the-line, so its complete skeleton is retained. The club-overlay donor
is selected independently by the following rule; it cannot change the body
or the numerical scoring reference.

Fresh reference generation and `standard_frames --rebuild` both apply this
rule. Rebuild uses saved 2D confidence/validity and 3D poses, regenerating the
body image and overlay together without neural-network inference. Existing
completed folders are not rewritten automatically; `--from-run` alone copies
stored results and does not apply this revision.

Validation: all 49 tests passed. A rebuild of a copy of the saved P9 folder
selected `face_on`; its numerical reference exactly matches the complete
aligned face-on pose, and its overlay retains the same body. Original per-view
3D poses and the selected production run were preserved. The rebuilt example
is in `runs/validation_body_selection_20261103/`.

Club overlay selection rule (2026-11-03): consider **only `face_on` and
`down_the_line`**. `rear` is never an overlay donor. There are no fixed
per-position view overrides. A candidate must have all five club landmarks
valid, including the head markers, finite 2D/aligned 3D coordinates and
confidence, and at least 12 pixels between shaft and hosel. These complete
five-point candidates always take priority over incomplete candidates.

Among eligible candidates choose the higher score:
`min(confidence) + 0.25 * mean(clip(confidence, 0, 1)) - fallback_penalty`,
where the penalty is 0.05 when a crop fallback was used, otherwise zero.
Confidence here is the ordered five-point club confidence. An exact tie
prefers `face_on`.

**Missing-joint fallback:** only when neither eligible view has all five
reliable 2D club landmarks, a view may qualify using **shaft alone**. The shaft
must have a valid, finite 2D observation and confidence at least 0.10. The other
four landmarks may all be absent, invalid, or out of frame. Rank these partial
candidates by `shaft_confidence - fallback_penalty`, with face-on winning an
exact tie. The 12-pixel shaft/hosel length requirement does not apply when using
this fallback. Finite aligned MixSTE geometry for all five club joints is
still required so an overlay can be rendered. If a complete view exists, this
fallback is not used. If no complete or confident-shaft candidate qualifies,
no overlay is produced.

For a fallback overlay, copy the same selected view's five aligned MixSTE
coordinates together; missing endpoints/head markers are **inferred geometry**,
not recovered 2D observations. Keep original 2D validity flags unchanged.
`reference_club_overlay` records `shaft_only_fallback`,
`observed_club_landmarks`, and `inferred_club_landmarks`; unavailable observed
shaft length/direction is null. The image title identifies inferred points.
This applies to fresh inference and saved-pose rebuilds. The output remains
`reference_3d_club_overlay.jpg` / `reference_3d_club_overlay` in the NPZ file.

Validation: all 52 tests passed, including a shaft-only observation with the
other four 2D joints missing. Rebuilt copies of P8 and P10 both produced overlays:
P8 selected face-on (inferred `toe_down`), and P10 selected down-the-line
(inferred `heel`, `toe_down`, `toe_up`). Examples are saved under
`runs/validation_shaft_fallback_20261103/`; production outputs were not overwritten.

P10 overlay refresh (2026-11-03): the subsequently selected
`runs/workflows/references/` folder still contained a pre-fallback P10 result.
Its overlay was refreshed from saved aligned coordinates, selecting
`down_the_line` with shaft confidence 0.9449 versus 0.6468 for face-on. All five
club coordinates were copied together from that donor; missing head markers
remain labeled inferred. The numerical body reference was verified unchanged,
and previous P10 files were backed up under
`runs/workflow_snapshots/p10_overlay_before_refresh_*`.

Copy the selected view's **entire aligned MixSTE club geometry as one unit**,
preserving channel order: `shaft`, `hosel`, `heel`, `toe_down`, `toe_up`.
Do not mix endpoints or head markers from different cameras. The workflow
does not attach the shaft to the wrists or force a shaft/head size.
This selection is applied during reference generation and recorded in
`joints.json:reference_club_overlay`, including eligible views, chosen view,
score and confidence. It may write `reference_3d_club_overlay.jpg` and
`joints.npz:reference_3d_club_overlay` by copying that view's already aligned
MixSTE club coordinates. This is a labeled visualization overlay only.
`joints.npz:reference_3d` is unchanged and remains the sole scoring input;
`joints.json:reference_club_overlay` records the source view and confidence.
The base `reference_3d.jpg` is body-only so imputed or out-of-frame club points
cannot be mistaken for observations.

To reproduce the paper's separate-club route, select the restored two-class
GolfPose YOLOX-s detector as the club crop source. Class `1` supplies the
per-frame club box, and the existing club-only ViTPose-H model supplies the
five landmarks. This is per-frame detection and pose estimation, not a learned
temporal tracker:

```powershell
python workflow.py references --pose-backend vitpose-huge --club-crop-backend official-yolox-2cls
python workflow.py video <video.mp4> --pose-backend vitpose-huge --club-crop-backend official-yolox-2cls
```

The explicit `segmentation` crop option remains available. Two-class YOLOX
supplies default video club crops. Reference crops also default to
`official-yolox-2cls`. The joint backend uses one-class YOLOX golfer crops with
the tracked-ROI and reference-panel fallback policies described above.

Reuse completed references without new inference:

```powershell
python workflow.py references --from-run runs/standard_frames_vitpose_huge_17plus5_20261027
```

## 2. Identify ten video frames

### Video/reference inference parity audit (2026-11-03)

Historical joint-backend validation: reference panels and video frames shared `GolferViTPose.predict_frame`: one
`golfpose_golfer_ViTPose_huge.pth`, native 22-channel joint order and flip map,
UDP/DARK decoding, confidence >= 0.10, in-frame validity, and normalization from
reliable body-joint bounds. Video uses native wrist joints 13/16, not COCO
indices. This earlier audit used full single-golfer images with 1.25 expansion.
The detector revision above now supplies one-class YOLOX crops in both paths.

Latest missing-club rules now apply per video frame:

- `complete`: all club channels are observed and shaft/hosel separation is at
  least 12 pixels.
- `shaft_fallback`: the club is incomplete but shaft is observed with score
  >= 0.10. Show that frame's MixSTE club geometry as a unit and label missing
  endpoints/head as inferred. Never change the raw 2D validity flags.
- `none`: no qualifying club evidence; hide the 3D club for that frame.

The full video arrays and ten-frame export retain `keypoints_2d_confidence`
(frames x 22), `keypoints_2d_valid`, five club confidences and `club_evidence`.
Each selected frame's manifest records its evidence category and the shared
pose-backend metadata. Older cached runs without an evidence array derive it
from saved coordinates/validity/confidence during export. Source frame indices
and numerical 3D poses remain unchanged by export. Inferred 3D club geometry
is visualization/model output, not a newly detected 2D endpoint.

For recognition, use a single observed motion landmark consistently across the
clip: hosel when it has enough valid samples, otherwise shaft when available,
otherwise wrist-based evidence. `phase_detection.club_track_source` records
the choice. The existing body-relative wrist rise/return, finish-hold checks,
bounded ball correction, and ordered P1-P10 selection remain in effect.
Shaft-only frames do not become valid endpoint pairs for geometric fitting or
club-based impact correction. A usable shaft does not by itself prove that all
ten swing phases can be identified.

Reference face-on/down-the-line competition does not apply to a single-camera
video: do not replace a temporal event frame with an unrelated higher-confidence
frame. Video keeps each frame's complete body prediction through lifting.
Club trails remain opt-in through `--track-club-path`; displaying a shaft
fallback does not enable a trajectory automatically.

Validation: all 56 tests pass. Replaying the current `w07` run's saved 22-joint
2D observations through automatic recognition, MixSTE2, refinement, rendering
and ten-frame export produced indices `[7, 15, 18, 27, 32, 33, 34, 35, 37, 39]`,
matching its previous selection. Exported XYZ shape is `(10, 22, 3)` and full
confidence shape is `(10, 22)`. This replay reused saved 2D inference; no latest
video/reference pointer was changed. Artifacts are in
`runs/validation_video_reference_parity_20261103/`.

`video` runs YOLO26n body inference plus club-only ViTPose, phase identification,
MixSTE2 17+5 lifting, and P1-P10 selection. Both the launcher and standalone
module default to `yolo26n-pose`. Existing club refinement remains enabled by
default in the launcher.
Refinement uses angular gap filling, blended windows and a constrained shaft
segment. Its length is inferred unless `--club-marker-length-m` supplies a
measured marker distance. See [the improvement log](SHAFT_TRACKING_IMPROVEMENT_LOG.md).

Output: `runs/workflows/videos/<video-stem>/`:

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

Automatic phase detection remains heuristic. A run cannot publish as complete
unless it supplies ten distinct, increasing indices with finite 3D poses.
With refinement enabled, timing combines refined club geometry with wrist
rise/return in body-relative coordinates. A finish requires sustained low wrist
speed with raised hands; without that evidence the interval extends to clip end.
Ball correction is bounded near the wrist return and runs before lifting.
`phase_detection` in the run JSON records the evidence and any fallback.
Reviewed indices can be supplied when needed:

```powershell
python workflow.py video pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4 --positions-json runs/golf_pose_3d_vitpose_huge_17plus5/b85ed0b4db566e63bce190824c2f2196/selected_frames.json
```

Use `--no-club-refinement` to disable club refinement. `--pose-backend yolo26n-pose --club-pose-backend segmentation --club-crop-backend segmentation --lifter 17+2` restores the earlier
two-proxy club input. A measured marker length requires refinement.

Export ten poses from a completed run without running models again:

```powershell
python workflow.py video --from-run runs/golf_pose_3d_club_refined_20261027/b85ed0b4db566e63bce190824c2f2196
```

Cached import preserves saved numerical results and model choices. It does not
apply new refinement or selection overrides. It exports the ten-pose bundle
and 3D images from the saved coordinates, preserving the original run.

## 3. Score the selected poses

```powershell
python workflow.py score
# Or choose both inputs explicitly:
python workflow.py score runs/workflows/videos/b85ed0b4db566e63bce190824c2f2196 --references runs/workflows/references
```

The scorer reads `identified_frames.npz/json`, matches P1-P10 to the reference
`joints.npz:reference_3d` coordinates and keeps the original video frame indices.
JPGs visualize these poses; their pixels are not numerical scoring inputs.
The scorer does not identify frames again.

The existing 14 posture criteria, normalization, circular angle differences
and equal weighting are retained. See [SCORING_METHODOLOGY.md](SCORING_METHODOLOGY.md).
The three extra 17+5 head points do not add scoring criteria.

Output: `runs/workflows/scores/<video-run>/`, containing `score.json`, provenance
in `workflow.json`, ten user pose images and ten reference comparison images.
The score reports position/parameter coverage. It does not establish calibrated
physical 3D accuracy or coaching validity.

## Organization and preservation

`workflow.py` is the public launcher. Eight shared modules moved into
`golf_workflows/`: `standard_frames.py`, `golf_pose_3d.py`,
`compare_standard_frames.py`, `score_swing.py`, `trace_swing_plane.py`,
`swing_phases.py`, `vitpose_tracking.py` and `club_refinement.py`.
`cli.py` connects the three stages and validates their saved data contracts.
Detector helpers, training scripts, models, datasets and previous runs stay in place.

Low-level commands remain available as modules, for example
`python -m golf_workflows.golf_pose_3d --help`. Historical code snapshots remain
unchanged; their old imports may need adjustment if executed directly.

A nonempty explicit output folder is rejected. Default outputs receive a
timestamp suffix on reruns. Latest-run pointers under `runs/workflows/latest_*.json`
update only after successful completion. Failed runs retain diagnostics but
cannot replace a completed active result.

## Reorganization validation (2026-10-27)

All three public commands were exercised with the completed ViTPose 17+5
references and refined video via `--from-run`, followed by `score`. This validates
staging, export and scoring without repeating expensive ViTPose inference.
Ten reference and video poses were exported, with all 140 comparisons available.
The result was **93.8/100**. A fresh reference build was also completed using
the real models: all 30 camera poses and ten references succeeded, under
`runs/workflows/references_fresh_validation_20261027`. The older 92.9 result
used different inputs and is preserved.

Tests cover the existing lifting/refinement behavior, the ten-pose contract,
original source indices, explicit reference/output routing, identical-pose
scoring, incomplete references, inconsistent manifests and output preservation.
All 33 tests passed with `python -m unittest discover -s tests -v`.
