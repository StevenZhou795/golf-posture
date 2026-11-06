# Shaft tracking improvement log

Date: 2026-10-27

## Objective and baseline

Improve shaft direction and length consistency while preserving the existing models,
raw observations, validity flags and previous outputs. This experiment uses
`pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4` and the reviewed P1–P10 frame indices.

Baseline run:
`runs/golf_pose_3d_vitpose_huge_17plus5/b85ed0b4db566e63bce190824c2f2196`.

Initial audit artifacts are in `runs/shaft_tracking_audit_20261027`.
The video contains 364 frames; 281 frames (60–340 inclusive) have lifted poses.
Both club endpoints were detected in 170 of these frames. At least one was
missing in 111 frames; the longest consecutive gap was 25 frames.

## Findings

1. The custom club segmentation model supplies PCA contour endpoints. The end
   nearest the wrists is mapped to GolfPose `shaft`; the other is mapped to
   `hosel`. These proxies do not establish the original training-marker locations.
2. The tracker already blends consecutive endpoint coordinates. The workflow
   then linearly fills missing coordinates and applies a five-frame moving average
   at 30 fps. Interpolating/averaging endpoints independently can shorten a
   rotating shaft's projected vector.
3. MixSTE2 predicts XYZ coordinates for every joint. Spatial attention links
   golfer and club points; temporal attention links each joint across 243 frames.
   There is no explicit rigid club, fixed segment length or calibrated reprojection
   constraint. The published losses concern joint position, velocity and temporal
   consistency: [GolfPose paper](https://minghanlee.github.io/papers/ICPR_2024_GolfPose.pdf).
4. The 17+5 trial still has only two independently observed club points. `heel`,
   `toe_down` and `toe_up` are unmeasured and filled at hosel. Their validity flags
   remain false. This experiment does not change that fact.
5. The old window assembly selects the first available prediction in overlaps,
   producing a hard context handoff. This is secondary to missing observations
   in this particular video, but is straightforward to improve.
6. The virtual body-normalized canvas is not a calibrated camera. Model metres
   and nominal equipment length cannot be equated without validation. The
   constrained segment is shaft-marker to hosel-marker, not full club length.

## Implemented changes

### Angular gap filling

`club_refinement.prepare_club_inputs` uses jointly observed endpoint pairs to
derive center, projected length and unwrapped image angle. It interpolates these
quantities separately through missing intervals and reconstructs the endpoints.
Detected pairs are preserved, rather than averaged with neighbouring frames.
When just one endpoint is observed, that endpoint is preserved and the missing
one is reconstructed from the interpolated vector. Raw arrays and validity flags
are never changed. Body smoothing is unchanged.

This prevents chord shortening during a rotating gap, but cannot recover
unobserved fast turns, full rotations, occlusion or endpoint identity errors.
Projected length remains variable to allow foreshortening. The existing detector
and its endpoint blending remain upstream; replacing these with supervised
shaft/hosel landmarks is still future work.

### Overlapping MixSTE window blending

With refinement enabled, windows use a 121-frame stride and an end-aligned final
window. Predictions are averaged with triangular weights favouring window
centres. The old first-window ownership remains available with refinement off.
Short clips retain last-frame padding to 243 frames.

### Constrained 3D segment fit

`club_refinement.refine_club_3d` fits unit direction vectors `u[t]` and constructs
`hosel[t] = shaft[t] + L * u[t]`. The positive distance `L` is either explicitly
supplied with `--club-marker-length-m`, or inferred as the median MixSTE distance
over reliable jointly detected endpoint frames. The latter is a model prior,
not a physical measurement.

A positive weak-perspective projection scale `s[t]` is fitted to centered body XY
coordinates and normalized 2D body coordinates. Scale is bounded to 0.5–2 times
its positive sequence median. The normalized target club vector is
`v[t] = (hosel_2d[t] - shaft_2d[t]) / (s[t] * L)`.

The robust least-squares residual combines:

- `sqrt(w[t]) * (u[t].xy - v[t])` for observed endpoint pairs;
- `0.3 * (u[t] - MixSTE_direction[t])` as a depth/direction prior;
- `0.4 * second_difference(u[t])` for temporal regularity.

`w[t]` is joint endpoint validity times detector confidence. Missing pairs exert
zero image-data force. The solver uses `soft_l1`, scale 0.1 and at most 100
function evaluations. These weights are initial engineering choices, not tuned
against ground-truth golf motion. Fewer than three reliable pairs or failed
convergence skips the constraint and records the reason.

The shaft anchor and all body joints remain unchanged by this fitting stage.
Additional predicted head markers translate with the hosel, preserving their
relative offsets; they do not become measured landmarks. This is a rigid shaft
segment approximation and does not model shaft flex.

### Integration and preservation

`golf_pose_3d.py` adds `--club-refinement` and optional
`--club-marker-length-m`. Without an explicit output directory, refinement adds
`_club_refined` to the output directory name. The option is experimental and
off by default. Existing model defaults and checkpoint files are retained.

NPZ output now also retains `keypoints_3d_before_club_refinement` and
`club_confidence`. JSON records the refinement method, inferred/measured length,
optimizer result and projection caveat. The pre-refinement array has the new
inputs and window blending; it is not a copy of the previous run's predictions.

`replay_club_refinement.py` (now in `archive/scripts/`) reuses cached 2D detections for a controlled test,
repeats MixSTE inference with the new preparation and blending, fits the club,
and exports a full video and P1–P10 stills to a separate empty folder.

## Trial execution and results

Executed:

```powershell
python replay_club_refinement.py runs/golf_pose_3d_vitpose_huge_17plus5/b85ed0b4db566e63bce190824c2f2196 --output-dir runs/golf_pose_3d_club_refined_20261027/b85ed0b4db566e63bce190824c2f2196
```

The cached input lacks original detector confidence values. This replay uses
binary valid/missing weights and explicitly records that limitation. Fresh video
runs use the detector confidence from each accepted club sample. The replay
reuses detections and reviewed indices; it does not run ViTPose again.

| Metric | Previous run | New inputs + blended windows | Final constrained fit |
|---|---:|---:|---:|
| Mean shaft–hosel distance, model m | 0.7502 | 0.7223 | 0.7302 |
| Distance standard deviation, model m | 0.1099 | 0.0818 | <0.000001 |
| Minimum distance, model m | 0.5418 | 0.3258 | 0.7302 |
| Maximum distance, model m | 1.3091 | 0.9260 | 0.7302 |
| Median projected-axis disagreement | 2.795° | 2.615° | 0.690° |
| 90th percentile projected-axis disagreement | 12.129° | 10.629° | 1.979° |
| Median weak-perspective projected-length relative error | 19.53% | 20.21% | 6.71% |
| 90th percentile weak-perspective projected-length relative error | 53.77% | 53.19% | 14.82% |

Axis disagreement compares the predicted camera-XY shaft vector against the
cached detected 2D vector on the same 170 jointly detected frames. It uses an
orthographic direction approximation, not calibrated perspective reprojection.
The earlier audit used smoothed input directions; those numbers are not directly
comparable to this table, which uses the detected input directions.

Projected-length relative error is `abs(s * length(predicted_vector.xy) -
length(detected_vector_2d)) / length(detected_vector_2d)`, with both quantities in
normalized image units and the body-derived weak-perspective scale described
above. It measures agreement under this camera approximation, not true physical
reprojection accuracy. Very short projected segments can inflate relative error.

The input/window changes alone reduce the maximum length spike but introduce a
shorter minimum. They are not sufficient by themselves. The final fixed length
is true by construction, and closer agreement to 2D observations does not prove
correct out-of-plane depth. The longest missing interval still has no direct
club observations. No measured club length or ground-truth 3D was supplied.

The solver converged in 37 evaluations, using 170 reliable pairs and an inferred
marker distance of 0.730154 m. Its weak-perspective median scale was 0.449729
normalized-image units per model metre.

## Verification and outputs

- All 17 tests passed with `python -m unittest discover -s tests -v`, including
  five refinement tests and the 12 existing tests.
- Tests cover angular gap reconstruction, preservation of observed endpoints,
  exact 3D segment length, body preservation, head-marker offset preservation,
  projection consistency on synthetic data, insufficient observations, invalid
  lengths, NaN frames and window blending.
- The output contains 364 written and independently decoded video frames.
- P1–P10 stills were exported with the original reviewed indices.
- The constraint left body/shaft coordinates unchanged relative to its MixSTE
  input. Raw 2D observations and validity flags were preserved.
- SHA-256 verification confirmed all 47 files in the source run were unchanged.

Output folder:
`runs/golf_pose_3d_club_refined_20261027/b85ed0b4db566e63bce190824c2f2196`.

It contains the AVI, impact preview, P1–P10 stills, NPZ arrays, workflow JSON,
`validation.json`, `source_sha256.json` and `length_comparison.png`.

## Running fresh input

```powershell
python -m golf_workflows.golf_pose_3d pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4 --pose-backend vitpose-huge --lifter 17+5 --pose-cpu-threads 8 --club-refinement --positions-json runs/golf_pose_3d_vitpose_huge_17plus5/b85ed0b4db566e63bce190824c2f2196/selected_frames.json --output-dir runs/golf_pose_3d_club_refined_fresh
```

Add `--club-marker-length-m DISTANCE` only with the measured distance between
the intended shaft and hosel marker locations, in metres. Do not substitute a
nominal driver length for this marker distance. Omit `--club-refinement` to
retain the previous lifting behavior.

## Remaining accuracy work

The most useful next evidence is labeled shaft/hosel positions and a measured
marker distance. Calibrated multi-view or known-camera footage is needed to
validate depth and physical projected length. A GolfPose-compatible five-point
2D club model would address the missing head landmarks; this change does not
train or import one. Endpoint identity rejection, better shaft-line extraction
and supervision for difficult occlusions remain detector-stage improvements.

## Follow-up: swing interval refinement (2026-10-27)

The recent `w03` and `8b83acf159769157da0baa709a1fe55b` outputs exposed two
independent problems: early club-trajectory speed dips truncated the lifted
interval, and ball correction subsequently selected an impact outside that
already-lifted interval. The 3D panel therefore displayed `outside swing` during
the actual action. The label is temporal, not an inside-out/outside-in diagnosis.

### Changes

- Added `detect_refined_phases` in `golf_workflows/swing_phases.py`, enabled by the
  existing `--club-refinement` option (on by default in `workflow.py video`).
- Timing now applies the 2D angular club preparation before choosing the 3D
  interval. The rigid 3D fit remains after lifting, avoiding a circular dependency.
- Club endpoint pairs farther than 0.8 median torso lengths from the wrists at
  the grip are excluded from the phase track. This only changes timing inputs;
  the raw observations and validity flags are preserved.
- Body-relative wrist motion corroborates the club track. Root subtraction
  reduces camera-translation effects; median root-to-thorax normalization reduces
  image-scale dependence. Wrist height peaks/troughs with 25% of the sequence's
  height-span prominence identify a backswing rise and subsequent return.
- A genuinely observed club pair can refine contact within about 0.12 seconds of
  the wrist return only when the head also returns near its address location.
- Address includes a short pre-roll before sustained wrist displacement. Finish
  requires raised hands and a roughly 0.25-second quiet hold, followed by a short
  post-roll. The quiet threshold is 18% of the 95th-percentile wrist speed, with
  a minimum of 0.08 torso units/second. Without a hold, retain the clip's tail
  and record that uncertainty instead of clipping at the first speed dip.
- Ball correction runs before interval selection/lifting. For corroborated
  phases, search only within about 0.2 seconds of the wrist return and reject
  head-to-ball distances exceeding 0.6 torso lengths. An initial bounded-search
  trial still accepted false club points hundreds of pixels away; this distance
  gate was added after reviewing those results.
- The legacy option also applies ball correction before lifting and extends the
  interval when needed. Reviewed frame indices remain authoritative.
- Refined automatic P-position selection reserves intermediate frames for
  P2/P6/P8/P10. In particular, P9 cannot consume the first frame after impact,
  leaving no possible P8. The same geometric criteria remain; unsupported
  positions are not replaced with uniformly spaced frame indices.
- Run JSON now includes `phase_detection` diagnostics: method, observation
  counts, wrist evidence, finish rule, fallback and ball-hint acceptance.

### Replay and results

The actual video processor was exercised on four recent clips using their saved
2D observations. ViTPose was not repeated. Saved same-video ball locations were
reused; a constant detection box recovered the original normalization affine.
All 3D poses, intervals and automatic P selections were recomputed. Source runs
were SHA-256 checked and remained unchanged.

| Clip | Previous lifted interval | Refined interval | Refined impact | Ten-position result |
| --- | --- | --- | --- | --- |
| w01 | 19-52 | 21-57 | 45 | Complete; 92.2/100 |
| w02 | 12-49 | 15-58 | 45 | Complete; 92.1/100 |
| w03 | 3-13 | 5-51 | 38 | Complete; 90.2/100 |
| 8b83acf159769157da0baa709a1fe55b | 39-50 | 41-93 | 65 | P5/P6 remain unavailable |

All indices are zero-based. These are observed algorithm changes, not
ground-truth timing-accuracy measurements. The previous w03 impact was frame 14
and the other failed clip's impact was frame 144, both outside their old lifted
intervals. The new phases are ordered and lie inside the recomputed intervals.

The final AVI files independently decoded all 58, 59, 52 and 156 frames,
respectively. Every frame inside each new interval has finite 3D coordinates.
Raw 2D arrays and validity flags were preserved. All 27 tests passed, including
missing-club timing, image-scale/camera-translation invariance, bounded/distance-
gated ball correction and intermediate-frame reservation.

Outputs: `runs/phase_refinement_20261027/final/`. Each completed clip includes
ten source/3D images, the explicit ten-pose bundle and `scoring/score.json`.
`results.json` summarizes all four clips. The long club-gap clip remains flagged
for review and has no complete ten-position score. Earlier trial outputs are
retained one directory level above `final`.

The completed w03 result was also imported through `workflow.py video --from-run`
to `runs/workflows/videos/w03_phase_refined`, making it the latest completed
video bundle. The normal `workflow.py score` command was then run against it.

Reproducible replay scripts are `runs/phase_refinement_20261027/reprocess.py`
and `update_positions.py`. Use a new `--output-dir` for a replay rerun.

### Remaining limitations

This is a single-swing heuristic, not a trained temporal action model. Long gaps,
wrong-person body tracking, camera zoom/rotation, multiple swings, partial clips
and unreliable club geometry can still require review. Conservative clip-end
retention improves coverage but does not establish the exact finish time.
The P5/P6 failure in the long club-gap clip is a landmark/pose-selection issue
that interval refinement alone cannot resolve. The successful posture scores
measure agreement with reference angles, not timing or reconstruction accuracy.
