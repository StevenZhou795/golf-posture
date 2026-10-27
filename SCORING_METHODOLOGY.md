# Standard-reference pose scoring methodology

This document specifies the scoring implemented by [compare_standard_frames.py](golf_workflows/compare_standard_frames.py), used by `python workflow.py score` to compare the saved P1–P10 poses with ten reference poses. New references are under `runs/workflows/references/`; earlier reference sets remain available through explicit paths. It describes the current implementation as of 2026-10-27.

The result is a 0–100 similarity score based on 14 posture angles at each available position. A larger score means the measured angles are closer to the reference. The score has no calibrated pass/fail threshold.

## 1. Inputs and position matching

The public scoring workflow reads the explicit P1-P10 bundle produced by
`python workflow.py video`, and a validated set of ten reference folders.

| Input | Data used |
| --- | --- |
| Video `identified_frames.npz` | Ten `keypoints_3d` poses, GolfPose joint names, position labels and original frame indices |
| Video `identified_frames.json` | Matching position/frame records, handedness and lead side |
| Reference `P<number>-*/joints.npz` | `reference_3d`: 19 or 22 joints × 3 coordinates |
| Matching `reference_3d.jpg` | Reference rendering used in comparison pictures |

Scores use saved 3D joint coordinates. Source-frame JPGs and `reference_3d.jpg`
visualize those poses; their pixels are not numerical scoring inputs. The bundle
must contain P1-P10 in order, finite coordinates, the expected joint order and
ten distinct increasing zero-based source indices. Its JSON and NPZ selections
must agree. Scoring does not identify frames again or choose better-scoring
neighbours. The video command can accept `--positions-json` for reviewed indices.

By default, the scorer uses the latest completed reference and video workflows,
recorded under `runs/workflows/latest_references.json` and `latest_video.json`.
Explicit folders can be supplied. Exactly one complete reference per P position
is required. With 22 joints, the same 14 criteria use the original body and
shaft/hosel indices; the three extra head points add no scoring criteria.

The lower-level comparator still supports older `*_3d.npz/json` runs. That legacy
path uses the saved `positions` dictionary, with phase fallbacks for P1, P4 and
P7. The public scoring command uses the explicit ten-pose bundle instead.

## 2. How the reference skeleton is built

[standard_frames.py](golf_workflows/standard_frames.py) crops each standard sheet into face-on, down-the-line, and rear views and lifts each view into 3D. A still image is repeated for the temporal model's input window.

The reference normally uses the face-on skeleton as its base. The down-the-line skeleton is normalized and aligned to it by trying mirrored/unmirrored configurations and yaw rotations in 15-degree steps. The alignment maximizing mean bone-direction cosine similarity is selected.

Let `F` be a normalized face-on joint, `D` the corresponding aligned down-the-line joint, and `a` the unit side axis derived from the down-the-line camera orientation. Fusion applies:

```text
reference_joint = F + dot(D - F, a) * a
```

This copies the donor's component along `a` while preserving the face-on components perpendicular to it. The rear view is retained for inspection and does not supply the fused reference. With only face-on or only down-the-line available, that view supplies the reference; rear alone does not produce a reference in the current implementation.

## 3. Normalization, coordinates, and handedness

Each user and reference pose is independently normalized:

```text
centered_joint = joint - root
scale = norm(thorax - root)
normalized_joint = centered_joint / scale
```

`root` is the hip-center joint. The scale is the root-to-thorax length, rather than the length of the joint named `spine`. A pose with any non-finite coordinate or a scale below `1e-6` is excluded. Normalization removes translation and uniform size differences; the angle calculations are already insensitive to uniform scaling.

Calculations use the saved camera coordinates, with upward direction `U = (0, -1, 0)` and horizontal plane XZ. Rendering may rearrange the axes for display; scoring uses the saved coordinates.

Lead side comes from `lead_side` when it is `left` or `right`. Otherwise, a right-handed golfer has left lead side, and a left-handed golfer has right lead side. The opposite side is the trail side. The user's lead-side choice is applied to both user and reference calculations; the scorer does not automatically mirror a reference of opposite handedness.

## 4. Angle definitions

Use the following notation on each normalized skeleton:

```text
d(A, B) = (B - A) / norm(B - A)
angle(v, w) = degrees(acos(clip(dot(v, w), -1, 1)))
joint_angle(A, B, C) = angle(d(B, A), d(B, C))
horizontal(v) = normalize((v.x, 0, v.z))
signed_up(v, axis) = degrees(atan2(dot(v, axis), dot(v, U)))
```

Directions with non-finite or near-zero lengths are unavailable; the vector length cutoff is `1e-8`.

The shared directions are:

```text
S = d(root, thorax)
T = normalize(d(left_hip, left_knee) + d(right_hip, right_knee))
L = horizontal(lead_shoulder - trail_shoulder)
F = normalize(cross(U, L))
```

`T` averages the two unit thigh directions. `L` points horizontally toward the lead shoulder. `F` is the forward axis constructed from `U` and `L`.

All 14 measurements below are attempted at every position; there are no position-specific weights or different criterion lists in this scorer.

| Parameter in `score.json` | Exact measurement |
| --- | --- |
| Alignment | `degrees(acos(abs(dot(horizontal(left_shoulder - right_shoulder), horizontal(left_hip - right_hip)))))`, with cosine clamping. The absolute dot product treats the lines as unoriented. |
| Hip Bend | `angle(S, T)` |
| Hip Depth | `angle(T, U)` |
| Knee Bend | Mean of left and right `joint_angle(hip, knee, foot)`; both must be available. |
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

These names are labels for the geometric definitions above. For example, Hip Depth is an angle, not a measured displacement; Stance Width is an angle, not a distance; Knee Bend is the internal knee angle, not `180 - internal_angle`.

## 5. Convert an angle difference to a score

For each parameter available in both skeletons, calculate the shortest circular angular difference:

```text
gap = abs(user_angle - reference_angle) % 360
if gap > 180:
    gap = 360 - gap

parameter_score = round(100 * (1 - gap / 180), 1)
```

| Angular difference | Parameter score |
| --- | ---: |
| 0 degrees | 100.0 |
| 10 degrees | 94.4 |
| 30 degrees | 83.3 |
| 90 degrees | 50.0 |
| 180 degrees | 0.0 |

The same linear 180-degree scale is used for every parameter, including parameters with smaller natural ranges. It is relatively tolerant of moderate differences: a high score does not imply all angles are nearly identical.

## 6. Weighting, rounding, and missing data

Each available parameter comparison has equal weight:

```text
position_score = round(mean(available parameter scores at that position), 1)
overall_pose_score = round(mean(all available parameter scores across P1–P10), 1)
```

Parameter scores are rounded to one decimal before aggregation. The overall score is computed from those parameter scores directly, rather than by averaging the displayed, rounded position scores.

With all 14 parameters available at all ten positions, the overall score averages 140 values. Every position then contributes equally before display rounding. With missing parameters, a position contributes in proportion to its number of available comparisons.

A missing reference, missing user frame selection, non-finite pose, or failed normalization causes the entire position to be recorded as `null`. Within an otherwise valid pose, an undefined parameter is omitted from both the parameter dictionary and the denominator. A position with no available comparisons is `null`; an overall result with no comparisons is also `null`. Missing values are not scored as zero. Saved frame indices must be within the user pose array's bounds.

The scorer does not apply detector-confidence weights or check whether a finite joint was originally detected, interpolated, or filled from another joint. A filled joint can therefore contribute to the score.

## 7. Display alignment and score invariance

For the comparison pictures, `best_yaw()` rotates the normalized user skeleton about the vertical axis in 15-degree steps and selects the rotation with the highest mean bone-direction cosine similarity to the reference.

The saved `yaw_deg` describes this display alignment. Parameter scores use the original normalized poses, before that display rotation. Their definitions are invariant to a common yaw rotation of a skeleton, because they use internal angles, horizontal relative alignment, and shoulder-relative axes. This does not remove camera pitch/roll, errors introduced during 3D reconstruction, or errors in reference fusion.

## 8. Execution and saved results

```powershell
python workflow.py score
# Or choose both inputs explicitly:
python workflow.py score runs/workflows/videos/b85ed0b4db566e63bce190824c2f2196 --references runs/workflows/references
```

Results go to a separate `runs/workflows/scores/<video-run>/` folder. Existing
folders are preserved by using a versioned name on subsequent runs.

```text
score.json
workflow.json
P1_user_3d.jpg ... P10_user_3d.jpg
P1_vs_reference.jpg ... P10_vs_reference.jpg
```

`score.json` records the overall score, equal weighting, original video frame
indices, per-position scores, the 14 parameter scores, input folders, criteria
names and coverage. Raw user/reference angles are not stored. The selected source
indices are preserved even though the exported bundle contains only ten poses.

The reorganization smoke test reused the completed refined video and ViTPose
17+5 references. It scored all ten positions and all 140 parameter comparisons,
producing **93.8/100**. This is a different input pair from the earlier
**92.9/100** baseline; both earlier output folders are preserved.

## 9. Interpretation and scope

This score measures agreement with the saved reference skeletons under the formulas above. It does not measure ball-striking quality, swing speed, timing, or clubface orientation, and it is not a probability or a percentage of correct technique. Accuracy depends on frame selection, 2D detection, gap filling, 3D lifting, and reference construction. The method has no empirical calibration against coaching assessments documented here.

Some measurements overlap: Trail Knee and Lead Knee repeat the individual knee angles already represented in Knee Bend. Equal weights therefore do not imply independent biomechanical evidence. A single aggregate score can also conceal a larger difference in one parameter; inspect the parameter scores and comparison images together.

The separate [score_swing.py](golf_workflows/score_swing.py) workflow compares a user with lifted reference videos using position-specific criteria and reports a separate clubhead-path score. Its criterion list and path score are not used by the standard-sheet method specified in this document.

## Implementation references

- [compare_standard_frames.py](golf_workflows/compare_standard_frames.py): posture definitions, circular difference, aggregation, display alignment, and output.
- [standard_frames.py](golf_workflows/standard_frames.py): normalization and multi-view reference construction.
- [score_swing.py](golf_workflows/score_swing.py): shared frame lookup, lead-side selection, vector normalization, and linear angle-score helper.
- [WORKFLOW.md](WORKFLOW.md): the surrounding detection, lifting, and comparison pipeline.
