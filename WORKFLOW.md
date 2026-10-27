# Three golf posture workflows

Run each stage with one command from `J:\Golf_Posture`:

```powershell
python workflow.py references
python workflow.py video pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4
python workflow.py score
```

Scoring uses the latest successfully completed reference and video workflows.
Each command accepts `--output-dir`. Run `python workflow.py <stage> --help`
for options. `python -m golf_workflows` is an alternative launcher.

## 1. Build ten references

`references` reads the ten JPG sheets in `pro_refs/Standard_10_frames`, crops
face-on/down-the-line/rear views, runs body and club inference, lifts the views
with MixSTE and applies the existing alignment/fusion. Face-on supplies the
reference space; the aligned down-the-line pose supplies its side component.
The rear view is retained for inspection.

Output: `runs/workflows/references/`, with `workflow.json` and ten position
folders. Each contains `reference_3d.jpg`, `joints.npz` (numerical coordinates),
`joints.json` and camera images/poses. Fresh inference also writes `run_summary.json`.
Completion requires one finite reference and readable image per P1-P10 position.

Reuse completed references without new inference:

```powershell
python workflow.py references --from-run runs/standard_frames_vitpose_huge_17plus5_20261027
```

## 2. Identify ten video frames

`video` performs body inference, club segmentation, phase identification,
MixSTE lifting and P1-P10 selection. The new launcher defaults to ViTPose-Huge,
GolfPose 17+5 and club refinement. Heel/toe inputs remain unmeasured and imputed.
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

Use `--no-club-refinement`, `--lifter 17+2` or `--pose-backend yolo26n-pose`
for the earlier methods. A measured marker length requires refinement.

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
All 22 tests passed with `python -m unittest discover -s tests -v`.
