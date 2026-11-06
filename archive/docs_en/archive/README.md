# Archived scripts

Neither workflow (`workflow.py`, `python -m golfpose_workflow`), the tests nor
`golf_workflows.benchmark` imports these files. They are kept here so they can be restored.

## `scripts/` (tracked in git)

| File | Purpose |
|---|---|
| `prepare_golf_club_seg.py` | Builds `datasets/golf-club-seg` for the club segmentation model |
| `train_golf_club_seg.py` | Trains `runs/segment/golf_club_train/weights/best.pt` |
| `replay_club_refinement.py` | 2026-10-27 club-refinement replay on cached 2D (see `SHAFT_TRACKING_IMPROVEMENT_LOG.md`); superseded by `python -m golf_workflows.benchmark replay` |

These scripts locate the project with `Path(__file__).parent`, so run them from the
repository root after restoring:

```powershell
git mv archive/scripts/train_golf_club_seg.py .
```

## `scratch/` (formerly `.tmp/*.py`)

These are one-off inspection and validation scripts from development (2026-11-02 to 2026-11-05), such as DEKR/HRNet comparisons,
detector-routing audits, per-clip diagnostics, `replay_phases.py` (now `golf_workflows.benchmark`)
and `eval_measurements.py` (shaft streak/mask accuracy). They expect to run from `.tmp/` with
`PYTHONPATH` set to the repository root:

```powershell
Move-Item archive\scratch\replay_phases.py .tmp\
```

The `.tmp/` data folders (for example `.tmp/club_005`) were not moved; the benchmark labels refer to them.
