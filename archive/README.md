# 已归档脚本

两个工作流（`workflow.py`、`python -m golfpose_workflow`）、测试以及
`golf_workflows.benchmark` 都不会导入这些文件。将它们保留在这里，是为了便于恢复。

`docs_en/` 目录保存了项目 Markdown 文件（README.md、WORKFLOW.md、SCORING_METHODOLOGY.md、SHAFT_TRACKING_IMPROVEMENT_LOG.md、golfpose_workflow/README.md、archive/README.md）的英文原版，这些文件是在 2026-11-05 翻译为中文之前保存的。

`docs_zh_20261106/` 保存了 2026-11-06 精简前的中文 `README.md`、`WORKFLOW.md`、`FRAME_RECOGNITION.md` 和 `golfpose_workflow_README.md`（含已删除的早期实验与旧后端说明）。

## `scripts/`（已纳入 git 跟踪）

| 文件 | 用途 |
|---|---|
| `prepare_golf_club_seg.py` | 为球杆分割模型构建 `datasets/golf-club-seg` |
| `train_golf_club_seg.py` | 训练得到 `runs/segment/golf_club_train/weights/best.pt` |
| `replay_club_refinement.py` | 2026-10-27 基于缓存 2D 结果的球杆精修回放（见 `SHAFT_TRACKING_IMPROVEMENT_LOG.md`）；已被 `python -m golf_workflows.benchmark replay` 取代 |

这些脚本通过 `Path(__file__).parent` 定位项目，因此恢复后请在
仓库根目录下运行：

```powershell
git mv archive/scripts/train_golf_club_seg.py .
```

## `scratch/`（原为 `.tmp/*.py`）

这些是开发期间（2026-11-02 至 2026-11-05）编写的一次性检查和验证临时脚本，例如 DEKR/HRNet 对比、
检测器路由审查、逐片段诊断、`replay_phases.py`（现为 `golf_workflows.benchmark`）
以及 `eval_measurements.py`（杆身条纹/掩码准确率）。它们需要在 `.tmp/` 下运行，并将
`PYTHONPATH` 设置为仓库根目录：

```powershell
Move-Item archive\scratch\replay_phases.py .tmp\
```

`.tmp/` 数据文件夹（例如 `.tmp/club_005`）没有移动；基准测试标签引用了它们。
