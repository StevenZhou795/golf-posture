from pathlib import Path
p=Path('WORKFLOW.md');s=p.read_text(encoding='utf8')
s=s.replace('joints. Reference alignment, view fusion, body/club-overlay rendering, video','joints. Reference alignment, body/club-overlay rendering, video')
s=s.replace('6. **融合参考。** 将各视角姿态归一化并对齐，以正面姿态为基础，\n   使用沿目标线视角替换其对应侧向分量，保存 `joints.npz:reference_3d`，\n   并渲染 `reference_3d.jpg`。背面结果保留供检查，不参与当前参考融合。','6. **选择整体参考。** 将各视角姿态归一化并对齐，在 face_on 与 down_the_line 之间按 17 个人体关节平均置信度选择较高者，完整保留其对齐后的骨架，不再替换另一视角的空间分量。保存 `joints.npz:reference_3d` 并渲染 `reference_3d.jpg`。背面仅供检查。')
s=s.replace('这种融合不是基于已标定相机的多视角三角测量','这种单视角选择与对齐不是基于已标定相机的多视角三角测量')
s=s.replace('with MixSTE and applies the existing alignment/fusion. Face-on supplies the\nreference space; the aligned down-the-line pose supplies its side component.','with MixSTE and aligns the poses in face-on space. The higher-confidence\nface-on or down-the-line body supplies the entire numerical reference.')
s=s.replace('17+5. All 22 native output channels feed the existing lifting/fusion pipeline.','17+5. All 22 native output channels feed the lifting and whole-pose selection pipeline.')
needle='Club overlay selection rule (2026-11-03):'
section='''Body-reference selection rule (2026-11-03): compare only `face_on` and
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
and the rule. In the examined P9 run, face-on scores 0.9271 versus 0.7799 for
down-the-line, so its complete skeleton is retained. The club-overlay donor
is selected independently by the following rule; it cannot change the body
or the numerical scoring reference.

Fresh reference generation and `standard_frames --rebuild` both apply this
rule. Rebuild uses saved 2D confidence/validity and 3D poses, regenerating the
body image and overlay together without neural-network inference. Existing
completed folders are not rewritten automatically; `--from-run` alone copies
stored results and does not apply this revision.

'''
s=s.replace(needle,section+needle)
p.write_text(s,encoding='utf8')
