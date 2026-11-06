from pathlib import Path
p=Path('golfpose_workflow/README.md');s=p.read_text(encoding='utf-8')
s=s.replace('This workflow always runs fresh 2D inference with one checkpoint:', 'This workflow runs fresh 2D pose inference with one pose checkpoint:')
s=s.replace('It uses full-frame single-golfer crops, UDP/DARK decoding, flip averaging,\nconfidence >= 0.10 and in-frame validity. Raw coordinates and confidence are saved.\nNo YOLO body detector, club segmenter or separate club pose model supplies 2D poses.', '''Before pose inference, `golfpose_checkpoints/golfpose_detector_1cls_yolox_s.pth`
locates the single `golfer` class. The selected box supplies the 192x256 UDP crop
with the existing 1.25 expansion. UDP/DARK decoding maps coordinates directly into
the original image; flip averaging, confidence >= 0.10 and in-frame validity remain.
Raw 22-joint coordinates and confidence are saved.

For video, overlapping detections retain the golfer identity; on detector gaps,
pose inference uses the last detected ROI and records `tracked-detector-roi`.
An initial detector miss produces no observed pose. Video never falls back to a
full-frame pose crop. Still reference panels can fall back to the complete panel
when detection fails, recorded as `reference-panel-fallback`. The detector and
pose confidence values remain separate; a tracked ROI is not marked as a new detection.
There is no separate club pose model in this workflow.''')
s=s.replace('Validation on 2026-11-03: all 60 tests passed.', 'Validation before the detector revision on 2026-11-03: all 60 tests passed.')
p.write_text(s,encoding='utf-8')
p=Path('WORKFLOW.md');s=p.read_text(encoding='utf-8')
intro='''## YOLOX detector routing revision (2026-11-03)

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

'''
s=s.replace('# Golf posture workflows\n\n','# Golf posture workflows\n\n'+intro,1)
s=s.replace('Validation: 60 tests passed; fresh inference', 'Validation before the detector revision: 60 tests passed; fresh inference',1)
s=s.replace('  References keep `golfer-box` coarse-to-fine crops without club segmentation.\n  Videos keep the existing `segmentation` crop backend; segmentation supplies the\n  crop, while the five landmark coordinates come from club-only ViTPose.', '  Two-class YOLOX supplies person/club boxes in references and videos; club-only\n  ViTPose supplies the five landmark coordinates. Golfer-box refinement is the\n  fallback when no associated club detection is available.')
s=s.replace('Validation: all 56 tests pass.', 'Validation before the detector revision: all 56 tests passed.',1)
s=s.replace('Reference crops default to `golfer-box`.', 'Reference crops also default to `official-yolox-2cls`.')
s=s.replace('The `segmentation` crop backend supplies video club crops with the current\nYOLO26n body default.', 'The explicit `segmentation` crop option remains available. Two-class YOLOX\nsupplies default video club crops.')
p.write_text(s,encoding='utf-8')
