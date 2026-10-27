"""Re-export checkpoint selection after ordering fix; retain computed 3D/video."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from golf_workflows import golf_pose_3d as workflow
from golf_workflows.cli import export_identified_frames,load_identified_frames
from golf_workflows.compare_standard_frames import compare
from golf_workflows.swing_phases import find_checkpoint_frames,SwingPhases

base=Path(__file__).resolve().parent/'final'
records=json.loads((base/'results.json').read_text())
refs=Path(json.loads((ROOT/'runs/workflows/latest_references.json').read_text())['folder'])
for stem,row in records.items():
    out=base/stem
    p=next(out.glob('*_3d.npz'))
    data=np.load(p)
    meta=json.loads(p.with_suffix('.json').read_text())
    phases=SwingPhases(**meta['phases'])
    positions=find_checkpoint_frames(data['keypoints_3d'],phases.address,phases.top,phases.impact,meta['lead_side'],reserve_positions=True)
    swing=workflow.Swing2D(len(data['keypoints_3d']),float(data['fps']),tuple(data['frame_size']),
                          data['keypoints_2d'],data['keypoints_2d_valid'],data['keypoints_2d_filled'],
                          np.full((len(data['keypoints_3d']),4),np.nan),[],np.zeros(len(data['keypoints_3d'])))
    meta['positions']=positions
    meta['position_selection']='geometric_criteria_with_reserved_intermediate_frames'
    meta['position_stills']=workflow.save_position_stills(ROOT/meta['source'],swing,positions,out,phases,meta['ball']['p1_xy'])
    p.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    row['positions']=positions
    try:
        export_identified_frames(out)
        result=compare(out,refs,out/'scoring',load_identified_frames(out))
        row.update(ten_frame_export=True,score=result['pose_score'])
        row.pop('selection_error',None)
    except ValueError as error:
        row.update(ten_frame_export=False,selection_error=str(error))
    (out/'replay_validation.json').write_text(json.dumps(row,indent=2))
    print(stem,positions,row.get('score'),row['ten_frame_export'])
(base/'results.json').write_text(json.dumps(records,indent=2))
