"""Exercise the actual video processor with cached, unchanged 2D observations."""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from golf_workflows import golf_pose_3d as workflow
from golf_workflows.cli import export_identified_frames,load_identified_frames
from golf_workflows.compare_standard_frames import compare
from golf_workflows.trace_swing_plane import ClubSample


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'final')
    args=parser.parse_args()
    torch.set_num_threads(8)
    model=workflow.load_lifter(workflow.CHECKPOINT_22)
    base=args.output_dir.resolve()
    base.mkdir(parents=True,exist_ok=True)
    refs=Path(json.loads((ROOT/'runs/workflows/latest_references.json').read_text())['folder'])
    records={}
    for stem in ['w01','w02','w03','8b83acf159769157da0baa709a1fe55b']:
        source=ROOT/'runs/workflows/videos'/stem
        path=next(source.glob('*_3d.npz'))
        old=json.loads(path.with_suffix('.json').read_text())
        if old.get('joint_count')!=22:
            raise ValueError('Replay checkpoint and cached joint count must match')
        hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
        data=np.load(path)
        raw,valid=data['keypoints_2d'].copy(),data['keypoints_2d_valid'].copy()
        filled=workflow.fill_gaps(raw,valid)
        x=data['keypoints_2d_filled'][:,:17].reshape(-1,2).astype(float)
        y=data['keypoints_2d_normalized'][:,:17].reshape(-1,2).astype(float)
        xc,yc=x-x.mean(0),y-y.mean(0)
        k=float(np.sum(xc*yc)/np.sum(xc**2))
        center=x.mean(0)-y.mean(0)/k
        height=workflow.CANVAS_H*workflow.GOLFER_FRACTION/(k*workflow.CANVAS_W/2)
        box=np.array([center[0]-height/2,center[1]-height/2,center[0]+height/2,center[1]+height/2])
        confidence=data['club_confidence'] if 'club_confidence' in data else np.ones(len(raw))
        samples=[]
        for f in np.flatnonzero(valid[:,18]):
            samples.append(ClubSample(int(f),raw[f,17].copy() if valid[f,17] else None,raw[f,18].copy(),
                                      float(confidence[f]),bool(samples and f-samples[-1].frame<=6)))
        swing=workflow.Swing2D(len(raw),float(data['fps']),tuple(data['frame_size']),raw,valid,filled,
                              np.repeat(box[None],len(raw),axis=0),samples,(filled[:,13,1]+filled[:,16,1])/2)
        out=base/stem
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'Output must be empty: {out}')
        pose_model=SimpleNamespace(metadata=copy.deepcopy(old.get('pose_backend',{})))
        video=ROOT/old['source']
        # Reuse the same video's saved detections, not the old 3D poses/phases.
        with patch.object(workflow,'detect_2d',return_value=swing),patch.object(workflow,'track_ball',return_value=copy.deepcopy(old['ball'])):
            workflow.process(video,pose_model,None,model,handedness=old['handedness'],view=old['view'],
                             club_refinement=True,output_folder=out)
        newpath=out/path.with_suffix('.json').name
        new=json.loads(newpath.read_text())
        new['checkpoint_selection']['fresh_2d_inference']=False
        new['phase_replay']={'source':str(source),'confidence_source':'saved_detector_confidence' if 'club_confidence' in data else 'binary_validity_only',
                             'ball_source':'cached_same_video','boxes':'constant box reproducing cached normalization affine'}
        newpath.write_text(json.dumps(new,indent=2))
        generated=np.load(out/path.name)
        np.testing.assert_array_equal(generated['keypoints_2d'],raw)
        np.testing.assert_array_equal(generated['keypoints_2d_valid'],valid)
        a,b=new['lift_window']
        assert np.isfinite(generated['keypoints_3d'][a:b+1]).all()
        assert a < new['phases']['top'] < new['phases']['impact'] < b
        cap=cv2.VideoCapture(str(out/(stem+'_3d.avi')));decoded=0
        while cap.read()[0]:decoded+=1
        cap.release()
        assert decoded==len(raw)
        result={'old_phases':old['phases'],'new_phases':new['phases'],'decoded_frames':decoded,'source_files_unchanged':len(hashes)}
        try:
            export_identified_frames(out)
            selected=load_identified_frames(out)
            scores=compare(out,refs,out/'scoring',selected)
            result.update(ten_frame_export=True,score=scores['pose_score'],positions=new['positions'])
        except ValueError as error:
            result.update(ten_frame_export=False,selection_error=str(error),positions=new['positions'])
        for name,digest in hashes.items():
            assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
        (out/'replay_validation.json').write_text(json.dumps(result,indent=2))
        records[stem]=result
        (base/'results.json').write_text(json.dumps(records,indent=2))
        print(json.dumps({stem:result},indent=2),flush=True)


if __name__=='__main__':
    main()
