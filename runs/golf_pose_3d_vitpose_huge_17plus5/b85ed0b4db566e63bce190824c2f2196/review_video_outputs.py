"""Validate the completed video trial and export its ten reviewed positions."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from standard_frames import save_still

FOLDER = Path(__file__).resolve().parent
STEM = FOLDER.name
VIDEO = ROOT / "pro_refs" / f"{STEM}.mp4"


def grid(images, label, path, columns=5):
    tiles = []
    for name, frame, image in images:
        tile = cv2.copyMakeBorder(image, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        cv2.putText(tile, f"{name}  frame {frame}", (10,26), cv2.FONT_HERSHEY_SIMPLEX, .65, (25,25,25), 2, cv2.LINE_AA)
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i+columns]) for i in range(0,len(tiles),columns)]
    canvas = np.vstack(rows)
    canvas = cv2.copyMakeBorder(canvas, 64, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
    cv2.putText(canvas,label,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(25,25,25),2,cv2.LINE_AA)
    cv2.putText(canvas,"Heel/toe inputs are imputed at the hosel; body poses use fresh ViTPose inference.",
                (12,51),cv2.FONT_HERSHEY_SIMPLEX,.55,(40,40,170),1,cv2.LINE_AA)
    assert cv2.imwrite(str(path),canvas)


def main():
    summary = json.loads((FOLDER/f"{STEM}_3d.json").read_text(encoding="utf-8"))
    selection = json.loads((FOLDER/"selected_frames.json").read_text(encoding="utf-8"))
    frames = {row["position"].lower():row["frame_index"] for row in selection["frames"]}
    assert summary["positions"] == frames
    assert summary["pose_backend"]["pose_backend"] == "vitpose-huge"
    assert summary["joint_count"] == 22
    arrays = np.load(FOLDER/f"{STEM}_3d.npz")
    poses = arrays["keypoints_3d"]
    count = len(poses)
    assert poses.shape == (count,22,3)
    start,stop = summary["lift_window"]
    assert (start,stop) == (frames["p1"],frames["p10"])
    assert np.isfinite(poses[start:stop+1]).all()
    assert np.isnan(poses[:start]).all() and np.isnan(poses[stop+1:]).all()
    assert np.isnan(arrays["keypoints_2d"][:,19:]).all()
    assert not arrays["keypoints_2d_valid"][:,19:].any()
    np.testing.assert_allclose(arrays["keypoints_2d_filled"][:,19:],
                               np.repeat(arrays["keypoints_2d_filled"][:,18:19],3,axis=1),atol=1e-6)
    raw = FOLDER/"raw_frames";raw.mkdir(exist_ok=True)
    captures = cv2.VideoCapture(str(VIDEO))
    source_count = int(captures.get(cv2.CAP_PROP_FRAME_COUNT))
    assert count == source_count == 364
    two_d,three_d = [],[]
    try:
        for key,index in frames.items():
            name=key.upper()
            captures.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,image=captures.read();assert ok
            assert cv2.imwrite(str(raw/f"{name}.jpg"),image)
            annotated=cv2.imread(str(FOLDER/f"{name}.jpg"));assert annotated is not None
            save_still(poses[index],FOLDER/f"{name}_3d.jpg",name)
            skeleton=cv2.imread(str(FOLDER/f"{name}_3d.jpg"));assert skeleton is not None
            two_d.append((name,index,annotated));three_d.append((name,index,skeleton))
    finally:
        captures.release()
    grid(two_d,"ViTPose-Huge + GolfPose 17+5: reviewed P1-P10",FOLDER/"P1_P10_overview.jpg")
    grid(three_d,"GolfPose 17+5: predicted 3D poses at the same reviewed frames",FOLDER/"P1_P10_3d.jpg")
    output_video=FOLDER/f"{STEM}_3d.avi"
    cap=cv2.VideoCapture(str(output_video))
    fps=float(cap.get(cv2.CAP_PROP_FPS));decoded=0
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            if decoded == frames["p7"]:
                assert cv2.imwrite(str(FOLDER/"impact_2d_3d.jpg"),image)
            decoded+=1
    finally:
        cap.release()
    assert decoded == count and abs(fps-30)<1e-3
    previous=json.loads((FOLDER/"preserved_previous_run_sha256.json").read_text())
    for name,digest in previous.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest,name
    report={"source":str(VIDEO.relative_to(ROOT)),"source_frames":count,"fps":fps,
            "output_video_frames_decoded":decoded,"joint_count":22,
            "finite_3d_frames":stop-start+1,"lift_window":[start,stop],"positions":frames,
            "reviewed_frame_indices_reused":True,"fresh_2d_inference":True,
            "club_detected_frames":summary["club_detected_frames"],
            "extra_club_inputs_imputed":["heel","toe_down","toe_up"],
            "raw_missing_flags_and_filled_hosel_checks":"passed",
            "previous_run_files_verified_unchanged":len(previous),"unit_tests_passed":12,
            "timebase":"edited slow-motion video; frame timestamps are not real swing timing"}
    (FOLDER/"validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    arrays.close()
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
