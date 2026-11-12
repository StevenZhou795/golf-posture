"""Pinned upstream source and model loading. No model downloads at inference time."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REVISION = '759b39c13fea6ba094afc1fa932f51dc1b11cbf9'
UPSTREAM = HERE / 'vendor' / ('mmpose-' + REVISION)
CHECKPOINT = ROOT / 'models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth'
CONFIG = UPSTREAM / 'configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb704-270e_cocktail14-256x192.py'


def load_model(checkpoint=CHECKPOINT, threads=8):
    import torch
    import os
    # platformdirs otherwise puts YAPF's grammar cache outside the workspace.
    # This override applies only to this Python process, not Windows settings.
    cache = HERE / '.cache'
    cache.mkdir(exist_ok=True)
    os.environ.setdefault('WIN_PD_OVERRIDE_LOCAL_APPDATA', str(cache))
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    if not CONFIG.is_file():
        raise FileNotFoundError('Run rtmw_workflow/scripts/setup.ps1 to fetch pinned upstream source')
    sys.path.insert(0, str(UPSTREAM))
    import runpy
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.model.utils import revert_sync_batchnorm
    from .official import register
    register(UPSTREAM)
    from mmpose.registry import MODELS
    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    torch.set_num_threads(threads)
    source = runpy.run_path(str(CONFIG))
    config = Config(dict(model=source['model'], default_scope='mmpose',
        test_dataloader=dict(dataset=dict(pipeline=[
            dict(type='LoadImage'), dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=source['input_size']), dict(type='PackPoseInputs')]))))
    config.model.backbone.init_cfg = None
    init_default_scope('mmpose')
    model = revert_sync_batchnorm(MODELS.build(config.model))
    model.cfg = config
    model.dataset_meta = parse_pose_metainfo(dict(from_file=str(UPSTREAM/'configs/_base_/datasets/coco_wholebody.py')))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload.get('state_dict', payload)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    if payload.get('meta', {}).get('dataset_meta'):
        model.dataset_meta = payload['meta']['dataset_meta']
    expected = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    actual = [model.dataset_meta['keypoint_id2name'][i] for i in range(17)]
    if actual != expected or model.dataset_meta['num_keypoints'] != 133:
        raise ValueError('Checkpoint keypoint metadata does not match COCO-WholeBody 133')
    import hashlib
    model.workflow_metadata = dict(pose_backend='rtmw-x-2d', input_size_wh=list(source['input_size']),
        config=str(CONFIG.relative_to(UPSTREAM)), state_dict_items=len(state), strict_parameter_load=True,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        flip_test=bool(config.model.test_cfg.flip_test), native_3d_available=False)
    model.eval()
    return model
