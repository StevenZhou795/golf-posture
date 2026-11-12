"""Load only RTMW's official modules, avoiding unrelated CUDA/training imports.

MMPose's package initializers eagerly import every architecture and dataset.
On Windows mmcv-lite has no compiled deformable-convolution operators, which
RTMW does not use. Namespace packages below bypass only those initializers;
every executed model, transform, codec and estimator is unmodified upstream.
No operator is mocked or replaced. This loader is inference-only.
"""
import importlib
from importlib.machinery import ModuleSpec
import sys
from types import ModuleType


def register(upstream):
    import mmpose
    for suffix in ('models', 'models.backbones', 'models.necks', 'models.heads',
                   'models.pose_estimators', 'datasets', 'datasets.datasets',
                   'datasets.transforms', 'evaluation', 'apis'):
        name = 'mmpose.' + suffix
        if name in sys.modules:
            continue
        module = ModuleType(name)
        module.__path__ = [str(upstream / 'mmpose' / suffix.replace('.', '/'))]
        module.__package__ = name
        module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
        parent, _, child = name.rpartition('.')
        setattr(sys.modules[parent], child, module)
    from mmpose.models.heads.base_head import BaseHead
    sys.modules['mmpose.models.heads'].BaseHead = BaseHead
    from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
    sys.modules['mmpose.models.pose_estimators'].TopdownPoseEstimator = TopdownPoseEstimator
    for name in ('models.backbones.cspnext', 'models.necks.cspnext_pafpn',
                 'models.data_preprocessors', 'models.losses',
                 'datasets.transforms.loading', 'datasets.transforms.common_transforms',
                 'datasets.transforms.topdown_transforms', 'datasets.transforms.formatting'):
        importlib.import_module('mmpose.' + name)
    importlib.import_module('mmpose.models.heads.coord_cls_heads.rtmw_head')


def inference(model, image, bboxes=None):
    """Official 2D transforms and estimator; keypoints are restored image pixels."""
    import numpy as np
    import torch
    from mmengine.dataset import Compose, pseudo_collate
    if bboxes is None:
        h, w = image.shape[:2]
        bboxes = np.array([[0, 0, w, h]], np.float32)
    if len(bboxes) == 0:
        return []
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    samples = []
    for bbox in bboxes:
        sample = pipeline(dict(img=image, bbox=np.asarray(bbox, np.float32)[None],
                               bbox_score=np.ones(1, np.float32), **model.dataset_meta))
        samples.append(sample)
    with torch.inference_mode():
        return model.test_step(pseudo_collate(samples))
