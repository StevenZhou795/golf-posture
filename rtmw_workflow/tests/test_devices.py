import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtmw_workflow.device import resolve_devices, StagedLifter, move_detector


class DeviceTests(unittest.TestCase):
    def test_auto_and_explicit_stage_override(self):
        args = SimpleNamespace(device='auto', club_device='cpu')
        with patch('torch.cuda.is_available', return_value=True):
            resolve_devices(args)
        self.assertEqual((args.pose_device, args.aux_device, args.lifter_device), ('cuda',) * 3)
        self.assertEqual(args.club_device, 'cpu')

    def test_cpu_fallback_and_explicit_cuda_failure(self):
        with patch('torch.cuda.is_available', return_value=False):
            args = resolve_devices(SimpleNamespace(device='auto'))
            self.assertEqual(args.pose_device, 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                resolve_devices(SimpleNamespace(device='cuda'))

    def test_predictor_cache_is_cleared(self):
        model = Mock(overrides={})
        move_detector(model, 'cuda')
        model.to.assert_called_once_with('cuda')
        self.assertIsNone(model.predictor)
        self.assertEqual(model.overrides, dict(device='cuda', half=False))

    def test_lifter_releases_other_models_once_per_video_and_returns_cpu(self):
        model = torch.nn.Linear(2, 3)
        model.Spatial_pos_embed = torch.zeros(1, 19, 2)
        model.checkpoint_path = 'synthetic.bin'
        release = Mock()
        wrapped = StagedLifter(model, 'cpu', release)
        inputs = torch.zeros(2, 4, 19, 2)
        for _ in range(2):
            with torch.inference_mode():
                actual = wrapped(inputs)
                torch.testing.assert_close(actual, model(inputs))
            self.assertEqual(actual.device.type, 'cpu')
        release.assert_called_once()
        self.assertIs(wrapped.Spatial_pos_embed, model.Spatial_pos_embed)
        self.assertEqual(wrapped.checkpoint_path, model.checkpoint_path)
        wrapped.release()
        with torch.inference_mode():
            wrapped(inputs)
        self.assertEqual(release.call_count, 2)
