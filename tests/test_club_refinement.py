import unittest
import numpy as np
from unittest.mock import patch
from types import SimpleNamespace
from golf_workflows.club_refinement import prepare_club_inputs, refine_club_3d


class ClubRefinementTests(unittest.TestCase):
    def test_window_blending_removes_hard_context_handoff(self):
        import torch
        from golf_workflows import golf_pose_3d as workflow
        model=SimpleNamespace(Spatial_pos_embed=torch.zeros(1,19,1))
        points=np.zeros((281,19,2),np.float32)
        points[:,0,0]=np.arange(281)
        def window_prediction(model, window):
            return np.full((243,19,3),window[0,0,0],np.float32)
        with patch.object(workflow,'lift_window',side_effect=window_prediction):
            old=workflow.lift_sequence(model,points)
            blended=workflow.lift_sequence(model,points,blend_windows=True)
        self.assertEqual(float(np.max(np.abs(np.diff(old[:,0,0])))),38)
        self.assertLess(float(np.max(np.abs(np.diff(blended[:,0,0])))),1)
        self.assertTrue(np.isfinite(blended).all())

    def test_rotating_gap_preserves_arc_and_observed_endpoints(self):
        raw = np.zeros((3,22,2))
        raw[:,18] = [[1,0],[np.nan,np.nan],[0,1]]
        valid = np.ones((3,22),bool)
        valid[1,17:19] = False
        valid[:,19:] = False
        raw[:,19:] = np.nan
        out, weight = prepare_club_inputs(raw, np.zeros_like(raw), valid)
        np.testing.assert_allclose(out[[0,2],17:19],raw[[0,2],17:19],atol=1e-8)
        self.assertAlmostEqual(np.linalg.norm(out[1,18]-out[1,17]),1)
        np.testing.assert_allclose(out[:,19],out[:,18])
        self.assertEqual(weight[1],0)
        self.assertFalse(valid[:,19:].any())

    def test_rigid_fit_preserves_body_and_moves_extra_markers_with_head(self):
        rng = np.random.default_rng(10)
        p = rng.normal(0,.2,(15,22,3))
        p[:,17] = [0,0,0]
        p[:,18] = np.column_stack([np.linspace(.6,.9,15),np.zeros(15),np.full(15,.3)])
        q = p[:,:,:2]*2
        valid = np.ones((15,22),bool)
        out, meta = refine_club_3d(p,q,valid,length_m=1)
        self.assertTrue(meta['applied'])
        np.testing.assert_allclose(np.linalg.norm(out[:,18]-out[:,17],axis=1),1,atol=1e-7)
        np.testing.assert_array_equal(out[:,:18],p[:,:18])
        np.testing.assert_allclose(out[:,19]-out[:,18],p[:,19]-p[:,18],atol=1e-7)
        self.assertLess(np.mean(np.abs((out[:,18]-out[:,17])[:,0]-(q[:,18]-q[:,17])[:,0]/2)),.04)

    def test_insufficient_observations_skip_and_invalid_length_rejected(self):
        p = np.ones((4,19,3))
        q = np.ones((4,19,2))
        valid = np.zeros((4,19),bool)
        out, meta = refine_club_3d(p,q,valid)
        self.assertFalse(meta['applied'])
        np.testing.assert_array_equal(out,p)
        for length in [0,-1,np.nan]:
            with self.assertRaises(ValueError):
                refine_club_3d(p,q,valid,length_m=length)

    def test_nan_frames_stay_nan(self):
        rng=np.random.default_rng(2)
        p=rng.normal(size=(10,19,3))
        p[[0,9]]=np.nan
        q=np.nan_to_num(p[:,:,:2])*2
        valid=np.ones((10,19),bool)
        out,meta=refine_club_3d(p,q,valid)
        self.assertTrue(meta['applied'])
        self.assertTrue(np.isnan(out[[0,9]]).all())


if __name__ == '__main__':
    unittest.main()
