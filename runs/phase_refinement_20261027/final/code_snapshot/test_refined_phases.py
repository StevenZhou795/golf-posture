import unittest
from types import SimpleNamespace

import numpy as np

from golf_workflows.swing_phases import detect_refined_phases,find_checkpoint_frames
from golf_workflows.golf_pose_3d import impact_from_ball


def swing_fixture():
    n=100
    points=np.zeros((n,19,2),float)
    points[:,8]=[0,-100]
    height=np.interp(np.arange(n),[0,15,38,52,65,72,99],[0,0,1.5,0,1.3,1.4,1.4])
    points[:,13,1]=points[:,16,1]=-100*height
    points[:,17]=(points[:,13]+points[:,16])/2
    points[:,18]=points[:,17]+[0,150]
    valid=np.ones((n,19),bool)
    return points,valid


class RefinedPhasesTests(unittest.TestCase):
    def test_postimpact_selection_reserves_a_frame_for_release_and_finish(self):
        poses=np.zeros((40,19,3))
        poses[:,11]=[1,0,0]
        poses[:,13]=[2,-.5,0]
        poses[:,18]=[1,0,0]
        poses[21,13]=[2,0,0]
        poses[22,13]=[2,-.01,0]
        poses[23,13]=[.1,-.2,0]
        poses[30:]=np.nan
        legacy=find_checkpoint_frames(poses,0,10,20)
        improved=find_checkpoint_frames(poses,0,10,20,reserve_positions=True)
        self.assertIsNone(legacy['p8'])
        self.assertEqual([improved[key] for key in ['p7','p8','p9','p10']],[20,21,22,23])

    def test_wrist_cycle_survives_missing_club_and_preserves_flags(self):
        points,valid=swing_fixture()
        valid[30:70,17:19]=False
        raw=points.copy();raw[30:70,17:19]=np.nan
        phases,details=detect_refined_phases(raw,valid,points,30)
        self.assertTrue(10<=phases.address<=20)
        self.assertTrue(34<=phases.top<=42)
        self.assertTrue(48<=phases.impact<=56)
        self.assertGreater(phases.follow_through,72)
        self.assertFalse(valid[30:70,17:19].any())
        self.assertEqual(details['method'],'refined_club_with_wrist_corroboration')

    def test_camera_translation_and_image_scale_do_not_change_interval(self):
        points,valid=swing_fixture()
        first,_=detect_refined_phases(points,valid,points,30)
        camera=np.column_stack([np.arange(len(points))*3,np.arange(len(points))*2])[:,None]
        transformed=points*2.7+camera
        second,_=detect_refined_phases(transformed,valid,transformed,30)
        self.assertEqual(first.as_dict(),second.as_dict())

    def test_ball_hint_cannot_move_impact_to_late_false_detection(self):
        points,valid=swing_fixture()
        base,_=detect_refined_phases(points,valid,points,30)
        phases,_=detect_refined_phases(points,valid,points,30,impact_hint=95)
        self.assertEqual(phases.impact,base.impact)
        self.assertGreater(phases.follow_through,phases.impact)
        close,_=detect_refined_phases(points,valid,points,30,impact_hint=53)
        self.assertEqual(close.impact,53)

    def test_ball_search_is_bounded_to_correlated_impact(self):
        points,valid=swing_fixture()
        points[:,18]=[100,100]
        points[53,18]=[1,0]
        points[95,18]=[0,0]
        swing=SimpleNamespace(points=points,valid=valid,frames=len(points))
        self.assertEqual(impact_from_ball(swing,[0,0],38,52,(48,56)),53)
        self.assertEqual(impact_from_ball(swing,[0,0],38,52),95)
        self.assertIsNone(impact_from_ball(swing,[0,0],38,52,(48,56),.5))


if __name__=='__main__':
    unittest.main()
