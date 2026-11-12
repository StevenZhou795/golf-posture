import unittest
from types import SimpleNamespace

import numpy as np

from golf_workflows.swing_phases import _club_reversal,_club_rest_end,detect_refined_phases,find_checkpoint_frames
from golf_workflows.golf_pose_3d import contact_frame,impact_from_ball


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
    def test_shaft_track_is_used_when_hosel_observations_are_missing(self):
        points, valid = swing_fixture()
        valid[:, 18] = False
        raw = points.copy()
        raw[:, 18] = np.nan
        phases, details = detect_refined_phases(raw, valid, points, 30)
        self.assertEqual(details['club_track_source'], 'shaft')
        self.assertEqual(details['reliable_club_pairs'], 0)
        self.assertTrue(34 <= phases.top <= 42)
        self.assertTrue(48 <= phases.impact <= 56)
        self.assertFalse(valid[:, 18].any())

    def test_clip_starting_at_the_top_is_flagged(self):
        # 009: recording starts at the top, so the hands only fall, then rise to the finish.
        points,valid=swing_fixture()
        points,valid=points[38:],valid[38:]
        _,details=detect_refined_phases(points,valid,points,30)
        self.assertEqual(details['clip_starts_after_top']['highest_hands_frame'],0)
        _,details=detect_refined_phases(*swing_fixture()[:2],swing_fixture()[0],30)
        self.assertNotIn('clip_starts_after_top',details)

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
        self.assertEqual([improved[key] for key in ['p7','p8','p9','p10']],[20,21,22,24])

    def test_p8_extended_arms_p9_hands_reach_shoulder_p10_before_lowering(self):
        poses=np.zeros((40,19,3))
        poses[:,10]=[0,-.5,0]
        poses[:,11]=[.2,0,0]
        poses[:,14]=[-.2,0,0]
        def arms(frame,elevation,bend):
            direction=np.array([np.cos(np.radians(elevation)),-np.sin(np.radians(elevation)),0])
            for shoulder,elbow,wrist in ((11,12,13),(14,15,16)):
                poses[frame,elbow]=poses[frame,shoulder]+.3*direction+[0,0,bend]
                poses[frame,wrist]=poses[frame,shoulder]+.6*direction
        # Straight arms at and just after impact, before the hands have risen.
        for frame,elevation,bend in [(20,-80,0),(21,-70,0),(22,-50,.1),(23,-35,.02),
                                     (24,-12,.15),(25,8,.15),(26,40,.15),(27,55,.15)]:
            arms(frame,elevation,bend)
        for frame in range(28,31):
            arms(frame,70,.15)
        # Hands come down towards the head after the finish.
        for frame in range(31,36):
            arms(frame,70,.15)
            poses[frame,13]=[.1,-.3,0]
        poses[36:]=np.nan
        positions=find_checkpoint_frames(poses,0,10,20,reserve_positions=True)
        self.assertEqual(positions['p8'],23)
        self.assertEqual(positions['p9'],25)
        self.assertIn(positions['p10'],range(28,31))

    def test_top_follows_club_reversal_after_wrist_peak(self):
        points=np.full((40,19,2),np.nan)
        theta=np.interp(np.arange(40),[0,10,22,30,39],[0,60,150,40,0])
        points[:,17]=[0,0]
        points[:,18]=200*np.column_stack([np.cos(np.radians(theta)),np.sin(np.radians(theta))])
        reliable=np.ones(40,bool)
        self.assertEqual(_club_reversal(points,reliable,20,30,30,100),22)
        reliable[19:26]=False
        self.assertIsNone(_club_reversal(points,reliable,20,30,30,100))

    def test_address_is_last_club_rest_and_ignores_jumps(self):
        club=np.zeros((40,2))
        club[30:]=np.linspace(.5,3,10)[:,None]*[1,0]
        club[22:25]=np.linspace(.05,.25,3)[:,None]*[1,0]
        club[25:30]=np.linspace(.35,.5,5)[:,None]*[1,0]
        # A one-frame misdetection would otherwise inflate the rest radius.
        club[10]=[10,0]
        reliable=np.ones(40,bool)
        self.assertEqual(_club_rest_end(club,reliable,np.zeros(2),40,30),24)

    def test_unobserved_shaft_falls_back_to_lead_arm_rotation(self):
        poses=np.zeros((40,19,3))
        poses[:,11]=[0,0,0]
        angles=np.interp(np.arange(40),[0,10,20,30,39],[-80,0,60,0,-80])
        poses[:,13]=np.column_stack([np.cos(np.radians(angles)),-np.sin(np.radians(angles)),np.zeros(40)])
        poses[:,10]=[0,-.5,0]
        poses[:,17]=[0,0,0]
        poses[:,18]=[0,1,0]
        poses[2,18]=[1,0,0]
        poses[6,18]=[1,.1,0]
        observed=np.ones(40,bool)
        sources={}
        positions=find_checkpoint_frames(poses,0,20,39,club_observed=observed,sources=sources)
        self.assertEqual((positions['p2'],sources['p2']),(2,'observed_shaft'))
        observed[2]=False
        positions=find_checkpoint_frames(poses,0,20,39,club_observed=observed,sources=sources)
        self.assertEqual((positions['p2'],sources['p2']),(6,'observed_shaft'))
        observed[6]=False
        positions=find_checkpoint_frames(poses,0,20,39,club_observed=observed,sources=sources)
        self.assertEqual((positions['p2'],sources['p2']),(3,'lead_arm_progress'))

    def test_finish_window_stays_open_until_raised_hands_come_down(self):
        n=120
        points=np.zeros((n,19,2),float)
        points[:,8]=[0,-100]
        # A pause with raised hands (62-72) precedes the held finish (80-95).
        height=np.interp(np.arange(n),[0,15,38,52,62,72,80,95,100,119],[0,0,1.5,0,1.3,1.3,1.4,1.4,0,0])
        points[:,13,1]=points[:,16,1]=-100*height
        points[:,17]=(points[:,13]+points[:,16])/2
        points[:,18]=points[:,17]+[0,150]
        valid=np.ones((n,19),bool)
        phases,details=detect_refined_phases(points,valid,points,30)
        self.assertEqual(details['finish_reason'],'raised_hands_lowered')
        self.assertLess(details['finish_hold_start'],75)
        self.assertTrue(92<=phases.follow_through<=100)

    def test_p5_is_lead_arm_parallel_even_when_shaft_is_parallel_at_top(self):
        poses=np.zeros((30,19,3))
        poses[:,10]=[0,-.3,0]
        poses[:,13]=[.5,-.5,0]
        poses[15,13]=[.7,-.05,0]
        poses[16:,13]=[.1,.6,0]
        poses[:,18]=[0,1,0]
        poses[10:14,18]=[1,0,0]
        poses[17,18]=[1,.05,0]
        positions=find_checkpoint_frames(poses,0,10,20,reserve_positions=True)
        self.assertEqual(positions['p5'],15)
        self.assertEqual(positions['p6'],17)

    def test_p9_is_postimpact_arm_parallel_and_p10_has_raised_hands(self):
        poses=np.zeros((40,19,3))
        poses[:,10]=[0,-.3,0]
        poses[:,11]=[.2,0,0]
        poses[:,13]=[.3,.8,0]
        poses[22,13]=[1.2,-.2,0]
        poses[23:35,13]=[.6,-.5,0]
        # Hands above and slightly behind the head: the lead-side dot product is negative.
        poses[27,13]=[-.05,-.45,0]
        poses[35,13]=[1.2,0,0]
        poses[36:,13]=[.3,.8,0]
        poses[:,18]=[0,1,0]
        positions=find_checkpoint_frames(poses,0,10,20,reserve_positions=True)
        self.assertEqual(positions['p9'],22)
        self.assertEqual(positions['p10'],27)

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
        close,details=detect_refined_phases(points,valid,points,30,impact_hint=53)
        self.assertEqual(close.impact,53)
        self.assertEqual(details['impact_source'],'ball')

    def test_contact_may_trail_hands_low_point_but_not_lead_it_far(self):
        points,valid=swing_fixture()
        _,details=detect_refined_phases(points,valid,points,30)
        wrist=details['wrist_impact']
        late,late_details=detect_refined_phases(points,valid,points,30,impact_hint=wrist+8,impact_hint_source='ball_departure')
        self.assertEqual(late.impact,wrist+8)
        self.assertEqual(late_details['impact_source'],'ball_departure')
        early,_=detect_refined_phases(points,valid,points,30,impact_hint=wrist-6)
        self.assertNotEqual(early.impact,wrist-6)

    def test_ball_departure_takes_priority_over_nearest_clubhead(self):
        points,valid=swing_fixture()
        points[:,18]=[100,100]
        points[50,18]=[1,0]
        points[56,18]=[2,0]
        swing=SimpleNamespace(points=points,valid=valid,frames=len(points))
        presence={f:f<56 for f in range(46,62)}
        frame,source,_=contact_frame(swing,[0,0],(49,61),30,presence=presence)
        self.assertEqual((frame,source),(56,'ball_departure'))
        frame,source,_=contact_frame(swing,[0,0],(49,61),30)
        self.assertEqual((frame,source),(50,'clubhead_at_ball'))
        points[56,18]=[400,0]
        frame,source,details=contact_frame(swing,[0,0],(49,61),30,presence=presence)
        self.assertEqual((frame,source),(55,'ball_departure'))
        self.assertTrue(details['clubhead_unconfirmed'])

    def test_contact_between_frames_is_reported(self):
        # w03: clubhead 212 px short of the ball on the last at-rest frame, gone past it on the next.
        points,valid=swing_fixture()
        points[:,18]=[1000,0]
        points[55,18]=[212,0]; points[56,18]=[118,0]
        valid[54,18]=False
        swing=SimpleNamespace(points=points,valid=valid,frames=len(points))
        presence={f:f<56 for f in range(46,62)}
        frame,source,details=contact_frame(swing,[0,0],(49,61),82,presence=presence)
        self.assertEqual((frame,source),(55,'ball_departure'))
        self.assertEqual(details['contact_between'],[55,56])
        self.assertFalse(details['contact_captured'])
        points[56,18]=[40,0]
        details=contact_frame(swing,[0,0],(49,61),82,presence=presence)[2]
        self.assertTrue(details['contact_captured'])

    def test_clubhead_motion_measures_both_contact_frames(self):
        # 008: clubhead short of the ball while it is at rest, then a streak over the ball spot.
        # w03: on the ball-gone frame the clubhead has already left the ground band.
        import tempfile
        from pathlib import Path
        import cv2
        from golf_workflows.golf_pose_3d import clubhead_motion
        scale,spot=100,(320,240)

        def frame(clubhead_x=None,ball=True):
            image=np.full((480,640,3),90,np.uint8)
            cv2.rectangle(image,(40,300),(120,330),(200,200,200),-1)  # a foot, unchanged from the top
            if ball: cv2.circle(image,spot,6,(255,255,255),-1)
            if clubhead_x is not None: cv2.rectangle(image,(clubhead_x-20,230),(clubhead_x+20,250),(10,10,10),-1)
            return image

        with tempfile.TemporaryDirectory() as folder:
            for name,frames,expected in (('008',[frame(),frame(240),frame(345,ball=False)],{'1':0.6,'2':0.05}),
                                         ('w03',[frame(),frame(240),frame(ball=False)],{'1':0.6,'2':None})):
                path=Path(folder)/f'{name}.avi'
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30,(640,480))
                for image in frames: writer.write(image)
                writer.release()
                motion=clubhead_motion(path,[1,2],0,list(spot),scale)
                for key,value in expected.items():
                    if value is None: self.assertIsNone(motion[key],name)
                    else: self.assertLess(abs(motion[key]-value),0.1,name)

    def test_blurred_impact_uses_clubhead_approach(self):
        # w01: clubhead 306, 204, 92 px from the ball, then undetectable as the ball leaves.
        points,valid=swing_fixture()
        points[:,18]=[1000,0]
        points[53,18]=[306,0]; points[54,18]=[204,0]; points[55,18]=[92,0]; points[56,18]=[0,230]
        swing=SimpleNamespace(points=points,valid=valid,frames=len(points))
        presence={f:f<56 for f in range(46,62)}
        frame,source,details=contact_frame(swing,[0,0],(49,61),81,presence=presence)
        self.assertEqual((frame,source),(56,'ball_departure'))
        self.assertAlmostEqual(details['clubhead_reaches_ball_at'],55.82,places=2)
        points[54,18]=[600,0]
        self.assertEqual(contact_frame(swing,[0,0],(49,61),81,presence=presence)[0],55)

    def test_tracked_clubhead_used_when_detection_is_blurred(self):
        points,valid=swing_fixture()
        points[:,18]=[100,100]
        valid[:,18]=False
        tracked=np.full((len(points),2),100.)
        tracked[57]=[3,0]
        swing=SimpleNamespace(points=points,valid=valid,frames=len(points))
        frame,source,_=contact_frame(swing,[0,0],(49,61),30,tracked=tracked,tracked_penalty=5)
        self.assertEqual((frame,source),(57,'tracked_clubhead_at_ball'))
        self.assertEqual(contact_frame(swing,[0,0],(49,61),5,tracked=tracked,tracked_penalty=5)[0],None)

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
