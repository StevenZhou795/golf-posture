"""No video reads: regressions, RTMW 2D coordinate restoration, SwingTrace smoke test."""
import unittest

import numpy as np


def run(args):
    import torch
    from rtmw_workflow.runtime import HERE, ROOT
    from rtmw_workflow.report import write_json
    from rtmw_workflow.swingtrace import SwingTraceClub
    from rtmw_workflow.runtime import load_model
    from rtmw_workflow.official import inference
    from rtmw_workflow.backend import extract_prediction
    torch.set_num_threads(args.cpu_threads)
    suite = unittest.TestLoader().discover(str(HERE/'tests'), pattern='test_*.py')
    for pattern in ('test_position_selection.py', 'test_refined_phases.py', 'test_club_tracking.py',
                    'test_skeleton_check.py', 'test_club_refinement.py', 'test_detector_pose.py'):
        suite.addTests(unittest.TestLoader().discover(str(ROOT/'tests'), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(unit_tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  video_inference_run=False, input='synthetic zero image 640x640', passed=False)
    if result.wasSuccessful():
        try:
            body = load_model(args.checkpoint, args.cpu_threads).to(args.pose_device)
            blank = np.zeros((640, 640, 3), np.uint8)
            # Identical blank crops at two translated boxes must produce equally
            # translated image pixels after the official inverse affine mapping.
            samples = inference(body, blank, bboxes=np.array([[50, 60, 242, 316], [150, 160, 342, 416]], np.float32))
            first, first_scores = extract_prediction(samples[0].pred_instances)
            second, second_scores = extract_prediction(samples[1].pred_instances)
            assert np.isfinite(first).all() and np.isfinite(second).all()
            assert np.isfinite(first_scores).all() and np.isfinite(second_scores).all()
            np.testing.assert_allclose(second-first, 100., rtol=0, atol=.001)
            report.update(body_model=body.workflow_metadata,
                          body_coordinate_translation_max_error_px=float(np.max(np.abs(second-first-100.))))
            body.to('cpu')
            del body
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            club = SwingTraceClub(args.club_checkpoint, args.club_device)
            _, confidence, row = club.observe(np.zeros((640, 640, 3), np.uint8), None, np.full(4, np.nan), 0)
            assert np.isfinite(confidence).all()
            assert not row['pair_usable'], 'No golfer anchor must not produce associated club geometry'
            assert all(p.dtype == torch.float32 for p in club.model.model.parameters())
            report.update(passed=True, model=club.metadata, synthetic_inference_seconds=row['inference_seconds'])
        except Exception as exc:
            report['model_smoke_error'] = repr(exc)
    output = HERE/'outputs/self_test_swingtrace.json'
    output.parent.mkdir(exist_ok=True)
    write_json(output, report)
    write_json(HERE/'outputs/self_test_rtmw2d.json', report)
    print(f'Self-test report: {output}')
    return 0 if report['passed'] else 1
