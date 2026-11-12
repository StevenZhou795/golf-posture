"""Explicit single-video FP32 benchmark; never stop or resume other processes."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from rtmw_workflow.report import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not args.video.is_file():
        parser.error('Video does not exist')
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Benchmark output already exists; use a fresh folder')
    report = dict(video=str(args.video.resolve()), precision='float32', pose_device='cuda', club_device='cuda',
                  execution='staged', club_backend='swingtrace-yolo11m')
    start = time.perf_counter()
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        from rtmw_workflow.cli import main as workflow
        report['exit_code'] = workflow(['video', str(args.video.resolve()),
            '--pose-device', 'cuda', '--club-device', 'cuda', '--cpu-threads', '6',
            '--output-dir', str(output)])
        report['peak_cuda_allocated_mb'] = torch.cuda.max_memory_allocated()/1024**2
    except BaseException as exc:
        report.update(error=repr(exc), exit_code=1)
        raise
    finally:
        report['wall_seconds_including_initialization'] = time.perf_counter()-start
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'benchmark.json', report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
