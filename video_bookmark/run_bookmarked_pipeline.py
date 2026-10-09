"""Run the existing Stage 1-9 pipeline freshly for bookmarked video frames.

From D:\\NCKH:
  .\\.venv\\Scripts\\python.exe .\\video_bookmark\\run_bookmarked_pipeline.py
  .\\.venv\\Scripts\\python.exe .\\video_bookmark\\run_bookmarked_pipeline.py --dry-run

Frame numbers are zero-based, as in the web demo. Decode in ascending order
from frame 0; no time-based seeking, old inference reuse, or model replacement.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
STAGE9 = ROOT / 'stage9_offside_position_v0.1.0'
PROGRESS_PREFIX = '__STAGE_PROGRESS__'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_bookmarks(path):
    frames = []
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        value = line.split('#', 1)[0].strip()
        if not value:
            continue
        if not value.isascii() or not value.isdecimal():
            raise ValueError(f'{path.name}:{number}: expected a non-negative integer frame, got {value!r}')
        frames.append(int(value))
    if not frames:
        raise ValueError(f'No frame numbers found in {path}')
    return sorted(set(frames)), len(frames) - len(set(frames))


def save_summary(directory, report):
    report['updated_at'] = utc_now()
    temporary = directory / 'summary.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(directory / 'summary.json')
    fields = ['frame_index', 'timestamp_sec', 'status', 'last_stage', 'stage9_status',
              'elapsed_sec', 'output_directory', 'log', 'error']
    with (directory / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['frames'])


def stop_child(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        # Only the process tree created for this frame; never kill by name.
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_pipeline(python, image, directory, frame, row, persist):
    command = [str(python), '-u', str(STAGE9 / 'live_orchestrator.py'),
               '--image', str(image), '--frame', str(frame), '--outdir', str(directory)]
    row['command'] = command
    log_path = directory / 'orchestrator.log'
    row['log'] = str(log_path)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt'
               else {'start_new_session': True})
    tail = deque(maxlen=12)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                   errors='replace', bufsize=1, **options)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.strip():
                    tail.append(line.strip())
                if line.startswith(PROGRESS_PREFIX):
                    try:
                        event = json.loads(line[len(PROGRESS_PREFIX):])
                    except json.JSONDecodeError:
                        continue
                    stage, status = event.get('stage'), event.get('status')
                    row['last_stage'] = stage
                    row['stages'][str(stage)] = status
                    print(f'  S{stage}: {status}', flush=True)
                    persist()
            code = process.wait()
        except BaseException:
            stop_child(process)
            raise
        finally:
            process.stdout.close()
    row['returncode'] = code
    if code != 0:
        raise RuntimeError(f'Pipeline exit code {code}. See {log_path}\n' + '\n'.join(tail))
    # Completed execution is distinct from a resolved offside result.
    for stage in range(1, 10):
        path = directory / f'stage{stage}.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        actual_frame = payload.get('frame_index', payload.get('selected_frame',
                                   payload.get('replay_context', {}).get('selected_frame')))
        if actual_frame != frame:
            raise ValueError(f'{path.name}: frame mismatch ({actual_frame} != {frame})')
    row['stage9_status'] = payload.get('status', 'UNKNOWN')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', type=Path, default=ROOT / 'video_bookmark/vie_kor.mp4')
    parser.add_argument('--frames', type=Path, default=ROOT / 'video_bookmark/vie_kor.txt')
    parser.add_argument('--output-root', type=Path, default=STAGE9 / 'outputs/bookmark_runs')
    parser.add_argument('--python', type=Path, default=ROOT / '.venv/Scripts/python.exe',
                        help='Python interpreter for the existing pipeline')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and list frames; no inference/output files')
    parser.add_argument('--stop-on-error', action='store_true', help='Stop after the first failed frame')
    args = parser.parse_args(argv)
    cap = None
    try:
        import cv2
        video, bookmarks, python = args.video.resolve(), args.frames.resolve(), args.python.resolve()
        for path in (video, bookmarks, python, STAGE9 / 'live_orchestrator.py'):
            if not path.is_file():
                raise FileNotFoundError(path)
        frames, duplicates = load_bookmarks(bookmarks)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise ValueError('Invalid video FPS/frame count')
        count = int(count)
        invalid = [f for f in frames if f >= count]
        if invalid:
            raise ValueError(f'Frames outside video (0..{count-1}): {invalid}')
        print(f'Video: {video}\nFPS: {fps:g}; frames: {count}\n'
              f'Bookmarks: {len(frames)} (zero-based); duplicates skipped: {duplicates}', flush=True)
        print('Frames: ' + ', '.join(map(str, frames)), flush=True)
        if args.dry_run:
            print('Dry run OK. No inference started.')
            return 0

        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = args.output_root.resolve() / f'{video.stem}_{stamp}_{uuid.uuid4().hex[:8]}'
        output.mkdir(parents=True, exist_ok=False)
        report = dict(schema_version='bookmarked-pipeline-batch-1.0', status='running',
                      created_at=utc_now(), video=str(video), bookmark_file=str(bookmarks),
                      bookmark_sha256=hashlib.sha256(bookmarks.read_bytes()).hexdigest(),
                      video_size_bytes=video.stat().st_size, fps=fps, frame_count=count,
                      frame_index_basis='ZERO_BASED_SEQUENTIAL_DECODE', reuse_previous_inference=False,
                      frames=[dict(frame_index=f, timestamp_sec=f/fps, status='pending', stages={},
                                   output_directory=str(output / f'frame_{f:08d}')) for f in frames])
        persist = lambda: save_summary(output, report)
        persist()
        print(f'Output: {output}', flush=True)
        next_frame = 0
        interrupted = False
        for number, row in enumerate(report['frames'], 1):
            start = time.monotonic()
            frame = row['frame_index']
            directory = Path(row['output_directory'])
            try:
                directory.mkdir(exist_ok=False)
                row.update(status='extracting', started_at=utc_now())
                persist()
                print(f'[{number}/{len(frames)}] Extract frame {frame} (sequential decode)...', flush=True)
                # Count decoded frames, avoiding keyframe/time-seek ambiguity.
                while next_frame <= frame:
                    if not cap.grab():
                        raise RuntimeError(f'Video decoding stopped at frame {next_frame}')
                    next_frame += 1
                ok, pixels = cap.retrieve()
                if not ok or pixels is None:
                    raise RuntimeError(f'Cannot retrieve frame {frame}')
                image = directory / f'frame_{frame}.png'
                if not cv2.imwrite(str(image), pixels):
                    raise OSError(f'Cannot write {image}')
                selection = dict(source_video=str(video), source_kind='BOOKMARK_SEQUENTIAL_DECODE',
                                 frame_index=frame, timestamp_sec=frame/fps, fps=fps,
                                 pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
                (directory / 'frame_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
                row['status'] = 'running'
                persist()
                run_pipeline(python, image, directory, frame, row, persist)
                row['status'] = 'completed'
                print(f'  Completed. Stage 9: {row["stage9_status"]}', flush=True)
            except KeyboardInterrupt:
                row.update(status='interrupted', error='Stopped by user')
                interrupted = True
            except Exception as exc:
                row.update(status='failed', error=str(exc))
                print(f'  FAILED: {str(exc).splitlines()[0]}', flush=True)
            finally:
                row.update(finished_at=utc_now(), elapsed_sec=round(time.monotonic() - start, 2))
                persist()
            if interrupted or (args.stop_on_error and row['status'] == 'failed'):
                break
        failures = sum(r['status'] == 'failed' for r in report['frames'])
        completed = sum(r['status'] == 'completed' for r in report['frames'])
        report.update(status='interrupted' if interrupted else 'completed_with_errors' if failures else 'completed',
                      finished_at=utc_now(), completed_frames=completed, failed_frames=failures)
        persist()
        print(f'Finished: {completed}/{len(frames)} completed; {failures} failed.\nSummary: {output / "summary.csv"}', flush=True)
        return 130 if interrupted else 1 if failures else 0
    except (OSError, ValueError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            cap.release()


if __name__ == '__main__':
    raise SystemExit(main())
