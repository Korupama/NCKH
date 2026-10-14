import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('bookmark_runner', Path(__file__).resolve().parents[1] / 'video_bookmark/run_bookmarked_pipeline.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def probe(monkeypatch, stream):
    monkeypatch.setattr(runner.shutil, 'which', lambda _: 'ffprobe')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=json.dumps({'streams': [stream]}), stderr=''))


def test_audio_length_cannot_validate_nonexistent_video_frames(monkeypatch):
    probe(monkeypatch, {'duration': '1605.4'})
    with pytest.raises(ValueError, match='video-stream duration'):
        runner.validate_video_stream('mixed.mp4', [63788, 88859, 153167], 50)
    assert runner.validate_video_stream('mixed.mp4', [63788], 50)['duration'] == '1605.4'


def test_explicit_video_frame_count(monkeypatch):
    probe(monkeypatch, {'nb_frames': '80000', 'duration': '1600'})
    runner.validate_video_stream('trimmed.mp4', [0, 79999], 50)
    with pytest.raises(ValueError, match='video-stream frame count'):
        runner.validate_video_stream('trimmed.mp4', [80000], 50)


def test_decode_failure_blocks_later_bookmarks_and_preserves_completed():
    rows = [dict(frame_index=63788, status='completed'),
            dict(frame_index=88859, status='extracting'),
            dict(frame_index=90574, status='pending')]
    report = {'frames': rows}
    runner.block_after_decode_error(report, rows[1], 80073, runner.VideoDecodeError('Bad video'))
    assert [r['status'] for r in rows] == ['completed', 'failed', 'blocked']
    assert report['decode_failure']['next_expected_frame'] == 80073
    assert report['decode_failure']['requested_frame'] == 88859


def test_missing_probe_explicitly_reports_skipped_validation(monkeypatch, capsys):
    monkeypatch.setattr(runner.shutil, 'which', lambda _: None)
    assert runner.validate_video_stream('video.mp4', [10], 50) is None
    assert 'validation skipped' in capsys.readouterr().out
