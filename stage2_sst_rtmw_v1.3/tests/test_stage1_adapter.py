from __future__ import annotations

import os
from pathlib import Path

import pytest

from stage2_entities.stage1_adapter import replay_context_from_stage1_workspace


def test_supplied_stage1_workspace_if_available():
    root = Path(os.environ.get("STAGE1_TEST_ROOT", "/mnt/data/_stage1_review/stage_1_camera_v12"))
    if not root.is_dir() or not (root / "stage_1_camera").is_dir():
        pytest.skip("Supplied Stage-1 workspace is not fully mounted in this runtime")
    ctx = replay_context_from_stage1_workspace(root)
    assert ctx.selected_frame == 86
    assert (ctx.window_start, ctx.window_end) == (56, 116)
    assert (ctx.shot_start, ctx.shot_end) == (0, 212)
    assert ctx.frame_count == 213
    assert ctx.fps == pytest.approx(30.0)
    assert (ctx.image_width, ctx.image_height) == (1920, 1080)
    assert ctx.coordinate_space == "RAW_DISTORTED_PIXEL"
    assert ctx.extra["source_pixel_space"] == "original_raw"
