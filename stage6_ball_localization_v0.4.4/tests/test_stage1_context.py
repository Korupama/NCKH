import os
from pathlib import Path
import pytest
from ball_localization.stage1_context import load_replay_context_from_stage1

def test_stage1_context_real_workspace():
    root=Path(os.environ.get("STAGE1_ROOT","/mnt/data/_stage1/stage_1_camera_v12"))
    manifest = root / "outputs/stage1_final/camera_timeline_target_window_manifest.json"
    if not root.is_dir() or not manifest.is_file(): pytest.skip("Set STAGE1_ROOT to a complete Stage-1 v12 workspace to run integration test")
    c=load_replay_context_from_stage1(root); assert c["selected_frame"]==86 and c["window_start"]==56 and c["window_end"]==116 and c["coordinate_space"]=="RAW_DISTORTED_PIXEL"
