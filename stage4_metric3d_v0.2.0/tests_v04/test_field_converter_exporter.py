import json
import numpy as np

from stage4_metric3d.backends.field_converter.exporter import export_field_converter_raw, build_boxes_from_stage3
from stage4_metric3d.backends.field_converter.sam3d_cache import save_sam3d_cache
from stage4_metric3d.stage3_adapter import load_stage3_state
from .helpers import build_synthetic_stage3_and_cameras


def test_export_matches_official_inference_layout(tmp_path):
    stage3, camera_dir, frames, tids = build_synthetic_stage3_and_cameras(tmp_path / "case")
    s3 = load_stage3_state(stage3)
    boxes = build_boxes_from_stage3(s3, frames, tids)
    T, N = len(frames), len(tids)
    s2d = np.zeros((T, N, 25, 2), np.float32)
    s3d = np.zeros((T, N, 25, 3), np.float32)
    cache = save_sam3d_cache(
        tmp_path / "cache.npz", frame_indices=frames, track_ids=tids,
        boxes_xyxy=boxes, skel_2d_px=s2d, skel_3d_relative_m=s3d,
        valid_mask=np.ones((T, N), bool),
    )
    out = tmp_path / "fc_input"
    manifest = export_field_converter_raw(
        stage3_state=stage3, camera_dir=camera_dir, sam3d_cache=cache,
        output_root=out, sequence_name="seq",
    )
    assert np.load(out / "boxes/seq.npy").shape == (T, N, 4)
    assert np.load(out / "skel_2d/seq.npy").shape == (T, N, 25, 2)
    assert np.load(out / "skel_3d_relative/seq.npy").shape == (T, N, 25, 3)
    with np.load(out / "cameras/seq.npz") as cam:
        assert cam["K"].shape == (T, 3, 3)
        assert cam["R"].shape == (T, 3, 3)
        assert cam["t"].shape == (T, 3)
        assert cam["k"].shape == (T, 2)
    assert manifest["track_ids"] == tids
    markers = sorted((out / "frames/seq").iterdir())
    assert len(markers) == T
