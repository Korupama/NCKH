import numpy as np
import pytest

from stage4_metric3d.backends.field_converter.sam3d_cache import save_sam3d_cache, Sam3DCache


def test_cache_roundtrip_and_strict_shapes(tmp_path):
    T, N, J = 3, 2, 25
    path = save_sam3d_cache(
        tmp_path / "cache.npz",
        frame_indices=[10, 11, 12],
        track_ids=["a", "b"],
        boxes_xyxy=np.ones((T, N, 4), np.float32),
        skel_2d_px=np.ones((T, N, J, 2), np.float32),
        skel_3d_relative_m=np.ones((T, N, J, 3), np.float32),
        valid_mask=np.ones((T, N), bool),
    )
    cache = Sam3DCache.load(path)
    assert cache.T == T and cache.N == N
    assert not cache.semantic_mapping_validated


def test_cache_rejects_non_25_joint_layout(tmp_path):
    with pytest.raises(ValueError):
        save_sam3d_cache(
            tmp_path / "bad.npz",
            frame_indices=[10], track_ids=["a"],
            boxes_xyxy=np.ones((1, 1, 4)),
            skel_2d_px=np.ones((1, 1, 24, 2)),
            skel_3d_relative_m=np.ones((1, 1, 24, 3)),
            valid_mask=np.ones((1, 1), bool),
        )
