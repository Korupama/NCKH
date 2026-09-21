import numpy as np
from stage4_metric3d.initializers.cache import InitializerCache, write_jsonl


def test_initializer_cache_roundtrip(tmp_path):
    raw = np.zeros((133,3), dtype=float)
    raw[:,2] = np.arange(133)
    p = tmp_path / "cache.jsonl"
    write_jsonl(p, {"schema_version":"x"}, [{
        "track_id":"t1", "frame_index":5, "backend":"test",
        "raw_keypoints_133":raw.tolist(), "keypoint_scores_133":[1.0]*133
    }])
    cache = InitializerCache.load_jsonl(p)
    rec = cache.get("t1",5)
    assert rec is not None
    assert rec.raw_keypoints_133.shape == (133,3)
    assert abs(np.nanmean(rec.relative_depth_133[[11,12]])) < 1e-12
