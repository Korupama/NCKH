import numpy as np

from stage4_metric3d.stage3_adapter import PoseObservation2D, Stage3State, Stage3Track
from stage4_metric3d.stage3_bridge_qa import audit_stage3_bridge


def _observation(track_id, pose_status, bbox, valid=True):
    uv = np.full((23, 2), np.nan)
    if valid:
        uv[:] = 1.0
    return PoseObservation2D(
        track_id=track_id,
        frame_index=86,
        bbox_xyxy=np.asarray(bbox, dtype=float),
        uv23=uv,
        states23=tuple("VALID" if valid else "MISSING" for _ in range(23)),
        state_weights23=np.ones(23) if valid else np.zeros(23),
        raw_scores23=np.ones(23),
        pose_status=pose_status,
        source={},
    )


def test_bridge_audit_reports_missing_edge_and_crowding(tmp_path):
    first = _observation("track_001", "MISSING", [0, 100, 40, 200], valid=False)
    second = _observation("track_002", "VALID", [20, 120, 70, 220])
    state = Stage3State(
        path=tmp_path / "state.json",
        raw={"coordinate_space": "RAW_DISTORTED_PIXEL"},
        replay_context={"image_width": 1920, "image_height": 1080},
        selected_frame=86,
        tracks=[
            Stage3Track("track_001", None, None, [first], {}),
            Stage3Track("track_002", None, None, [second], {}),
        ],
    )

    report = audit_stage3_bridge(state)

    assert report["summary"]["raw_pose_missing_tracks"] == ["track_001"]
    assert report["summary"]["edge_risk_tracks"] == ["track_001"]
    assert report["summary"]["crowded_pairs"] == 1
    assert report["tracks"][0]["missing_core_anchor_names"]
