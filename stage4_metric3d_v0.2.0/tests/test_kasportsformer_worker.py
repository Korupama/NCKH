from pathlib import Path

import numpy as np

from stage4_metric3d.initializers.kasportsformer import H36M17_TO_WHOLEBODY133
from stage4_metric3d.initializers.kasportsformer_worker import (
    build_model_inputs,
    coco17_to_h36m17,
    interpolate_track_coco17,
)
from stage4_metric3d.stage3_adapter import (
    PoseObservation2D,
    Stage3State,
    Stage3Track,
)


def _observation(frame: int, offset: float) -> PoseObservation2D:
    uv = np.zeros((23, 2), dtype=float)
    uv[:17, 0] = np.arange(17, dtype=float) + offset
    uv[:17, 1] = np.arange(17, dtype=float) * 2.0 + offset
    return PoseObservation2D(
        track_id="track_001",
        frame_index=frame,
        bbox_xyxy=np.asarray([100.0, 100.0, 300.0, 500.0]),
        uv23=uv,
        states23=tuple(["VALID"] * 23),
        state_weights23=np.ones(23),
        raw_scores23=np.full(23, 0.8),
        pose_status="VALID",
        source={},
    )


def _track() -> Stage3Track:
    return Stage3Track(
        track_id="track_001",
        role="player",
        identity_confidence=1.0,
        observations=[_observation(85, 0.0), _observation(87, 2.0)],
        raw={},
    )


def test_coco17_to_h36m17_preserves_named_limb_mapping():
    coco = np.arange(17 * 2, dtype=np.float32).reshape(1, 17, 2)
    scores = np.linspace(0.1, 0.9, 17, dtype=np.float32).reshape(1, 17)
    h36m, h36m_scores = coco17_to_h36m17(coco, scores)
    assert np.array_equal(h36m[0, 11], coco[0, 5])
    assert np.array_equal(h36m[0, 16], coco[0, 10])
    assert np.array_equal(h36m[0, 3], coco[0, 16])
    assert np.isclose(h36m_scores[0, 0], np.mean(scores[0, [11, 12]]))
    assert H36M17_TO_WHOLEBODY133[0] is None
    assert H36M17_TO_WHOLEBODY133[11] == 5


def test_interpolation_marks_filled_frame_with_low_confidence():
    xy, confidence = interpolate_track_coco17(_track(), [85, 86, 87])
    assert np.isclose(xy[1, 0, 0], 1.0)
    assert np.isclose(confidence[0, 0], 0.8)
    assert np.isclose(confidence[1, 0], 0.3)
    assert np.isclose(confidence[2, 0], 0.8)


def test_model_inputs_are_centered_windows_and_keep_stage4_cohort():
    state = Stage3State(
        path=Path("stage3.json"),
        raw={},
        replay_context={"image_width": 1920, "image_height": 1080},
        selected_frame=86,
        tracks=[_track()],
    )
    inputs, keys = build_model_inputs(
        state,
        window_length=27,
        window_radius_frames=13,
        selected_only=False,
    )
    assert inputs.shape == (2, 27, 17, 3)
    assert keys == [("track_001", 85), ("track_001", 87)]
    assert np.isfinite(inputs).all()
    assert np.all((inputs[..., 2] >= 0.0) & (inputs[..., 2] <= 1.0))
