from __future__ import annotations

import numpy as np
import pytest
import json

from stage5_team_affiliation.config import Stage5Config
from stage5_team_affiliation.residual_config import ResidualConfig
from stage5_team_affiliation.residual_pipeline import (
    combine_region_features,
    extract_human_multiregion_features,
)
from stage5_team_affiliation.residual_calibration import _load_rows


def _observations():
    return {
        "human": [
            {
                "frame_index": frame,
                "source_bbox_xyxy": [0, 0, 40, 100],
                "keypoints_133": [],
            }
            for frame in range(6)
        ]
    }


def _two_colour_image():
    image = np.zeros((100, 40, 3), dtype=np.uint8)
    image[:50] = (20, 20, 220)
    image[50:] = (220, 20, 20)
    return image


def test_weighted_composite_preserves_region_distance_weights():
    torso = np.asarray([1.0, 0.0], dtype=np.float32)
    lower = np.asarray([0.0, 1.0], dtype=np.float32)
    composite = combine_region_features(torso, lower, 0.7, 0.3)
    assert composite.shape == (4,)
    assert np.linalg.norm(composite) == pytest.approx(1.0)
    assert float(np.dot(composite[:2], composite[:2])) == pytest.approx(0.7)
    assert float(np.dot(composite[2:], composite[2:])) == pytest.approx(0.3)


def test_multiregion_extractor_uses_independent_torso_and_lower_evidence():
    features, diagnostics, config = extract_human_multiregion_features(
        _observations(), lambda _: _two_colour_image(), Stage5Config(),
        torso_weight=0.7, lower_weight=0.3, require_lower=True)
    assert features["human"].shape == (96,)
    assert diagnostics["human"]["valid_torso_frames"] == 3
    assert diagnostics["human"]["valid_lower_frames"] == 3
    assert diagnostics["human"]["descriptor_status"] == "MULTIREGION_VALID"
    assert config["descriptor"] == "TORSO_LOWER_WEIGHTED_CONCAT"


def test_multiregion_fail_closed_when_lower_evidence_is_unavailable():
    appearance = Stage5Config(allow_bbox_lower_body_fallback=False)
    features, diagnostics, _ = extract_human_multiregion_features(
        _observations(), lambda _: _two_colour_image(), appearance,
        torso_weight=0.7, lower_weight=0.3, require_lower=True)
    assert "human" not in features
    assert diagnostics["human"]["descriptor_status"] == "INSUFFICIENT_LOWER"


def test_multiregion_config_validates_weights_and_boolean_flags():
    ResidualConfig(multiregion_appearance_enabled=True).validate()
    with pytest.raises(ValueError, match="sum to 1"):
        ResidualConfig(
            multiregion_torso_weight=0.8,
            multiregion_lower_weight=0.3,
        ).validate()
    with pytest.raises(ValueError, match="must be boolean"):
        ResidualConfig(multiregion_require_lower=1).validate()


def test_calibration_rejects_non_abstaining_source_before_dataset_access(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = ResidualConfig(residual_player_recovery_enabled=True).to_dict()
    (source / "run_manifest.json").write_text(json.dumps({
        "status": "COMPLETE", "variant": "V3"
    }), encoding="utf-8")
    (source / "benchmark_summary.json").write_text(json.dumps({
        "split": "train",
        "method": "residual-v3",
        "dataset_root": str(tmp_path / "GSR"),
        "configuration": config,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="abstaining/default"):
        _load_rows(tmp_path / "GSR", "train", source)
