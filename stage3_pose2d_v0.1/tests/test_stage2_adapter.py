from __future__ import annotations
import pytest
from stage3_pose2d.stage2_adapter import load_stage2_bundle
from conftest import write_stage2_fixture


def test_adapter_accepts_stage2_contract(stage2_fixture):
    b=load_stage2_bundle(stage2_dir=stage2_fixture)
    assert b.validation["ready"] is True
    assert b.selected_frame==10
    assert b.candidate_track_ids==["track_001"]


def test_adapter_rejects_coordinate_space(tmp_path):
    d=write_stage2_fixture(tmp_path/"bad",bad_coord=True)
    with pytest.raises(ValueError,match="RAW_DISTORTED_PIXEL"):
        load_stage2_bundle(stage2_dir=d)


def test_missing_t0_pose_is_warning_not_structural_failure(tmp_path):
    d=write_stage2_fixture(tmp_path/"missing",missing_t0=True)
    b=load_stage2_bundle(stage2_dir=d)
    assert b.validation["ready"] is True
    assert b.validation["missing_pose_at_selected_frame"]==["track_001"]
