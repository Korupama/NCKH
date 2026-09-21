import json
from stage3_pose2d.processor import run_stage3
from stage3_pose2d.schemas import Stage3Config
from conftest import write_stage2_fixture


def test_processor_end_to_end(stage2_fixture,tmp_path):
    out=tmp_path/"out"
    state=run_stage3(stage2_dir=stage2_fixture,output_dir=out)
    assert state["schema_version"]=="tracked-pose-2d-state-1.0"
    assert state["keypoint_schema"]["count"]==133
    assert state["metrics"]["PoseCoverageAtT0_given_stage2_candidate"]==1.0
    assert state["diagnostics"]["legal_body_semantics_applied"] is False
    assert (out/"tracked_pose_2d_state.json").is_file()
    h=json.loads((out/"stage3_downstream_handoff.json").read_text())
    assert h["valid_track_ids"]==["track_001"]


def test_processor_missing_t0_pose_remains_explicit(tmp_path):
    d=write_stage2_fixture(tmp_path/"s2",missing_t0=True)
    state=run_stage3(stage2_dir=d,output_dir=tmp_path/"out")
    tr=state["tracks"][0]
    assert tr["selected_frame_pose_status"]=="MISSING"
    assert state["metrics"]["PoseCoverageAtT0_given_stage2_candidate"]==0.0
