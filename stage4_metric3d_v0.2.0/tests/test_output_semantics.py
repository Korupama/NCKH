from stage4_metric3d.output_semantics import build_stage5_handoff, recompute_selected_frame_metrics, selected_frame_availability


def _joint(name, xyz, legal=True):
    return {"name": name, "xyz_world_m": xyz, "candidate_for_legal_body_geometry": legal}


def test_zero_metric_xyz_is_missing_even_when_track_optimizer_converged():
    availability = selected_frame_availability(
        {"metric_pose23": [_joint("nose", None), _joint("left_hip", None)]},
        "VALID",
    )
    assert availability["status"] == "MISSING"
    assert availability["finite_metric_joint_count"] == 0


def test_repair_metrics_and_handoff_exclude_missing_selected_pose():
    observed = [_joint(name, [1.0, 2.0, 3.0]) for name in (
        "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    )]
    state = {
        "replay_context": {"selected_frame": 86},
        "world_frame": {"units": "metres"},
        "pose_schema": {"name": "METRIC_POSE_23"},
        "tracks": [
            {"track_id": "track_003", "selected_frame_pose_status": "VALID", "optimizer": {"status": "CONVERGED"}, "observations": [{"frame_index": 86, "metric_pose23": [_joint("nose", None)]}]},
            {"track_id": "track_004", "selected_frame_pose_status": "DEGRADED", "optimizer": {"status": "MAX_NFEV"}, "observations": [{"frame_index": 86, "metric_pose23": observed}]},
        ],
        "selected_frame_poses": [],
    }
    recompute_selected_frame_metrics(state)
    assert state["tracks"][0]["selected_frame_pose_status"] == "MISSING"
    assert state["tracks"][1]["selected_frame_pose_status"] == "DEGRADED"
    assert state["metrics"]["MetricPoseCoverageAtT0_given_stage3_candidate"] == 0.5
    handoff = build_stage5_handoff(state, "state.json")
    assert handoff["schema_version"] == "stage4-to-stage5-handoff-1.2"
    assert handoff["stage5_eligible_track_ids"] == ["track_004"]
    assert handoff["tracks"][1]["longitudinal_uncertainty"]["status"] == "NOT_RUN"
    assert "LONGITUDINAL_UNCERTAINTY_UNAVAILABLE" in handoff["tracks"][1]["qa"]["non_blocking"]


def test_handoff_exports_attack_direction_agnostic_extrema():
    observed = [_joint(name, [1.0, 2.0, 3.0]) for name in (
        "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    )]
    state = {
        "replay_context": {"selected_frame": 86},
        "world_frame": {"units": "metres"},
        "pose_schema": {"name": "METRIC_POSE_23"},
        "tracks": [{
            "track_id": "track_004",
            "selected_frame_pose_status": "VALID",
            "selected_frame_availability": {
                "status": "VALID", "finite_metric_joint_count": 8,
                "finite_legal_anchor_count": 8, "finite_core_anchor_count": 8,
                "finite_metric_joint_names": [], "finite_core_anchor_names": [],
            },
            "observations": [{"frame_index": 86, "metric_pose23": observed}],
            "selected_frame_uncertainty": {
                "status": "OK", "scope": "stage3_pixel_only_camera_fixed_selected_frame",
                "calibrated": False, "samples_requested": 16, "samples_usable": 16,
                "usable_fraction": 1.0,
                "longitudinal": {"legal_extrema_x": {"min_legal_x": {"std_m": 0.1}, "max_legal_x": {"std_m": 0.2}}},
            },
        }],
    }
    handoff = build_stage5_handoff(state, "state.json")
    entry = handoff["tracks"][0]
    assert entry["longitudinal_uncertainty"]["legal_extrema_x"]["max_legal_x"]["std_m"] == 0.2
    assert "UNCERTAINTY_NOT_CALIBRATED" in entry["qa"]["non_blocking"]
    assert handoff["uncertainty_contract"]["attack_direction_applied"] is False
