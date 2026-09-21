import json
import numpy as np

from stage4_metric3d.backends.field_converter.importer import import_field_converter_predictions, build_downstream_handoff


def test_importer_uses_source_world_and_nearest_source_frame(tmp_path):
    N, T, J = 2, 5, 25
    valid = np.ones((N, T), bool)
    roots_cam = np.zeros((N, T, 3), np.float32)
    roots_world = np.zeros((N, T, 3), np.float32)
    joints_cam = np.zeros((N, T, J, 3), np.float32)
    joints_source = np.zeros((N, T, J, 3), np.float32)
    for n in range(N):
        for t in range(T):
            roots_world[n, t] = [10+n, 20+t, 1]
            for j in range(J):
                joints_source[n, t, j] = [10+n + j*.01, 20+t + j*.005, j*.03]
                joints_cam[n, t, j] = [j*.01, j*.005, 5+j*.03]
    p = tmp_path / "predictions.npz"
    np.savez_compressed(
        p,
        valid_mask=valid,
        source_frame_numbers_float=np.array([10.0, 10.5, 11.0, 11.5, 12.0]),
        root_pred_m=roots_cam,
        root_source_world_pred_m=roots_world,
        root_init_cam_m=np.zeros_like(roots_cam),
        root_delta_pred_cam_m=np.zeros_like(roots_cam),
        joints_pred_cam_m=joints_cam,
        joints_pred_source_world_m=joints_source,
        joints_pred_2d=np.zeros((N,T,J,2), np.float32),
        world_alignment_rotation=np.eye(3, dtype=np.float32),
        output_fps=np.array(50.0),
        meta_json=np.array("{}", dtype=object),
    )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"proxy_reprojection_to_observed_sam2d": {"median_px": 3.0}}))
    state = import_field_converter_predictions(
        predictions_path=p, summary_path=summary,
        track_ids=["a", "b"], source_frames=[10,11,12], selected_frame=11,
        sam3d_joint_names=[f"j{i}" for i in range(25)], semantic_mapping_validated=False,
        camera_compatibility={"p95_px": 0.0}, camera_domain={"status": "IN_DOMAIN"},
        catastrophic_xy_span_m=4.0, catastrophic_z_span_m=4.0,
    )
    obs = next(o for o in state["tracks"][0]["observations"] if o["frame_index"] == 11)
    assert obs["root"]["pred_world_m"] == [10.0, 22.0, 1.0]
    assert obs["joints"][0]["xyz_world_m"] == [10.0, 22.0, 0.0]
    assert state["joint_schema"]["stage8_semantic_ready"] is False
    assert state["quality_gates"]["implementation_gate"] == "PASS"
    handoff = build_downstream_handoff(state, tmp_path / "state.json")
    assert handoff["schema_version"] == "stage4-downstream-handoff-2.0"
