from stage4_metric3d.visualization import save_topdown_selected_frame


def test_topdown_export_draws_metric_pitch(tmp_path):
    state = {
        "replay_context": {"selected_frame": 86},
        "selected_frame_poses": [
            {
                "track_id": "track_001",
                "observation": {
                    "metric_pose23": [
                        {"xyz_world_m": [1.0, 2.0, 0.0]},
                        {"xyz_world_m": [1.5, 2.5, 1.0]},
                    ]
                },
            }
        ],
    }

    output = save_topdown_selected_frame(state, tmp_path / "topdown.png")

    assert output.exists()
    assert output.stat().st_size > 10_000
