from pathlib import Path
import json

from ball_localization.evaluation.contact_preflight_v142 import preflight_contact_manifest
from ball_localization.evaluation.footpass_manifest_v14 import build_footpass_contact_manifest


def test_preflight_detects_placeholder_state(tmp_path: Path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"metadata":{"frozen":True},"cases":[{
        "case_id":"c1","state_json":"relative/path/to/ball_trajectory_state.json",
        "gt_track_id":"t1"
    }]}), encoding="utf-8")
    report = preflight_contact_manifest(manifest)
    assert report["status"] == "NOT_READY"
    assert any(x["kind"] == "PLACEHOLDER_STATE_PATH" for x in report["issues"])


def test_preflight_ready_with_real_state_and_direct_track_gt(tmp_path: Path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"selected_frame_ball":{"contact":{"track_id":"t1","region":"FOOT"}}}), encoding="utf-8")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"metadata":{"frozen":True},"cases":[{
        "case_id":"c1","state_json":"state.json","gt_track_id":"t1","gt_region":"FOOT"
    }]}), encoding="utf-8")
    report = preflight_contact_manifest(manifest)
    assert report["status"] == "READY"
    assert report["metric_readiness"]["contact_identity"]["ready"] is True
    assert report["metric_readiness"]["foot_vs_nonfoot"]["ready"] is True
    assert report["metric_readiness"]["production_ball_x"]["ready"] is False


def test_frozen_builder_rejects_placeholder(tmp_path: Path):
    play = tmp_path / "play.json"
    play.write_text(json.dumps({"events":{"game_18_H1":[[64,0,3,1,1,0]]}}), encoding="utf-8")
    bridge = tmp_path / "bridge.csv"
    bridge.write_text(
        "case_id,game_key,frame,team,jersey,action_class,state_json,gt_track_id,track_identity_map_json,gt_region,gt_contact_binary,gt_x_m,split\n"
        "c1,game_18_H1,64,0,3,Drive,relative/path/to/ball_trajectory_state.json,t1,,,,,validation\n",
        encoding="utf-8"
    )
    try:
        build_footpass_contact_manifest(playbyplay_json=play, bridge_csv=bridge, output_json=tmp_path/'out.json', frozen=True)
    except ValueError as exc:
        assert 'placeholder state_json' in str(exc)
    else:
        raise AssertionError('expected ValueError')
