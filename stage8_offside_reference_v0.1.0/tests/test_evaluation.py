import json
from pathlib import Path
from stage8_offside_reference.evaluation import evaluate_manifest
from helpers import joint, track, stage4, stage6, stage7


def test_partial_gt_denominators(tmp_path: Path):
    for name, payload in {
        "s4.json": stage4(104,[track("d1",104,[joint("nose",50)]),track("d2",104,[joint("nose",47)])]),
        "s6.json": stage6(104,(44,44.22)),
        "s7.json": stage7(104,1,["d1","d2"]),
    }.items():
        (tmp_path/name).write_text(json.dumps(payload),encoding="utf-8")
    manifest={"cases":[{"case_id":"c1","stage4":"s4.json","stage6":"s6.json","stage7":"s7.json","gt":{"status":"VALID","second_last_track_id":"d2","reference_source":"SECOND_LAST_OPPONENT"}}]}
    mp=tmp_path/"manifest.json"; mp.write_text(json.dumps(manifest),encoding="utf-8")
    report=evaluate_manifest(str(mp),str(tmp_path/"out"))
    assert report["metrics"]["status_accuracy"] == 1.0
    assert report["metrics"]["second_last_opponent_agreement"] == 1.0
    assert report["metrics"]["reference_source_agreement"] == 1.0
    assert report["metrics"]["reference_x_mae_m"] is None
