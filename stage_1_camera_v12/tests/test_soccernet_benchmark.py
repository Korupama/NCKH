from pathlib import Path
import json
import zipfile

from stage_1_camera.evaluation.soccernet_benchmark import (
    SoccerNetBenchmarkConfig,
    inspect_dataset,
    select_image_paths,
    summarize_records,
    ensure_split_extracted,
)


def test_select_uniform(tmp_path):
    for i in range(10):
        (tmp_path / f"{i:03d}.jpg").write_bytes(b"x")
    selected = select_image_paths(tmp_path, limit=4, strategy="uniform")
    assert len(selected) == 4
    assert selected[0].name == "000.jpg"
    assert selected[-1].name == "009.jpg"


def test_inspect_dataset_pairing(tmp_path):
    d = tmp_path / "valid"
    d.mkdir()
    for stem in ("a", "b"):
        (d / f"{stem}.jpg").write_bytes(b"x")
        (d / f"{stem}.json").write_text("{}")
    result = inspect_dataset(tmp_path, splits=("valid",))
    assert result["splits"]["valid"]["ready"] is True
    assert result["splits"]["valid"]["num_paired"] == 2


def test_ensure_extract(tmp_path):
    with zipfile.ZipFile(tmp_path / "test.zip", "w") as z:
        z.writestr("a.jpg", b"x")
        z.writestr("a.json", "{}")
    d = ensure_split_extracted(tmp_path, "test")
    assert (d / "a.jpg").exists()


def test_summary_metrics():
    cfg = SoccerNetBenchmarkConfig(split="valid", limit=2)
    rows = [
        {
            "solved": True, "official_prediction_written": True, "camera_status": "VALID",
            "ground_status": "VALID", "vertical_3d_status": "VALID", "offside_3d_ready": True,
            "mode": "full", "use_ransac": 0, "rep_err_px": 2.0, "runtime_ms": 100,
            "official_frame_accuracy": 0.8, "official_frame_precision": 0.7, "official_frame_recall": 0.6,
            "num_keypoints": 10, "num_lines": 2, "keypoint_hull_ratio": .1, "keypoint_x_span_ratio": .5,
            "keypoint_y_span_ratio": .2,
        },
        {
            "solved": False, "official_prediction_written": False, "camera_status": "INVALID",
            "ground_status": "INVALID", "vertical_3d_status": "INVALID", "offside_3d_ready": False,
            "mode": None, "use_ransac": None, "rep_err_px": None, "runtime_ms": 200,
            "official_frame_accuracy": None, "official_frame_precision": None, "official_frame_recall": None,
            "num_keypoints": 0, "num_lines": 0, "keypoint_hull_ratio": 0, "keypoint_x_span_ratio": 0,
            "keypoint_y_span_ratio": 0,
        },
    ]
    s = summarize_records(rows, cfg)
    assert s["solver_coverage"] == 0.5
    assert s["official_completeness"] == 0.5
    assert abs(s["official_like_metrics"]["finalScore"] - 0.4) < 1e-9


def test_review_vertical_policy_csv_reclassifies_without_inference(tmp_path):
    import csv
    from stage_1_camera.evaluation.soccernet_benchmark import review_vertical_policy_csv

    path = tmp_path / 'valid_frames.csv'
    fields = [
        'camera_status', 'mode', 'use_ransac', 'rep_err_px', 'num_keypoints',
        'keypoint_quadrants', 'keypoint_hull_ratio', 'keypoint_x_span_ratio',
        'keypoint_y_span_ratio', 'offside_3d_ready', 'official_frame_accuracy',
    ]
    rows = [
        dict(camera_status='VALID', mode='full', use_ransac='0', rep_err_px='3.0', num_keypoints='12',
             keypoint_quadrants='4', keypoint_hull_ratio='0.2', keypoint_x_span_ratio='0.5',
             keypoint_y_span_ratio='0.3', offside_3d_ready='True', official_frame_accuracy='1.0'),
        dict(camera_status='VALID', mode='full', use_ransac='0', rep_err_px='3.0', num_keypoints='7',
             keypoint_quadrants='4', keypoint_hull_ratio='0.2', keypoint_x_span_ratio='0.5',
             keypoint_y_span_ratio='0.3', offside_3d_ready='True', official_frame_accuracy='0.5'),
        dict(camera_status='DEGRADED', mode='ground_plane', use_ransac='5', rep_err_px='2.0', num_keypoints='16',
             keypoint_quadrants='4', keypoint_hull_ratio='0.2', keypoint_x_span_ratio='0.5',
             keypoint_y_span_ratio='0.3', offside_3d_ready='False', official_frame_accuracy='1.0'),
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    report = review_vertical_policy_csv(path)
    assert report['num_frames'] == 3
    assert report['old_ready_count'] == 2
    assert report['v12_ready_count'] == 1
    assert report['demoted_from_old_ready'] == 1
    assert report['promoted_from_old_not_ready'] == 0
    assert report['official_frame_accuracy_on_v12_ready']['mean'] == 1.0
