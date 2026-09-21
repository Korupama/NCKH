import json

from ball_localization.evaluation.issia3d import inspect_issia3d


def test_issia3d_preflight_accepts_sparse_public_gt_rows(tmp_path):
    csv_path = tmp_path / "ISSIA-3D.csv"
    csv_path.write_text(
        ",frame,x_cam1,y_cam1,opt_d_cam1,ball_3D\n"
        "0,10,100,200,8,\"[1.0, 2.0, -0.1]\"\n"
        "1,11,,,,\n",
        encoding="utf-8",
    )
    calibration = tmp_path / "issia_calibration.json"
    calibration.write_text(json.dumps({f"cam{i}": {} for i in range(1, 7)}), encoding="utf-8")
    report = inspect_issia3d(csv_path, calibration)
    assert report["status"] == "READY"
    assert report["rows"] == 2
    assert report["rows_with_gt_3d"] == 1
    assert report["oracle_2d_diameter_observations"]["cam1"] == 1
