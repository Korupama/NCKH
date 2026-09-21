from pathlib import Path

import pytest

from ball_localization.datasets import SoccerNetV3DCSV, gt_bbox_for_record, optimized_bbox_from_record
from ball_localization.evaluation.metrics2d import center, diameter


def test_optimized_box_preserves_center_and_uses_optimized_d(tmp_path: Path):
    src = Path(__file__).resolve().parents[1] / "validation_reports" / "synthetic_snv3d.csv"
    csv_path = tmp_path / "SNv3D.csv"
    csv_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    record = SoccerNetV3DCSV(csv_path).split("test")[0]

    box = optimized_bbox_from_record(record)
    assert box is not None
    assert center(box) == pytest.approx(center(record.ball_bbox))
    assert diameter(box) == pytest.approx(record.optimized_d)
    assert (box[2] - box[0]) == pytest.approx(record.optimized_d)
    assert (box[3] - box[1]) == pytest.approx(record.optimized_d)
    assert gt_bbox_for_record(record, "original") == pytest.approx(record.ball_bbox)
    assert gt_bbox_for_record(record, "optimized") == pytest.approx(box)
