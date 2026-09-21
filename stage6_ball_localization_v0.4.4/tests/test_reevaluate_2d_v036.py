from pathlib import Path
import csv

from ball_localization.datasets import SoccerNetV3DCSV, optimized_bbox_from_record
from ball_localization.evaluation.checkpointing import RecordCheckpointStore
from ball_localization.evaluation.common import sha256_file
from ball_localization.evaluation.metrics2d import diameter
from ball_localization.evaluation.reevaluate_2d import run_cached_2d_gt_comparison


def _write_modified_csv(tmp_path: Path) -> Path:
    src = Path(__file__).resolve().parents[1] / "validation_reports" / "synthetic_snv3d.csv"
    rows = list(csv.DictReader(src.open("r", encoding="utf-8")))
    rows[0]["optimized_d"] = "10.0"
    dst = tmp_path / "SNv3D.csv"
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return dst


def test_cached_reevaluation_uses_same_predictions_and_changes_only_gt(tmp_path: Path):
    csv_path = _write_modified_csv(tmp_path)
    ds = SoccerNetV3DCSV(csv_path)
    r = ds.split("test")[0]
    optimized_box = optimized_bbox_from_record(r)
    assert optimized_box is not None

    source = tmp_path / "source_v035"
    identity = {
        "benchmark": "2d",
        "stage6_version": "0.3.5",
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path),
        "image_root": str(tmp_path.resolve()),
        "split": "test",
        "records": 1,
        "detector": {"name": "synthetic", "top_k": 10},
        "candidate_k": 5,
    }
    store = RecordCheckpointStore(source, "stage6-ball-record-checkpoint-1.0", identity, 1, result_subdir="predictions")
    store.mark_running()
    store.commit(r.record_id, {
        "row_index": r.row_index,
        "image_path": None,
        "image_source_type": "synthetic",
        "img_w": r.img_w,
        "img_h": r.img_h,
        "action": r.action,
        "replay": r.replay,
        # Intentionally stale/original GT fields from a v0.3.5 cache. v0.3.6 must ignore these.
        "gt_bbox": r.ball_bbox,
        "gt_diameter_px": diameter(r.ball_bbox),
        "predictions": [{
            "bbox_xyxy": optimized_box,
            "score": 0.99,
            "diameter_px": diameter(optimized_box),
            "candidate_id": "p1",
            "source": "synthetic",
        }],
        "missing_image": False,
    })
    store.mark_complete()

    out = tmp_path / "reeval"
    report = run_cached_2d_gt_comparison(
        csv_path=csv_path,
        benchmark_2d_dir=source,
        split="test",
        output_dir=out,
        candidate_ks=(1, 5, 10),
        failure_images=0,
    )
    assert report["status"] == "COMPLETE"
    assert report["cached_records_found"] == 1
    assert report["metrics"]["optimized"]["AP50"] > 0.999
    assert report["metrics"]["optimized"]["CandidateRecall@10"] == 1.0
    assert report["metrics"]["original"]["AP50"] == 0.0
    assert report["delta_optimized_minus_original"]["AP50_optimized_minus_original"] > 0.99
    assert (out / "benchmark_2d_gt_comparison.json").is_file()
    assert (out / "frame_metrics_gt_comparison.csv").is_file()
