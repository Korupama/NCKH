from pathlib import Path
import json

from ball_localization.evaluation.benchmark_3d_oracle import run_3d_oracle_benchmark
from ball_localization.evaluation.benchmark_3d_e2e import run_3d_e2e_benchmark
from ball_localization.evaluation.checkpointing import RecordCheckpointStore
from ball_localization.evaluation.common import sha256_file
from ball_localization.evaluation.reporting import compose_combined_report
from ball_localization.datasets import SoccerNetV3DCSV


def _synthetic_csv(tmp_path: Path) -> Path:
    src = Path(__file__).resolve().parents[1] / "validation_reports" / "synthetic_snv3d.csv"
    dst = tmp_path / "SNv3D.csv"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def _make_perfect_2d_cache(tmp_path: Path, csv_path: Path) -> Path:
    ds = SoccerNetV3DCSV(csv_path)
    records = ds.split("test")
    out = tmp_path / "two_d"
    identity = {
        "benchmark": "2d",
        "stage6_version": "0.3.0",
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path),
        "image_root": str(tmp_path.resolve()),
        "split": "test",
        "records": len(records),
        "detector": {"name":"synthetic"},
        "candidate_k": 5,
    }
    store = RecordCheckpointStore(out, "stage6-ball-record-checkpoint-1.0", identity, len(records), result_subdir="predictions")
    store.mark_running()
    for r in records:
        store.commit(r.record_id, {
            "row_index": r.row_index,
            "image_path": None,
            "img_w": r.img_w,
            "img_h": r.img_h,
            "action": r.action,
            "replay": r.replay,
            "gt_bbox": r.ball_bbox,
            "gt_diameter_px": r.optimized_d,
            "predictions": [{"bbox_xyxy": r.ball_bbox, "score": 0.99, "diameter_px": r.optimized_d, "candidate_id":"p1", "source":"synthetic"}],
            "missing_image": False,
        })
    store.mark_complete()
    return out


def test_oracle_and_e2e_synthetic_geometry(tmp_path: Path):
    csv_path = _synthetic_csv(tmp_path)
    oracle_dir = tmp_path / "oracle"
    oracle = run_3d_oracle_benchmark(
        csv_path=csv_path,
        split="test",
        output_dir=oracle_dir,
        diameter_source="optimized",
        progress_every=0,
    )
    assert oracle["metrics"]["coverage"] == 1.0
    assert oracle["metrics"]["MAE_3D_m"] < 0.05
    assert (oracle_dir / "stratification.csv").is_file()

    two_d_dir = _make_perfect_2d_cache(tmp_path, csv_path)
    e2e_dir = tmp_path / "e2e"
    e2e = run_3d_e2e_benchmark(
        csv_path=csv_path,
        benchmark_2d_dir=two_d_dir,
        split="test",
        output_dir=e2e_dir,
        progress_every=0,
        failure_images=0,
    )
    assert e2e["metrics"]["coverage"] == 1.0
    assert e2e["metrics"]["MAE_3D_m"] < 0.20
    assert (e2e_dir / "frame_metrics.csv").is_file()

    report_dir = tmp_path / "report"
    combined = compose_combined_report(report_dir, oracle_optimized_dir=oracle_dir, e2e_dir=e2e_dir)
    assert "e2e_minus_oracle_MAE_3D_m" in combined["error_budget"]
    assert (report_dir / "benchmark_report.md").is_file()


def test_combined_report_rejects_pre_v032_3d_summary(tmp_path: Path):
    old_dir=tmp_path/'old_oracle'; old_dir.mkdir()
    (old_dir/'benchmark_3d_oracle_summary.json').write_text(json.dumps({
        'schema_version':'stage6-ball-3d-oracle-benchmark-1.1',
        'stage6_version':'0.3.1',
        'metrics':{'coverage':0.038},
    }),encoding='utf-8')
    import pytest
    with pytest.raises(RuntimeError, match='pre-v0.3.2'):
        compose_combined_report(tmp_path/'report_old', oracle_optimized_dir=old_dir)
