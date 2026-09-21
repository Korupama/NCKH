from pathlib import Path

from ball_localization.datasets import SoccerNetV3DCSV
from ball_localization.evaluation.benchmark_2d import finalize_2d
from ball_localization.evaluation.checkpointing import RecordCheckpointStore
from ball_localization.evaluation.common import sha256_file
from ball_localization.evaluation.metrics2d import diameter


def test_2d_finalize_writes_failure_and_stratification(tmp_path: Path):
    src = Path(__file__).resolve().parents[1] / "validation_reports" / "synthetic_snv3d.csv"
    csv_path = tmp_path / "SNv3D.csv"
    csv_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    ds = SoccerNetV3DCSV(csv_path)
    records = ds.split("test")
    out = tmp_path / "run2d"
    identity = {
        "benchmark":"2d","stage6_version":"0.3.0","csv":str(csv_path.resolve()),
        "csv_sha256":sha256_file(csv_path),"image_root":str(tmp_path),"split":"test",
        "records":1,"detector":{"name":"synthetic"},"candidate_k":5,
    }
    store = RecordCheckpointStore(out, "stage6-ball-record-checkpoint-1.0", identity, 1, result_subdir="predictions")
    r = records[0]
    store.commit(r.record_id, {
        "row_index":r.row_index,"image_path":None,"img_w":r.img_w,"img_h":r.img_h,
        "action":r.action,"replay":r.replay,"gt_bbox":r.ball_bbox,"gt_diameter_px":diameter(r.ball_bbox),
        "predictions":[{"bbox_xyxy":r.ball_bbox,"score":0.9,"diameter_px":diameter(r.ball_bbox)}],"missing_image":False,
    })
    store.mark_complete()
    report = finalize_2d(store=store, records=records, dataset=ds, candidate_k=5, failure_images=0, partial=False)
    assert report["metrics"]["AP50"] > 0.999
    assert (out / "frame_metrics.csv").is_file()
    assert (out / "stratification.csv").is_file()
    assert (out / "failure_cases.json").is_file()
