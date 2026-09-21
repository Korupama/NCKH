from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import json
import time

from ..consolidation import consolidate_human_detections
from .metrics_detection import (
    DetectionAccumulator,
    evaluate_selected_frame,
    gt_cross_class_duplicate_rate,
    gt_post_consolidation_duplicate_rate,
)
from .soccernet_gsr import SoccerNetGSRDataset, HUMAN_ROLES
from .reporting import write_json, write_rows_csv


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest_index(manifest: Mapping[str, Any]) -> Dict[int, Path]:
    return {
        int(x["frame_index"]): Path(str(x["perception_json"]))
        for x in manifest.get("frames", [])
    }


def _active_raw(records: Sequence[Mapping[str, Any]], thresholds: Mapping[int, float]) -> List[Dict[str, Any]]:
    return [
        dict(r)
        for r in records
        if float(r.get("score", 0.0)) >= float(thresholds.get(int(r.get("label_id", -1)), 1.0))
    ]


def _pred_from_humans(humans: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "bbox_xyxy": list(h["bbox_xyxy"]),
            "role": str(h.get("resolved_role", "other")),
            "candidate_for_stage3": bool(h.get("candidate_hint", False)),
            "score": float(h.get("detector_score", 0.0)),
        }
        for h in humans
    ]


def parse_float_grid(text: str) -> Tuple[float, ...]:
    values=[]
    for token in str(text).split(","):
        token=token.strip()
        if not token:
            continue
        value=float(token)
        if not 0 <= value <= 1:
            raise ValueError("threshold values must be in [0,1]")
        values.append(value)
    if not values:
        raise ValueError("threshold grid cannot be empty")
    return tuple(sorted(set(values)))


def run_threshold_sweep(
    *,
    dataset_root: str | Path,
    split: str,
    benchmark_dir: str | Path,
    player_thresholds: Sequence[float],
    goalkeeper_thresholds: Sequence[float],
    referee_threshold: float = 0.55,
    staff_threshold: float = 0.60,
    ball_threshold: float = 0.35,
    consolidation_high_iou: float = 0.82,
    consolidation_low_iou: float = 0.60,
    consolidation_max_center_distance: float = 0.18,
    consolidation_min_area_similarity: float = 0.65,
    consolidation_min_intersection_over_min: float = 0.75,
    output_dir: str | Path | None = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    bench=Path(benchmark_dir).expanduser().resolve()
    protocol_path=bench/"protocol_windows.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing {protocol_path}. Run benchmark_stage2.py run first.")
    windows=_load_json(protocol_path)
    if not isinstance(windows,list) or not windows:
        raise ValueError("protocol_windows.json is empty or invalid")

    ds=SoccerNetGSRDataset(dataset_root, split)
    seq_cache={}
    manifest_cache={}
    payload_cache={}

    # Validate that the cache actually contains M1 low-score output and learn its floor.
    floors=[]
    for item in windows:
        sid=str(item["sequence_id"]); t0=int(item["target_frame"])
        if sid not in manifest_cache:
            mp=bench/"sequences"/sid/"perception"/"perception_manifest.json"
            if not mp.is_file():
                raise FileNotFoundError(mp)
            manifest_cache[sid]=_load_json(mp)
        manifest=manifest_cache[sid]
        floors.append(float((manifest.get("configuration") or {}).get("raw_human_score_floor", 1.0)))
        idx=_manifest_index(manifest)
        if t0 not in idx:
            raise KeyError(f"{sid}: perception cache missing target frame {t0}")
        payload=_load_json(idx[t0])
        if "raw_low_score_detections" not in payload:
            raise RuntimeError(
                f"{sid} frame {t0}: cache predates Stage-2 v1.3 raw_low_score_detections. "
                "Rebuild perception once with v1.3 before threshold sweeping."
            )
        payload_cache[(sid,t0)]=payload
        if sid not in seq_cache:
            seq_cache[sid]=ds.load(sid)
    floor=max(floors) if floors else 1.0
    requested_min=min(min(player_thresholds),min(goalkeeper_thresholds))
    if requested_min < floor - 1e-12:
        raise ValueError(
            f"Requested threshold {requested_min:.3f} is below cached raw_human_score_floor={floor:.3f}. "
            "Rebuild perception with a lower --raw-human-score-floor."
        )

    out=Path(output_dir).expanduser().resolve() if output_dir else bench/"threshold_sweep"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    total=len(player_thresholds)*len(goalkeeper_thresholds)
    started=time.perf_counter(); done=0
    for pt in player_thresholds:
        for gt_thr in goalkeeper_thresholds:
            acc=DetectionAccumulator()
            for item in windows:
                sid=str(item["sequence_id"]); t0=int(item["target_frame"])
                seq=seq_cache[sid]
                payload=payload_cache[(sid,t0)]
                thresholds={
                    1:float(ball_threshold),2:float(pt),3:float(gt_thr),
                    4:float(referee_threshold),5:float(referee_threshold),6:float(staff_threshold),
                }
                active=_active_raw(payload["raw_low_score_detections"],thresholds)
                humans=consolidate_human_detections(
                    active,
                    high_iou_threshold=consolidation_high_iou,
                    low_iou_threshold=consolidation_low_iou,
                    max_center_distance=consolidation_max_center_distance,
                    min_area_similarity=consolidation_min_area_similarity,
                    min_intersection_over_min=consolidation_min_intersection_over_min,
                )
                human_dicts=[h.to_dict() for h in humans]
                pred=_pred_from_humans(human_dicts)
                gt=[g.to_dict() for g in seq.gt(t0,roles=HUMAN_ROLES)]
                ev=evaluate_selected_frame(pred,gt,iou_threshold=0.5)
                raw_dup=gt_cross_class_duplicate_rate(active,gt)
                post_dup=gt_post_consolidation_duplicate_rate(human_dicts,gt)
                acc.add(ev,raw_dup,post_dup)
            summary=acc.summary()
            row={
                "player_threshold":float(pt),
                "goalkeeper_threshold":float(gt_thr),
                "CandidatePrecision":summary["CandidatePrecision"],
                "CandidateRecall":summary["CandidateRecall"],
                "RefereeLeakageRate":summary["RefereeLeakageRate"],
                "AnyDuplicateRate_after":summary["AnyDuplicateRate_after"],
                "CrossClassConflictRate_after":summary["CrossClassConflictRate_after"],
                "PlayerRecall":summary["role_metrics"]["player"]["recall"],
                "GoalkeeperRecall":summary["role_metrics"]["goalkeeper"]["recall"],
                "RoleMacroF1":summary["role_macro_f1_supported"],
            }
            row["hard_detection_gate"]=(
                row["CandidatePrecision"]>=0.95
                and row["CandidateRecall"]>=0.95
                and row["RefereeLeakageRate"]<=0.02
                and row["AnyDuplicateRate_after"]<0.02
            )
            rows.append(row)
            done+=1
            if not quiet:
                elapsed=time.perf_counter()-started
                eta=(elapsed/done)*(total-done) if done else 0.0
                print(
                    f"[THRESHOLD SWEEP {done}/{total}] P={pt:.2f} GK={gt_thr:.2f} "
                    f"CandP={row['CandidatePrecision']:.3f} CandR={row['CandidateRecall']:.3f} "
                    f"RefLeak={row['RefereeLeakageRate']:.3f} ETA={eta:.1f}s",
                    flush=True,
                )

    # Prefer configs that satisfy the hard detection gate, then maximize candidate recall,
    # candidate precision, and exact role F1. If none pass, report the best recall/precision tradeoff.
    ranked=sorted(
        rows,
        key=lambda r:(
            bool(r["hard_detection_gate"]),
            float(r["CandidateRecall"]),
            float(r["CandidatePrecision"]),
            -float(r["RefereeLeakageRate"]),
            float(r["RoleMacroF1"]),
        ),
        reverse=True,
    )
    report={
        "schema_version":"stage2-threshold-sweep-1.0",
        "benchmark_dir":str(bench),
        "split":split,
        "windows":len(windows),
        "raw_human_score_floor":floor,
        "grid":{"player":list(player_thresholds),"goalkeeper":list(goalkeeper_thresholds)},
        "recommended":ranked[0] if ranked else None,
        "rows":rows,
        "note":"Selected-frame threshold sweep only; it reuses cached SST output and does not rerun RTMW/tracking.",
    }
    write_json(out/"threshold_sweep.json",report)
    write_rows_csv(out/"threshold_sweep.csv",ranked)
    return report
