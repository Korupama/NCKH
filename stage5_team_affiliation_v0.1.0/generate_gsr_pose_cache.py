#!/usr/bin/env python3
"""Generate an oracle-track RTMW pose cache for the Stage-5 GSR benchmark.

The cache uses GSR boxes and track IDs only to organize model outputs. Team labels
are never read. It is an oracle-box pose benchmark, not an end-to-end Stage-2 run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from stage5_team_affiliation.config import Stage5Config
from stage5_team_affiliation.gsr_benchmark import SoccerNetGSRDataset, GSRAnn
from stage3_pose2d.rtmw_onnx import RTMWOpenCVDNN  # type: ignore


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sampled(anns: list[GSRAnn], cfg: Stage5Config) -> list[GSRAnn]:
    ordered = sorted(anns, key=lambda a: a.frame_index)[::cfg.sample_every_n_frames]
    if len(ordered) <= cfg.max_samples_per_track:
        return ordered
    ids = np.linspace(0, len(ordered) - 1, cfg.max_samples_per_track).round().astype(int)
    return [ordered[i] for i in ids]


def keypoint_records(pose, threshold: float) -> list[dict]:
    names = __import__("stage3_pose2d.wholebody133", fromlist=["WHOLEBODY_KEYPOINT_NAMES"]).WHOLEBODY_KEYPOINT_NAMES
    rows = []
    for i, name in enumerate(names):
        xy = pose.keypoints_xy[i]
        score = float(pose.scores[i])
        finite = bool(np.isfinite(xy).all())
        rows.append({
            "index": i,
            "name": name,
            "x": float(xy[0]) if finite else None,
            "y": float(xy[1]) if finite else None,
            "raw_model_score": score,
            "state": "VALID" if finite and score >= threshold else "LOW_MODEL_EVIDENCE",
        })
    return rows


def build_sequence(seq, estimator: RTMWOpenCVDNN, out: Path, cfg: Stage5Config, threshold: float) -> dict:
    selected = {tid: sampled(anns, cfg) for tid, anns in seq.anns_by_track.items()}
    by_frame: dict[int, list[GSRAnn]] = {}
    for anns in selected.values():
        for ann in anns:
            if ann.role in {"player", "goalkeeper"}:
                by_frame.setdefault(ann.frame_index, []).append(ann)

    observations: dict[str, list[dict]] = {tid: [] for tid in selected}
    for frame_index in sorted(by_frame):
        frame = seq.read_frame(frame_index)
        if frame is None:
            raise FileNotFoundError(f"{seq.sequence_id}: frame {frame_index} cannot be read")
        anns = by_frame[frame_index]
        poses = estimator.infer_one_batch(frame, [list(a.bbox_xyxy) for a in anns]) if hasattr(estimator, "infer_one_batch") else [estimator.infer_one(frame, list(a.bbox_xyxy)) for a in anns]
        for ann, pose in zip(anns, poses):
            observations[ann.track_id].append({
                "frame_index": ann.frame_index,
                "source_bbox_xyxy": list(ann.bbox_xyxy),
                "keypoints_133": keypoint_records(pose, threshold),
                "pose_status": "VALID",
            })

    tracks = [{"track_id": tid, "observations": sorted(obs, key=lambda x: x["frame_index"])} for tid, obs in observations.items() if obs]
    payload = {
        "schema_version": "stage5-gsr-pose-cache-1.0",
        "sequence_id": seq.sequence_id,
        "model": {"name": "RTMW-L WholeBody 384x288", "sha256": sha256(Path(estimator.model_path))},
        "oracle_inputs": ["SoccerNet-GSR bbox_image", "SoccerNet-GSR track_id"],
        "team_labels_used": False,
        "tracks": tracks,
    }
    target = out / seq.sequence_id / "stage5_gsr_pose_cache.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"sequence_id": seq.sequence_id, "tracks": len(tracks), "observations": sum(len(t["observations"]) for t in tracks), "path": str(target)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gsr-root", type=Path, required=True)
    p.add_argument("--split", default="valid")
    p.add_argument("--rtmw-model", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--sample-every", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=30)
    p.add_argument("--score-threshold", type=float, default=1.0)
    args = p.parse_args()

    cfg = Stage5Config(sample_every_n_frames=args.sample_every, max_samples_per_track=args.max_samples)
    cfg.validate()
    dataset = SoccerNetGSRDataset(args.gsr_root, args.split)
    sequence_ids = dataset.discover()
    sequence_ids = sequence_ids[args.start:]
    if args.limit:
        sequence_ids = sequence_ids[:args.limit]
    estimator = RTMWOpenCVDNN(args.rtmw_model, device="cpu")
    reports = []
    for i, sid in enumerate(sequence_ids, 1):
        target = args.output_root / sid / "stage5_gsr_pose_cache.json"
        if args.skip_existing and target.is_file():
            cached = json.loads(target.read_text(encoding="utf-8"))
            report = {
                "sequence_id": sid,
                "tracks": len(cached.get("tracks") or []),
                "observations": sum(len(t.get("observations") or []) for t in cached.get("tracks") or []),
                "path": str(target),
                "reused": True,
            }
            reports.append(report)
            print(f"[{i}/{len(sequence_ids)}] {sid}: reused", flush=True)
            continue
        seq = dataset.load(sid)
        report = build_sequence(seq, estimator, args.output_root, cfg, args.score_threshold)
        reports.append(report)
        print(f"[{i}/{len(sequence_ids)}] {sid}: {report['tracks']} tracks, {report['observations']} observations", flush=True)
    (args.output_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps({"schema_version": "stage5-gsr-pose-cache-1.0", "sequences": reports}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
