from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import hashlib

import cv2
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, precision_recall_fscore_support

from .clustering import fit_two_teams
from .config import Stage5Config
from .features import aggregate_features, extract_color_feature, fuse_region_features
from .goalkeeper import assign_goalkeeper, lower_body_centroids_by_team
from .regions import region_polygon

TEAM_TO_INT = {"left": 0, "right": 1}
INT_TO_TEAM = {0: "left", 1: "right"}
TEAM_ROLES = {"player", "goalkeeper"}


def _bbox_xyxy(ann: Mapping[str, Any]) -> Optional[List[float]]:
    b = ann.get("bbox_image") or ann.get("bbox")
    if not isinstance(b, Mapping):
        if isinstance(b, Sequence) and len(b) == 4:
            x, y, w, h = map(float, b)
            return [x, y, x + w, y + h]
        return None
    try:
        x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def _role(v: Any) -> str:
    return str(v or "").strip().lower()


def _team(v: Any) -> Optional[str]:
    s = str(v or "").strip().lower()
    return s if s in TEAM_TO_INT else None


@dataclass(frozen=True)
class GSRAnn:
    frame_index: int
    image_id: str
    track_id: str
    role: str
    team: Optional[str]
    bbox_xyxy: Tuple[float, float, float, float]


@dataclass
class GSRSequence:
    sequence_id: str
    sequence_dir: Path
    labels_path: Path
    version: str
    fps: float
    width: int
    height: int
    images: List[Dict[str, Any]]
    anns_by_frame: Dict[int, List[GSRAnn]]
    anns_by_track: Dict[str, List[GSRAnn]]
    video_path: Optional[Path]

    @property
    def num_frames(self) -> int:
        return len(self.images)

    def resolve_image_path(self, frame_index: int) -> Optional[Path]:
        if frame_index < 0 or frame_index >= len(self.images):
            return None
        rec = self.images[frame_index]
        raw = rec.get("file_name") or rec.get("path") or rec.get("file_path")
        if not raw:
            return None
        p = Path(str(raw))
        candidates = [
            p if p.is_absolute() else self.sequence_dir / p,
            self.sequence_dir / "img1" / p.name,
            self.sequence_dir / "frames" / p.name,
            self.sequence_dir / "images" / p.name,
            self.sequence_dir.parent / p,
        ]
        for q in candidates:
            if q.is_file():
                return q.resolve()
        return None

    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        p = self.resolve_image_path(frame_index)
        if p is not None:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None:
                return im
        if self.video_path is not None:
            cap = cv2.VideoCapture(str(self.video_path))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    return frame
            else:
                cap.release()
        return None

    def team_gt_by_track(self, role: Optional[str] = None) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for tid, anns in self.anns_by_track.items():
            votes: Dict[str, int] = {}
            roles: Dict[str, int] = {}
            for a in anns:
                roles[a.role] = roles.get(a.role, 0) + 1
                if a.team is not None:
                    votes[a.team] = votes.get(a.team, 0) + 1
            stable_role = max(roles, key=roles.get) if roles else ""
            if role is not None and stable_role != role:
                continue
            if stable_role not in TEAM_ROLES or not votes:
                continue
            team = max(votes, key=votes.get)
            out[tid] = TEAM_TO_INT[team]
        return out

    def role_by_track(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for tid, anns in self.anns_by_track.items():
            votes: Dict[str, int] = {}
            for a in anns:
                votes[a.role] = votes.get(a.role, 0) + 1
            if votes:
                out[tid] = max(votes, key=votes.get)
        return out


class SoccerNetGSRDataset:
    def __init__(self, root: str | Path, split: str = "valid", require_version_13: bool = True):
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.split_dir = self.root / self.split
        self.require_version_13 = bool(require_version_13)
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Expected SoccerNet-GSR split at {self.split_dir}")

    def labels_paths(self) -> Dict[str, Path]:
        result: Dict[str, Path] = {}
        for p in sorted(self.split_dir.rglob("Labels-GameState.json")):
            sid = p.parent.name
            if sid in result:
                raise RuntimeError(f"Duplicate sequence id {sid}: {result[sid]} and {p}")
            result[sid] = p
        return result

    def discover(self) -> List[str]:
        return list(self.labels_paths())

    def load(self, sequence_id: str) -> GSRSequence:
        paths = self.labels_paths()
        if sequence_id not in paths:
            raise KeyError(sequence_id)
        lp = paths[sequence_id]
        d = json.loads(lp.read_text(encoding="utf-8"))
        info = dict(d.get("info") or {})
        version = str(info.get("version", "unknown"))
        if self.require_version_13:
            try:
                parts = tuple(int(x) for x in version.split(".")[:2])
            except Exception:
                parts = ()
            if parts and parts < (1, 3):
                raise ValueError(f"{sequence_id}: SoccerNet-GSR v1.3+ required, found {version}")
        images = [dict(x) for x in d.get("images") or []]
        if not images:
            raise ValueError(f"{sequence_id}: no images")
        # Released labels are chronological. Explicit frame/frame_id wins if present.
        explicit = []
        explicit_ok = True
        for pos, rec in enumerate(images):
            v = rec.get("frame", rec.get("frame_id"))
            if v is None:
                explicit_ok = False
                break
            try:
                explicit.append((int(v), pos, rec))
            except Exception:
                explicit_ok = False
                break
        if explicit_ok:
            explicit.sort(key=lambda x: (x[0], x[1]))
            images = [x[2] for x in explicit]
        iid_to_frame: Dict[str, int] = {}
        for fi, rec in enumerate(images):
            iid = rec.get("image_id", rec.get("id"))
            if iid is None:
                raise ValueError(f"{sequence_id}: image {fi} has no image id")
            iid_to_frame[str(iid)] = fi
        first = images[0]
        width = int(first.get("width", info.get("width", 0)) or 0)
        height = int(first.get("height", info.get("height", 0)) or 0)
        fps = float(info.get("frame_rate", info.get("fps", 25.0)) or 25.0)
        by_frame: Dict[int, List[GSRAnn]] = {}
        by_track: Dict[str, List[GSRAnn]] = {}
        for ann in d.get("annotations") or []:
            if ann.get("supercategory") not in (None, "object"):
                continue
            attrs = dict(ann.get("attributes") or {})
            role = _role(attrs.get("role", ann.get("role")))
            if role not in {"player", "goalkeeper", "referee"}:
                continue
            iid = str(ann.get("image_id"))
            if iid not in iid_to_frame:
                continue
            box = _bbox_xyxy(ann)
            if box is None:
                continue
            tid = ann.get("track_id", attrs.get("track_id", ann.get("id")))
            if tid is None:
                continue
            rec = GSRAnn(
                frame_index=iid_to_frame[iid], image_id=iid, track_id=str(tid), role=role,
                team=_team(attrs.get("team")), bbox_xyxy=tuple(map(float, box)),
            )
            by_frame.setdefault(rec.frame_index, []).append(rec)
            by_track.setdefault(rec.track_id, []).append(rec)
        seq_dir = lp.parent
        video = None
        for name in ("video.mp4", "video.720p.mp4", f"{sequence_id}.mp4"):
            p = seq_dir / name
            if p.is_file():
                video = p.resolve(); break
        if video is None:
            vids = sorted(seq_dir.glob("*.mp4"))
            video = vids[0].resolve() if vids else None
        return GSRSequence(sequence_id, seq_dir, lp, version, fps, width, height, images, by_frame, by_track, video)

    def inspect(self, max_sequences: int = 5) -> Dict[str, Any]:
        ids = self.discover()
        rows = []
        for sid in ids[:max_sequences]:
            s = self.load(sid)
            roles = s.role_by_track()
            gt = s.team_gt_by_track()
            rows.append({
                "sequence_id": sid,
                "version": s.version,
                "frames": s.num_frames,
                "fps": s.fps,
                "resolution": [s.width, s.height],
                "tracks": len(roles),
                "team_tracks": len(gt),
                "player_tracks": sum(1 for r in roles.values() if r == "player"),
                "goalkeeper_tracks": sum(1 for r in roles.values() if r == "goalkeeper"),
                "referee_tracks": sum(1 for r in roles.values() if r == "referee"),
                "sample_image_resolves": any(s.resolve_image_path(i) is not None for i in range(min(5, s.num_frames))),
                "video_path": str(s.video_path) if s.video_path else None,
            })
        return {"root": str(self.root), "split": self.split, "num_sequences": len(ids), "sequences": rows}


def _bbox_observation(ann: GSRAnn) -> Dict[str, Any]:
    return {"frame_index": ann.frame_index, "source_bbox_xyxy": list(ann.bbox_xyxy), "keypoints_133": []}


def _sample_track_anns(anns: Sequence[GSRAnn], cfg: Stage5Config) -> List[GSRAnn]:
    ordered = sorted(anns, key=lambda a: a.frame_index)[::cfg.sample_every_n_frames]
    if len(ordered) > cfg.max_samples_per_track:
        ids = np.linspace(0, len(ordered)-1, cfg.max_samples_per_track).round().astype(int)
        ordered = [ordered[i] for i in ids]
    return ordered


def _load_pose_cache(path: Optional[Path]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    if path is None or not path.is_file():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("schema_version") == "tracked-pose-2d-state-1.0":
        out: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for t in d.get("tracks") or []:
            tid = str(t.get("track_id"))
            out[tid] = {int(o["frame_index"]): dict(o) for o in t.get("observations") or []}
        return out
    if d.get("schema_version") == "stage5-gsr-pose-cache-1.0":
        out = {}
        for t in d.get("tracks") or []:
            tid = str(t.get("track_id"))
            out[tid] = {int(o["frame_index"]): dict(o) for o in t.get("observations") or []}
        return out
    raise ValueError(f"Unsupported pose cache schema in {path}: {d.get('schema_version')}")


def _pose_cache_path(root: Optional[Path], seq_id: str) -> Optional[Path]:
    if root is None:
        return None
    candidates = [
        root / seq_id / "tracked_pose_2d_state.json",
        root / seq_id / "stage5_gsr_pose_cache.json",
        root / f"{seq_id}.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def predict_sequence_color(
    seq: GSRSequence,
    *,
    config: Stage5Config,
    use_pose: bool = False,
    pose_cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run intrinsic Stage-5 affiliation using GT track/role/bbox.

    Team labels are never read during feature extraction or clustering.  When
    ``use_pose`` is true, a precomputed pose cache can provide pose polygons;
    missing pose observations deliberately fall back to GT bbox regions.
    """
    role_by_track = seq.role_by_track()
    pose_cache = _load_pose_cache(pose_cache_path) if use_pose else {}
    torso_features: Dict[str, np.ndarray] = {}
    lower_features: Dict[str, np.ndarray] = {}
    diagnostics: Dict[str, Any] = {}
    sampled_by_track = {tid: _sample_track_anns(anns, config) for tid, anns in seq.anns_by_track.items()}
    needed_frames = sorted({a.frame_index for items in sampled_by_track.values() for a in items})
    frame_cache = {fi: seq.read_frame(fi) for fi in needed_frames}

    for tid, anns in seq.anns_by_track.items():
        role = role_by_track.get(tid, "")
        if role not in {"player", "goalkeeper", "referee"}:
            continue
        torso_vecs: List[np.ndarray] = []
        lower_vecs: List[np.ndarray] = []
        pose_torso = 0; bbox_torso = 0; pose_lower = 0; bbox_lower = 0; missing_frames = 0
        for ann in sampled_by_track.get(tid, []):
            frame = frame_cache.get(ann.frame_index)
            if frame is None:
                missing_frames += 1
                continue
            obs = dict((pose_cache.get(tid) or {}).get(ann.frame_index) or _bbox_observation(ann))
            obs.setdefault("source_bbox_xyxy", list(ann.bbox_xyxy))
            torso_poly, torso_source = region_polygon(obs, "torso", True)
            lower_poly, lower_source = region_polygon(obs, "lower", True)
            if torso_poly is not None:
                feat, _ = extract_color_feature(frame, torso_poly, config)
                if feat is not None:
                    torso_vecs.append(feat)
                    pose_torso += int(torso_source == "POSE_TORSO")
                    bbox_torso += int(torso_source != "POSE_TORSO")
            if lower_poly is not None:
                feat, _ = extract_color_feature(frame, lower_poly, config)
                if feat is not None:
                    lower_vecs.append(feat)
                    pose_lower += int(lower_source == "POSE_LOWER_BODY")
                    bbox_lower += int(lower_source != "POSE_LOWER_BODY")
        tf = aggregate_features(torso_vecs)
        lf = aggregate_features(lower_vecs)
        if tf is not None:
            torso_features[tid] = tf
        if lf is not None:
            lower_features[tid] = lf
        diagnostics[tid] = {
            "role": role,
            "valid_torso_frames": len(torso_vecs), "valid_lower_body_frames": len(lower_vecs),
            "pose_torso_frames": pose_torso, "bbox_torso_frames": bbox_torso,
            "pose_lower_frames": pose_lower, "bbox_lower_frames": bbox_lower,
            "missing_image_frames": missing_frames,
        }

    fused_features = {
        tid: fuse_region_features(
            torso_features.get(tid), lower_features.get(tid),
            torso_weight=config.torso_feature_weight,
            lower_weight=config.lower_feature_weight,
        ) if config.feature_fusion_enabled else torso_features.get(tid)
        for tid in role_by_track
    }
    outfield = {
        tid: fused_features[tid] for tid, role in role_by_track.items()
        if role == "player" and fused_features.get(tid) is not None
        and diagnostics[tid]["valid_torso_frames"] >= config.min_valid_torso_frames
    }
    pred: Dict[str, Optional[int]] = {}
    status: Dict[str, str] = {}
    clustering: Dict[str, Any]
    if len(outfield) >= 4:
        try:
            cluster = fit_two_teams(outfield, config)
            team_lower = lower_body_centroids_by_team(cluster.labels, cluster.status, lower_features)
            for tid, role in role_by_track.items():
                if role == "referee":
                    pred[tid] = None; status[tid] = "NOT_APPLICABLE"
                elif role == "player":
                    if tid not in cluster.labels or cluster.status.get(tid) != "VALID":
                        pred[tid] = None; status[tid] = "UNKNOWN"
                    else:
                        pred[tid] = int(cluster.labels[tid]); status[tid] = "VALID"
                elif role == "goalkeeper":
                    g = assign_goalkeeper(lower_features.get(tid), team_lower, config)
                    pred[tid] = g.get("team_id"); status[tid] = str(g.get("status"))
            clustering = {
                "status": "VALID",
                "cluster_sizes": cluster.diagnostics.get("cluster_sizes"),
                "inertia": cluster.diagnostics.get("inertia"),
            }
        except ValueError as exc:
            clustering = {"status": "UNAVAILABLE", "reason": str(exc)}
    else:
        clustering = {"status": "UNAVAILABLE", "reason": f"only {len(outfield)} usable outfield tracks"}
    if clustering.get("status") != "VALID":
        for tid, role in role_by_track.items():
            pred[tid] = None
            status[tid] = "NOT_APPLICABLE" if role == "referee" else "UNKNOWN"

    return {
        "sequence_id": seq.sequence_id,
        "method": "STAGE5_POSE_GUIDED_COLOR_KMEANS" if use_pose else "BBOX_COLOR_KMEANS",
        "pose_cache": str(pose_cache_path) if pose_cache_path else None,
        "pred_by_track": pred,
        "status_by_track": status,
        "roles_by_track": role_by_track,
        "diagnostics_by_track": diagnostics,
        "clustering": clustering,
    }


def _best_mapping(gt: Sequence[int], pred: Sequence[int]) -> Dict[int, int]:
    if not gt:
        return {0: 0, 1: 1}
    candidates = ({0:0,1:1},{0:1,1:0})
    return dict(max(candidates, key=lambda m: sum(int(m.get(int(p), -9) == int(g)) for g,p in zip(gt,pred))))


def evaluate_sequence(
    seq: GSRSequence,
    pred_by_track: Mapping[str, Optional[int] | str],
    *,
    predictions_are_gt_labels: bool = False,
) -> Dict[str, Any]:
    roles = seq.role_by_track()
    gt_all = seq.team_gt_by_track()
    gt_out = {t:g for t,g in gt_all.items() if roles.get(t) == "player"}
    gt_gk = {t:g for t,g in gt_all.items() if roles.get(t) == "goalkeeper"}

    # Mapping is derived only from outfield tracks to avoid leaking GK labels.
    valid_out = []
    if predictions_are_gt_labels:
        mapping = {0:0,1:1}
    else:
        for tid, g in gt_out.items():
            p = pred_by_track.get(tid)
            if p in (0,1):
                valid_out.append((g, int(p)))
        mapping = _best_mapping([x[0] for x in valid_out], [x[1] for x in valid_out])

    def canonical_pred(p: Any) -> Optional[int]:
        if isinstance(p, str):
            s = p.strip().lower()
            if s in TEAM_TO_INT:
                return TEAM_TO_INT[s]
            try:
                p = int(s)
            except Exception:
                return None
        if p in (0,1):
            return int(p) if predictions_are_gt_labels else mapping[int(p)]
        return None

    def group_metrics(gt: Mapping[str,int]) -> Dict[str, Any]:
        tids = list(gt)
        cp = {t:canonical_pred(pred_by_track.get(t)) for t in tids}
        valid = [t for t in tids if cp[t] in (0,1)]
        correct_valid = sum(int(cp[t] == gt[t]) for t in valid)
        coverage = len(valid) / max(1, len(tids))
        selective = correct_valid / max(1, len(valid)) if valid else None
        overall = correct_valid / max(1, len(tids))
        y_true = [gt[t] for t in valid]
        y_pred = [int(cp[t]) for t in valid]
        if valid:
            p,r,f1,sup = precision_recall_fscore_support(y_true, y_pred, labels=[0,1], zero_division=0)
            macro_f1 = float(np.mean(f1))
            ari = float(adjusted_rand_score(y_true, y_pred)) if len(set(y_true)) > 1 and len(valid) > 1 else None
            nmi = float(normalized_mutual_info_score(y_true, y_pred)) if len(valid) > 1 else None
            per_team = {str(i): {"precision":float(p[i]),"recall":float(r[i]),"f1":float(f1[i]),"support":int(sup[i])} for i in range(2)}
        else:
            macro_f1 = ari = nmi = None; per_team = {}
        return {
            "tracks": len(tids), "valid_predictions": len(valid), "correct_valid": correct_valid,
            "coverage": float(coverage), "unknown_rate": float(1-coverage),
            "selective_accuracy": None if selective is None else float(selective),
            "overall_accuracy": float(overall), "macro_f1": macro_f1,
            "ARI": ari, "NMI": nmi, "per_team": per_team,
        }

    refs = [t for t,r in roles.items() if r == "referee"]
    ref_assigned = sum(canonical_pred(pred_by_track.get(t)) in (0,1) for t in refs)
    return {
        "sequence_id": seq.sequence_id,
        "mapping_pred_to_gt": {str(k):int(v) for k,v in mapping.items()},
        "all_team_tracks": group_metrics(gt_all),
        "outfield": group_metrics(gt_out),
        "goalkeeper": group_metrics(gt_gk),
        "referee_tracks": len(refs),
        "referee_team_assignments": int(ref_assigned),
        "referee_team_contamination_rate": float(ref_assigned / max(1, len(refs))) if refs else 0.0,
    }


def load_external_predictions(path: Path) -> Tuple[Dict[str, Any], bool]:
    """Load a third-party sequence prediction file.

    Accepted forms:
      * {"track_id": 0/1/"left"/"right"}
      * {"track_team": {track_id: {team_id/team/...}}}
      * COCO-like {"annotations": [... attributes.team ...]}

    Returns (track->prediction, predictions_are_gt_labels).
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, dict) and "annotations" in d:
        votes: Dict[str, Dict[str,int]] = {}
        for a in d.get("annotations") or []:
            attrs = a.get("attributes") or {}
            t = _team(attrs.get("team"))
            tid = a.get("track_id")
            if tid is None or t is None:
                continue
            votes.setdefault(str(tid), {})[t] = votes.setdefault(str(tid), {}).get(t,0)+1
        return {tid:max(c,key=c.get) for tid,c in votes.items()}, True
    if isinstance(d, dict) and "track_team" in d:
        out = {}
        direct = False
        for tid, rec in (d.get("track_team") or {}).items():
            if isinstance(rec, Mapping):
                v = rec.get("team", rec.get("team_id"))
            else:
                v = rec
            out[str(tid)] = v
            if isinstance(v,str) and v.lower() in TEAM_TO_INT:
                direct = True
        return out, direct
    if isinstance(d, dict):
        direct = any(isinstance(v,str) and v.lower() in TEAM_TO_INT for v in d.values())
        return {str(k):v for k,v in d.items()}, direct
    raise ValueError(f"Unsupported prediction format: {path}")


def find_external_prediction(root: Path, sequence_id: str) -> Optional[Path]:
    candidates = [
        root / sequence_id / "Labels-GameState.json",
        root / sequence_id / "team_predictions.json",
        root / f"{sequence_id}.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _mean_defined(values: Iterable[Optional[float]]) -> Optional[float]:
    a = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(a)) if a else None


def _bootstrap_ci(sequence_values: Sequence[float], seed: int = 23, samples: int = 2000) -> Optional[List[float]]:
    vals = np.asarray([x for x in sequence_values if math.isfinite(float(x))], dtype=float)
    if vals.size == 0:
        return None
    if vals.size == 1:
        return [float(vals[0]), float(vals[0])]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        means[i] = float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate_benchmark(sequence_rows: Sequence[Mapping[str, Any]], method: str, split: str) -> Dict[str, Any]:
    if not sequence_rows:
        return {"method": method, "split": split, "num_sequences": 0}
    groups = ("all_team_tracks", "outfield", "goalkeeper")
    summary: Dict[str, Any] = {"method": method, "split": split, "num_sequences": len(sequence_rows), "groups": {}}
    for g in groups:
        total = sum(int(r[g]["tracks"]) for r in sequence_rows)
        valid = sum(int(r[g]["valid_predictions"]) for r in sequence_rows)
        correct = sum(int(r[g]["correct_valid"]) for r in sequence_rows)
        micro_cov = valid / max(1,total)
        micro_sel = correct / max(1,valid) if valid else None
        micro_overall = correct / max(1,total)
        seq_acc = [float(r[g]["overall_accuracy"]) for r in sequence_rows]
        seq_sel = [r[g]["selective_accuracy"] for r in sequence_rows if r[g]["selective_accuracy"] is not None]
        summary["groups"][g] = {
            "tracks": total, "valid_predictions": valid, "correct_valid": correct,
            "micro_coverage": float(micro_cov),
            "micro_selective_accuracy": None if micro_sel is None else float(micro_sel),
            "micro_overall_accuracy": float(micro_overall),
            "macro_overall_accuracy": _mean_defined(seq_acc),
            "macro_selective_accuracy": _mean_defined(seq_sel),
            "macro_f1": _mean_defined(r[g].get("macro_f1") for r in sequence_rows),
            "macro_ARI": _mean_defined(r[g].get("ARI") for r in sequence_rows),
            "macro_NMI": _mean_defined(r[g].get("NMI") for r in sequence_rows),
            "macro_overall_accuracy_95ci": _bootstrap_ci(seq_acc),
        }
    refs = sum(int(r.get("referee_tracks",0)) for r in sequence_rows)
    contam = sum(int(r.get("referee_team_assignments",0)) for r in sequence_rows)
    summary["referee"] = {
        "tracks": refs,
        "team_assignments": contam,
        "team_contamination_rate": float(contam / max(1,refs)) if refs else 0.0,
    }
    return summary


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# Stage 5 SoccerNet-GSR Benchmark — {summary.get('method')}", "",
        f"Split: `{summary.get('split')}`  ", f"Sequences: **{summary.get('num_sequences')}**", "",
        "| Group | Tracks | Coverage | Selective accuracy | Overall accuracy | Macro-F1 | Macro ARI | Macro NMI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g, m in (summary.get("groups") or {}).items():
        pct = lambda v: "—" if v is None else f"{100*float(v):.2f}%"
        num = lambda v: "—" if v is None else f"{float(v):.4f}"
        lines.append(f"| {g} | {m['tracks']} | {pct(m['micro_coverage'])} | {pct(m['micro_selective_accuracy'])} | {pct(m['micro_overall_accuracy'])} | {num(m['macro_f1'])} | {num(m['macro_ARI'])} | {num(m['macro_NMI'])} |")
    r = summary.get("referee") or {}
    lines += ["", f"Referee contamination: **{r.get('team_assignments',0)}/{r.get('tracks',0)}** ({100*float(r.get('team_contamination_rate',0)):.2f}%).", ""]
    ci = (((summary.get("groups") or {}).get("all_team_tracks") or {}).get("macro_overall_accuracy_95ci"))
    if ci:
        lines.append(f"Sequence-bootstrap 95% CI for macro overall accuracy: **[{100*ci[0]:.2f}%, {100*ci[1]:.2f}%]**.")
    lines += ["", "> Cluster IDs are aligned to SoccerNet `left/right` per sequence using outfield tracks only. UNKNOWN predictions count as wrong in overall accuracy and are excluded from selective accuracy.", ""]
    return "\n".join(lines)


def run_benchmark(
    *,
    dataset_root: str | Path,
    split: str,
    output_dir: str | Path,
    method: str = "bbox-color",
    pose_cache_root: str | Path | None = None,
    external_predictions_root: str | Path | None = None,
    limit: Optional[int] = None,
    config: Optional[Stage5Config] = None,
    progress_every: int = 5,
) -> Dict[str, Any]:
    cfg = config or Stage5Config()
    cfg.validate()
    ds = SoccerNetGSRDataset(dataset_root, split)
    ids = ds.discover()
    if limit is not None:
        ids = ids[:int(limit)]
    out = Path(output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    pose_root = Path(pose_cache_root).expanduser().resolve() if pose_cache_root else None
    ext_root = Path(external_predictions_root).expanduser().resolve() if external_predictions_root else None
    rows = []
    method_name = method
    for idx, sid in enumerate(ids, 1):
        seq = ds.load(sid)
        if method == "bbox-color":
            pred_payload = predict_sequence_color(seq, config=cfg, use_pose=False)
            pred = pred_payload["pred_by_track"]
            direct = False
        elif method == "stage5-color":
            if pose_root is None:
                raise ValueError("stage5-color requires --pose-cache-root so the benchmark cannot silently collapse to bbox-only crops")
            pp = _pose_cache_path(pose_root, sid)
            if pp is None:
                raise FileNotFoundError(f"{sid}: no pose cache found under {pose_root}")
            pred_payload = predict_sequence_color(seq, config=cfg, use_pose=True, pose_cache_path=pp)
            pred = pred_payload["pred_by_track"]
            direct = False
        elif method == "external":
            if ext_root is None:
                raise ValueError("external method requires --external-predictions-root")
            ep = find_external_prediction(ext_root, sid)
            if ep is None:
                pred = {}; direct = True
                pred_payload = {"sequence_id":sid,"method":"EXTERNAL_MISSING","prediction_file":None,"pred_by_track":{}}
            else:
                pred, direct = load_external_predictions(ep)
                pred_payload = {"sequence_id":sid,"method":"EXTERNAL","prediction_file":str(ep),"pred_by_track":pred,"predictions_are_gt_labels":direct}
        else:
            raise ValueError(f"Unknown method {method!r}")
        metrics = evaluate_sequence(seq, pred, predictions_are_gt_labels=direct)
        metrics["method"] = pred_payload.get("method", method)
        metrics["labels_path"] = str(seq.labels_path)
        rows.append(metrics)
        _write_json(out / "sequences" / sid / "prediction.json", pred_payload)
        _write_json(out / "sequences" / sid / "metrics.json", metrics)
        if progress_every and (idx % progress_every == 0 or idx == len(ids)):
            print(f"[{idx}/{len(ids)}] {sid}: all overall={metrics['all_team_tracks']['overall_accuracy']:.4f} coverage={metrics['all_team_tracks']['coverage']:.4f}")
    summary = aggregate_benchmark(rows, method_name, split)
    summary["dataset_root"] = str(Path(dataset_root).expanduser().resolve())
    summary["configuration"] = cfg.to_dict()
    summary["protocol"] = {
        "schema_version": "stage5-soccernet-gsr-benchmark-1.0",
        "unit": "track",
        "gt_inputs": ["bbox_image", "track_id", "role"],
        "gt_team_hidden_during_inference": True,
        "mapping": "per-sequence outfield-only binary Hungarian/permutation alignment",
        "unknown_policy": "wrong for overall accuracy; excluded from selective accuracy",
        "goalkeeper_mapping": "reuse outfield team mapping",
        "bootstrap": "sequence-level, deterministic seed 23",
    }
    _write_json(out / "benchmark_summary.json", summary)
    with (out / "sequence_metrics.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "benchmark_summary.md").write_text(_markdown_summary(summary), encoding="utf-8")
    return summary
