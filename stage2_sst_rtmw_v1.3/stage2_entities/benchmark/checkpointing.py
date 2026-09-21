from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import hashlib
import json
import os
import shutil
import time


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Crash-resistant JSON replacement on the same filesystem.

    A power loss can still lose the newest checkpoint, but it should not leave a half-written
    canonical JSON file. The previous complete file remains until os.replace succeeds.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, p)
    return p


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--:--"
    sec = int(round(seconds))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe_component(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


@dataclass
class ConsoleProgress:
    total_eval_units: int
    enabled: bool = True
    perception_every: int = 10
    started: float = 0.0
    completed_eval_units: int = 0
    resumed_eval_units: int = 0
    initial_completed_eval_units: int = 0

    def __post_init__(self) -> None:
        self.started = time.perf_counter()
        self.perception_every = max(1, int(self.perception_every))

    def _stamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def log(self, message: str) -> None:
        if self.enabled:
            print(f"[{self._stamp()}] {message}", flush=True)

    def set_initial_completed(self, completed: int) -> None:
        self.completed_eval_units = int(completed)
        self.initial_completed_eval_units = int(completed)
        self.resumed_eval_units = int(completed)

    def eval_done(self, label: str, *, resumed: bool = False, metric_text: str = "") -> None:
        self.completed_eval_units += 1
        if resumed:
            self.resumed_eval_units += 1
        elapsed = max(1e-9, time.perf_counter() - self.started)
        done = self.completed_eval_units
        total = max(1, self.total_eval_units)
        new_done = max(0, done - self.initial_completed_eval_units)
        rate = new_done / elapsed if new_done > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else None
        pct = 100.0 * done / total
        tag = "RESUME" if resumed else "CHECKPOINT"
        suffix = f" | {metric_text}" if metric_text else ""
        self.log(
            f"[EVAL {done}/{total} {pct:5.1f}%] {label}{suffix} | "
            f"elapsed {format_duration(elapsed)} ETA {format_duration(eta)} | {tag}"
        )

    def perception(self, *, sequence_id: str, index: int, total: int, frame_index: int,
                   reused: bool, humans: int, poses: int, balls: int, elapsed: float) -> None:
        if not self.enabled:
            return
        if index != 1 and index != total and index % self.perception_every != 0:
            return
        rate = index / max(1e-9, elapsed)
        eta = (total - index) / rate if rate > 0 else None
        status = "reuse" if reused else "infer"
        self.log(
            f"[PERCEPTION {sequence_id} {index}/{total} {100.0*index/max(1,total):5.1f}%] "
            f"frame={frame_index} {status} humans={humans} pose={poses} ball={balls} | "
            f"elapsed {format_duration(elapsed)} ETA {format_duration(eta)}"
        )


class BenchmarkCheckpointStore:
    """Fine-grained, atomic benchmark evaluation checkpoints.

    Perception caches live outside this directory. This store only owns benchmark/evaluation
    state, so --restart-evaluation never deletes expensive SST/RTMW perception outputs.
    """

    SCHEMA = "stage2-benchmark-checkpoint-1.0"

    def __init__(self, output_dir: str | Path, run_identity: Mapping[str, Any], *,
                 total_windows: int, experiments: tuple[str, ...], restart: bool = False) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.root = self.output_dir / "checkpoints"
        if restart and self.root.exists():
            shutil.rmtree(self.root)
        self.eval_root = self.root / "evaluation"
        self.state_path = self.root / "benchmark_state.json"
        self.run_identity = dict(run_identity)
        self.run_signature = stable_json_hash(self.run_identity)
        self.total_windows = int(total_windows)
        self.experiments = tuple(experiments)
        self.total_units = self.total_windows * (1 + len(self.experiments))
        self.root.mkdir(parents=True, exist_ok=True)
        self.eval_root.mkdir(parents=True, exist_ok=True)

        if self.state_path.is_file():
            state = load_json(self.state_path)
            old = str(state.get("run_signature", ""))
            if old != self.run_signature:
                raise RuntimeError(
                    "Existing benchmark evaluation checkpoints belong to a different configuration. "
                    "Use a new --output-dir or pass --restart-evaluation to discard evaluation checkpoints "
                    "while preserving perception caches."
                )
            self.state = state
        else:
            self.state = {
                "schema_version": self.SCHEMA,
                "run_signature": self.run_signature,
                "run_identity": self.run_identity,
                "status": "RUNNING",
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
                "total_windows": self.total_windows,
                "experiments": list(self.experiments),
                "total_eval_units": self.total_units,
                "model_fingerprints": {},
                "last_completed": None,
                "last_error": None,
            }
            self._sync_state()
        self._refresh_counts()

    def _window_dir(self, sequence_id: str, target_frame: int) -> Path:
        return self.eval_root / _safe_component(sequence_id) / f"t{int(target_frame):09d}"

    def detection_path(self, sequence_id: str, target_frame: int) -> Path:
        return self._window_dir(sequence_id, target_frame) / "detection.json"

    def tracking_path(self, sequence_id: str, target_frame: int, experiment: str) -> Path:
        return self._window_dir(sequence_id, target_frame) / f"tracking_{_safe_component(experiment)}.json"

    def _valid_loaded(self, path: Path, kind: str, sequence_id: str, target_frame: int,
                      experiment: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not path.is_file():
            return None
        try:
            data = load_json(path)
        except Exception:
            return None
        if str(data.get("run_signature")) != self.run_signature:
            return None
        if str(data.get("kind")) != kind:
            return None
        if str(data.get("sequence_id")) != str(sequence_id) or int(data.get("target_frame", -1)) != int(target_frame):
            return None
        if experiment is not None and str(data.get("experiment")) != str(experiment):
            return None
        return data

    def load_detection(self, sequence_id: str, target_frame: int) -> Optional[Dict[str, Any]]:
        return self._valid_loaded(self.detection_path(sequence_id, target_frame), "detection", sequence_id, target_frame)

    def load_tracking(self, sequence_id: str, target_frame: int, experiment: str) -> Optional[Dict[str, Any]]:
        return self._valid_loaded(self.tracking_path(sequence_id, target_frame, experiment), "tracking", sequence_id, target_frame, experiment)

    def save_detection(self, sequence_id: str, target_frame: int, payload: Mapping[str, Any]) -> Path:
        record = {
            "schema_version": self.SCHEMA,
            "run_signature": self.run_signature,
            "kind": "detection",
            "sequence_id": str(sequence_id),
            "target_frame": int(target_frame),
            "saved_at": utc_now_iso(),
            **dict(payload),
        }
        path_obj = self.detection_path(sequence_id, target_frame)
        existed = path_obj.is_file()
        path = atomic_write_json(path_obj, record)
        self.state["last_completed"] = {"kind": "detection", "sequence_id": str(sequence_id), "target_frame": int(target_frame)}
        if not existed:
            self.state["completed_detection_windows"] = int(self.state.get("completed_detection_windows", 0)) + 1
            self.state["completed_eval_units"] = int(self.state.get("completed_eval_units", 0)) + 1
        self._sync_progress_fields()
        return path

    def save_tracking(self, sequence_id: str, target_frame: int, experiment: str, payload: Mapping[str, Any]) -> Path:
        record = {
            "schema_version": self.SCHEMA,
            "run_signature": self.run_signature,
            "kind": "tracking",
            "sequence_id": str(sequence_id),
            "target_frame": int(target_frame),
            "experiment": str(experiment),
            "saved_at": utc_now_iso(),
            **dict(payload),
        }
        path_obj = self.tracking_path(sequence_id, target_frame, experiment)
        existed = path_obj.is_file()
        path = atomic_write_json(path_obj, record)
        self.state["last_completed"] = {
            "kind": "tracking", "experiment": str(experiment),
            "sequence_id": str(sequence_id), "target_frame": int(target_frame),
        }
        if not existed:
            by_exp = dict(self.state.get("completed_tracking_windows_by_experiment") or {})
            by_exp[str(experiment)] = int(by_exp.get(str(experiment), 0)) + 1
            self.state["completed_tracking_windows_by_experiment"] = by_exp
            self.state["completed_eval_units"] = int(self.state.get("completed_eval_units", 0)) + 1
        self._sync_progress_fields()
        return path

    def iter_detection(self):
        for p in sorted(self.eval_root.glob("*/t*/detection.json")):
            try:
                d = load_json(p)
            except Exception:
                continue
            if d.get("run_signature") == self.run_signature:
                yield d

    def iter_tracking(self, experiment: Optional[str] = None):
        pattern = "*/t*/tracking_*.json" if experiment is None else f"*/t*/tracking_{_safe_component(experiment)}.json"
        for p in sorted(self.eval_root.glob(pattern)):
            try:
                d = load_json(p)
            except Exception:
                continue
            if d.get("run_signature") == self.run_signature:
                yield d

    def _sync_progress_fields(self) -> None:
        completed_units = int(self.state.get("completed_eval_units", 0))
        self.state["updated_at"] = utc_now_iso()
        self.state["completion_fraction"] = completed_units / max(1, self.total_units)
        self._sync_state()

    def _refresh_counts(self) -> None:
        detection_count = sum(1 for _ in self.iter_detection())
        by_exp = {e: sum(1 for _ in self.iter_tracking(e)) for e in self.experiments}
        completed_units = detection_count + sum(by_exp.values())
        self.state.update({
            "status": self.state.get("status", "RUNNING"),
            "completed_detection_windows": detection_count,
            "completed_tracking_windows_by_experiment": by_exp,
            "completed_eval_units": completed_units,
        })
        self._sync_progress_fields()

    def _sync_state(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def assert_or_set_model_fingerprints(self, fingerprints: Mapping[str, str]) -> None:
        current = dict(self.state.get("model_fingerprints") or {})
        for key, value in fingerprints.items():
            if not value:
                continue
            if current.get(key) and current[key].lower() != str(value).lower():
                raise RuntimeError(
                    f"Benchmark checkpoint model fingerprint mismatch for {key}. "
                    "Use a new --output-dir or --restart-evaluation only after also selecting compatible perception caches."
                )
            current[key] = str(value)
        self.state["model_fingerprints"] = current
        self._sync_state()

    def mark_status(self, status: str, *, error: Optional[str] = None) -> None:
        self.state["status"] = str(status)
        self.state["updated_at"] = utc_now_iso()
        self.state["last_error"] = error
        self._refresh_counts()

    def status(self, *, rescan: bool = False) -> Dict[str, Any]:
        if rescan:
            self._refresh_counts()
        return dict(self.state)


def read_checkpoint_status(output_dir: str | Path) -> Dict[str, Any]:
    root = Path(output_dir).resolve() / "checkpoints"
    path = root / "benchmark_state.json"
    if not path.is_file():
        return {"status": "NOT_STARTED", "checkpoint": str(path)}
    data = load_json(path)
    sig = str(data.get("run_signature", ""))
    eval_root = root / "evaluation"
    detection_count = 0
    tracking_by_exp: Dict[str, int] = {str(e): 0 for e in data.get("experiments", [])}
    for p in eval_root.glob("*/t*/detection.json") if eval_root.is_dir() else []:
        try:
            d = load_json(p)
            if str(d.get("run_signature", "")) == sig:
                detection_count += 1
        except Exception:
            pass
    for p in eval_root.glob("*/t*/tracking_*.json") if eval_root.is_dir() else []:
        try:
            d = load_json(p)
            if str(d.get("run_signature", "")) != sig:
                continue
            exp = str(d.get("experiment", "unknown"))
            tracking_by_exp[exp] = tracking_by_exp.get(exp, 0) + 1
        except Exception:
            pass
    completed = detection_count + sum(tracking_by_exp.values())
    total = int(data.get("total_eval_units", 0) or 0)
    data["completed_detection_windows"] = detection_count
    data["completed_tracking_windows_by_experiment"] = tracking_by_exp
    data["completed_eval_units"] = completed
    data["completion_fraction"] = completed / max(1, total)
    data["checkpoint"] = str(path)
    return data
