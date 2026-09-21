from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import shutil

from .common import atomic_json, load_json, record_key, stable_hash


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RecordCheckpointStore:
    root: Path
    schema_version: str
    run_identity: dict[str, Any]
    total_units: int
    result_subdir: str = "rows"

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.result_dir = self.root / self.result_subdir
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "checkpoint.json"
        self.run_signature = stable_hash(self.run_identity)
        if self.state_path.is_file():
            state = load_json(self.state_path)
            if str(state.get("run_signature")) != self.run_signature:
                raise RuntimeError(
                    "Checkpoint configuration differs from the requested run. "
                    "Use a new --output-dir or explicitly delete/restart the old benchmark directory."
                )
            self.state = state
        else:
            self.state = {
                "schema_version": self.schema_version,
                "run_signature": self.run_signature,
                "run_identity": self.run_identity,
                "status": "NOT_STARTED",
                "created_at": _now(),
                "updated_at": _now(),
                "total_units": int(self.total_units),
                "completed_units": 0,
                "completion_fraction": 0.0,
                "last_completed_record_id": None,
                "last_error": None,
            }
            self._save()
        self.rescan()

    def _save(self) -> None:
        self.state["updated_at"] = _now()
        atomic_json(self.state_path, self.state)

    def result_path(self, record_id: str) -> Path:
        return self.result_dir / f"{record_key(record_id)}.json"

    def load_result(self, record_id: str) -> dict[str, Any] | None:
        p = self.result_path(record_id)
        if not p.is_file():
            return None
        payload = load_json(p)
        if str(payload.get("record_id")) != str(record_id):
            raise RuntimeError(f"Checkpoint hash collision or stale row at {p}")
        if str(payload.get("run_signature", self.run_signature)) != self.run_signature:
            raise RuntimeError(f"Result checkpoint at {p} belongs to another run")
        return payload

    def commit(self, record_id: str, payload: Mapping[str, Any]) -> Path:
        body = dict(payload)
        body["record_id"] = str(record_id)
        body["run_signature"] = self.run_signature
        path = atomic_json(self.result_path(record_id), body)
        self.state["last_completed_record_id"] = str(record_id)
        self.state["last_error"] = None
        self.rescan(save=False)
        self._save()
        return path

    def rescan(self, *, save: bool = True) -> dict[str, Any]:
        completed = sum(1 for p in self.result_dir.glob("*.json") if p.is_file())
        self.state["completed_units"] = int(completed)
        self.state["total_units"] = int(self.total_units)
        self.state["completion_fraction"] = completed / max(1, int(self.total_units))
        if save:
            self._save()
        return dict(self.state)

    def mark_running(self) -> None:
        self.state["status"] = "RUNNING"
        self.state["last_error"] = None
        self._save()

    def mark_complete(self) -> None:
        self.rescan(save=False)
        self.state["status"] = "COMPLETE"
        self.state["completion_fraction"] = self.state["completed_units"] / max(1, self.total_units)
        self._save()

    def mark_interrupted(self, message: str = "KeyboardInterrupt") -> None:
        self.rescan(save=False)
        self.state["status"] = "INTERRUPTED"
        self.state["last_error"] = str(message)
        self._save()

    def mark_failed(self, error: BaseException | str) -> None:
        self.rescan(save=False)
        self.state["status"] = "FAILED"
        self.state["last_error"] = str(error)
        self._save()

    def status(self) -> dict[str, Any]:
        return self.rescan()


def clear_checkpoint_outputs(root: str | Path, *, keep: tuple[str, ...] = ()) -> None:
    root = Path(root)
    if not root.exists():
        return
    keep_set = set(keep)
    for child in root.iterdir():
        if child.name in keep_set:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
