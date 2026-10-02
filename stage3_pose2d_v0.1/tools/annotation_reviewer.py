#!/usr/bin/env python3
"""Run a local, Stage-3-only WholeBody133 annotation reviewer.

The reviewer reads a task manifest and an optional RTMW preannotation artifact,
then writes approved human edits to a separate output manifest. It never
changes the source images, Stage-2 artifacts or input manifest in place.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import mimetypes
import sys
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = Path(__file__).resolve().with_name("annotation_reviewer.html")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage3_pose2d.wholebody133 import WHOLEBODY_KEYPOINT_NAMES


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _normalise_points(points: Any) -> list[Dict[str, Any]]:
    records = points if isinstance(points, list) else []
    output = []
    for index, name in enumerate(WHOLEBODY_KEYPOINT_NAMES):
        source = records[index] if index < len(records) and isinstance(records[index], dict) else {}
        x = source.get("x")
        y = source.get("y")
        visibility = source.get("visibility")
        output.append({
            "index": index,
            "name": name,
            "x": float(x) if _finite(x) else None,
            "y": float(y) if _finite(y) else None,
            "visibility": int(visibility) if visibility in (0, 1, 2) else None,
        })
    return output


class ReviewStore:
    def __init__(self, manifest: Path, output: Path, preannotations: Path | None) -> None:
        self.manifest = manifest.resolve()
        self.output = output.resolve()
        self.lock = threading.RLock()
        self.document = json.loads(self.manifest.read_text(encoding="utf-8"))
        if self.document.get("pose_schema", {}).get("name") != "COCO_WHOLEBODY_133":
            raise ValueError("manifest must use COCO_WHOLEBODY_133")
        self._merge_preannotations(preannotations)
        self._write()

    def _merge_preannotations(self, preannotations: Path | None) -> None:
        if not preannotations:
            return
        source = json.loads(preannotations.resolve().read_text(encoding="utf-8"))
        by_id = {str(task.get("task_id")): task for task in source.get("tasks") or []}
        for task in self.document.get("tasks") or []:
            other = by_id.get(str(task.get("task_id")))
            if not other:
                continue
            for key in ("model_preannotation_133", "model_preannotation_qa", "model_preannotation_hard_negative_flags", "model_preannotation_provenance"):
                if key in other:
                    task[key] = copy.deepcopy(other[key])

    def _write(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.document, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.output)

    def summaries(self) -> list[Dict[str, Any]]:
        with self.lock:
            return [{
                "task_id": task.get("task_id"),
                "split": task.get("split"),
                "sequence_id": task.get("sequence_id"),
                "image_id": task.get("image_id"),
                "role": task.get("role"),
                "review_status": task.get("review_status", "PENDING"),
                "has_preannotation": bool(task.get("model_preannotation_133")),
            } for task in self.document.get("tasks") or []]

    def task(self, task_id: str) -> Dict[str, Any] | None:
        with self.lock:
            for task in self.document.get("tasks") or []:
                if str(task.get("task_id")) == task_id:
                    result = copy.deepcopy(task)
                    result["keypoint_names"] = list(WHOLEBODY_KEYPOINT_NAMES)
                    return result
        return None

    def save(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_id = str(payload.get("task_id", ""))
        points = _normalise_points(payload.get("keypoints_133"))
        status = str(payload.get("review_status", "PENDING")).upper()
        if status not in {"PENDING", "APPROVED", "REJECTED"}:
            raise ValueError("review_status must be PENDING, APPROVED or REJECTED")
        reviewer_id = str(payload.get("reviewer_id", "")).strip()
        notes = str(payload.get("reviewer_notes", "")).strip()
        complete = all(_finite(point["x"]) and _finite(point["y"]) and point["visibility"] in (0, 1, 2) for point in points)
        if status in {"APPROVED", "REJECTED"} and (not reviewer_id or not complete):
            raise ValueError("final review requires reviewer_id and all 133 coordinates/visibility values")
        with self.lock:
            target = next((task for task in self.document.get("tasks") or [] if str(task.get("task_id")) == task_id), None)
            if target is None:
                raise KeyError(f"unknown task_id: {task_id}")
            target["keypoints_133"] = points
            target["review_status"] = status
            target["annotation_source"] = "HUMAN_VERIFIED" if status in {"APPROVED", "REJECTED"} else "UNANNOTATED"
            target["reviewer_id"] = reviewer_id or None
            target["reviewed_at"] = datetime.now(timezone.utc).isoformat() if status in {"APPROVED", "REJECTED"} else None
            target["reviewer_notes"] = notes
            target["review_provenance"] = {
                "tool": "stage3_annotation_reviewer",
                "source_manifest": str(self.manifest),
                "preannotations_are_not_ground_truth": True,
            }
            self.document["status"] = "ANNOTATION_IN_PROGRESS"
            self.document["train_ready"] = False
            self._write()
            return {"task_id": task_id, "review_status": status, "complete": complete, "output": str(self.output)}


class Handler(BaseHTTPRequestHandler):
    store: ReviewStore

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send(HTTPStatus.OK, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/tasks":
            self._json(HTTPStatus.OK, {"tasks": self.store.summaries()})
            return
        if parsed.path == "/api/task":
            task_id = parse_qs(parsed.query).get("task_id", [""])[0]
            task = self.store.task(task_id)
            self._json(HTTPStatus.OK if task else HTTPStatus.NOT_FOUND, task or {"error": "Unknown task."})
            return
        if parsed.path == "/api/image":
            task_id = parse_qs(parsed.query).get("task_id", [""])[0]
            task = self.store.task(task_id)
            if not task:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown task."})
                return
            path = Path(str(task.get("image_path", ""))).expanduser().resolve()
            if not path.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "Image not found."})
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, path.read_bytes(), content_type)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/save":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self._json(HTTPStatus.OK, self.store.save(payload))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage-3 WholeBody133 annotation reviewer")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preannotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    store = ReviewStore(args.manifest, args.output, args.preannotations)
    Handler.store = store
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Stage-3 annotation reviewer: http://{args.host}:{args.port}")
    print(f"Writing reviewed manifest: {store.output}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation reviewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
