from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse
import json
import mimetypes
from datetime import datetime, timezone
import uuid
import threading
import time
import os
from collections import deque

from .adapters import load_json, replay_context
from .core import build_offside_position_state
from .visualization import encode_jpeg, load_frame_with_source, render_overlay


@dataclass
class DemoContext:
    stage4: Dict[str, Any]
    stage7: Dict[str, Any]
    stage8: Dict[str, Any]
    state: Dict[str, Any]
    frame: Any
    stage1: Optional[Dict[str, Any]] = None
    stage3: Optional[Dict[str, Any]] = None
    frame_source: Optional[Dict[str, Any]] = None


def build_demo_context(
    *,
    stage4_path: str,
    stage7_path: str,
    stage8_path: Optional[str] = None,
    stage6_path: Optional[str] = None,
    stage1_path: Optional[str] = None,
    stage3_path: Optional[str] = None,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    epsilon_m: float = 1e-9,
) -> DemoContext:
    stage4 = load_json(stage4_path)
    stage7 = load_json(stage7_path)
    stage8 = load_json(stage8_path) if stage8_path else {}
    stage6 = load_json(stage6_path) if stage6_path else {}
    stage1 = load_json(stage1_path) if stage1_path else None
    stage3 = load_json(stage3_path) if stage3_path else None
    state = build_offside_position_state(stage4, stage7, stage8, stage6_input=stage6, epsilon_m=epsilon_m, best_effort=True).to_dict()

    inferred_video = video_path
    inferred_from_stage3 = False
    if inferred_video is None and image_path is None and stage3:
        replay = replay_context(stage3)
        candidate = replay.get("video_path") or replay.get("video")
        if candidate and Path(str(candidate)).is_file():
            inferred_video = str(candidate)
            inferred_from_stage3 = True

    frame, frame_source = load_frame_with_source(
        image_path=image_path,
        video_path=inferred_video,
        frame_index=int(state.get("frame_index") or 0),
        fallback_size=(1280, 720),
    )
    if inferred_from_stage3 and frame_source.get("kind") == "VIDEO":
        frame_source["origin"] = "stage3.replay_context.video_path"
        frame_source["warning"] = "INFERRED_SOURCE_MAY_ALREADY_CONTAIN_UPSTREAM_OVERLAYS"
    elif image_path:
        frame_source["origin"] = "explicit --image"
    elif video_path:
        frame_source["origin"] = "explicit --video"
    else:
        frame_source.setdefault("origin", "fallback")
    state["visualization"] = {
        "screen_geometry_policy": "NO_SYNTHETIC_COORDINATES",
        "default_defenders": "LAST_AND_SECOND_LAST_ONLY",
        "default_keypoints": False,
        "frame_source": frame_source,
    }
    return DemoContext(stage4=stage4, stage7=stage7, stage8=stage8, stage1=stage1, stage3=stage3, state=state, frame=frame, frame_source=frame_source)


def render_frame_bytes(ctx: DemoContext, query: Dict[str, list[str]] | None = None) -> bytes:
    if ctx.state.get("mode") == "UPSTREAM_1_7":
        from .pipeline import render_pipeline_frame
        return render_pipeline_frame(ctx, query)
    query = query or {}
    def flag(name: str, default: bool = True) -> bool:
        raw = query.get(name, [None])[0]
        if raw is None:
            return default
        return str(raw).lower() not in {"0", "false", "off", "no"}
    if not flag("overlay", True):
        img = ctx.frame
    else:
        img = render_overlay(
            ctx.frame,
            ctx.state,
            stage4=ctx.stage4,
            stage3=ctx.stage3,
            stage1=ctx.stage1,
            show_reference=flag("reference", True),
            show_defenders=flag("defenders", True),
            show_skeleton=flag("skeleton", False),
            show_labels=flag("labels", True),
            show_all_defenders=flag("all_defenders", False),
        )
    return encode_jpeg(img)


def make_handler(ctx: DemoContext):
    root = Path(__file__).resolve().parent.parent
    template = root / "templates" / ("pipeline.html" if ctx.state.get("mode") == "UPSTREAM_1_7" else "index.html")
    static_root = root / "static"
    video_root = root / "outputs" / "video_sources"
    progress = {}
    progress_lock = threading.Lock()
    analysis_lock = threading.Lock()

    def update_progress(key, **values):
        with progress_lock:
            if key in progress:
                progress[key].update(values)

    class Handler(BaseHTTPRequestHandler):
        server_version = "Stage9Demo/0.1"

        def _send(self, status: int, body: bytes, content_type: str, cache: str = "no-store"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/api/analysis_progress":
                with progress_lock:
                    record = progress.get(query.get("request_id", [""])[0])
                    body = json.dumps(record or {"status": "pending"}, ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            if path == "/api/video_source":
                try:
                    from .video_selection import validate_upload
                    validate_upload(int(query.get("size", ["0"])[0]), video_root)
                    self._send(200, b'{"ready":true}', "application/json")
                except (ValueError, OSError) as exc:
                    self._send(400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                return
            if path == "/":
                self._send(200, template.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                body = json.dumps(ctx.state, ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            if path == "/api/health":
                body = json.dumps({"ok": True, "stage": 9, "mode": ctx.state.get("mode"), "frame_index": ctx.state.get("frame_index"), "frame_source": ctx.frame_source}).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            if path == "/api/frame.jpg":
                self._send(200, render_frame_bytes(ctx, query), "image/jpeg")
                return
            if path == "/api/video_frame.jpg":
                try:
                    from .video_selection import read_video_frame
                    frame = read_video_frame(video_root, query.get("video_id", [""])[0], query.get("frame", ["0"])[0])
                    if query.get("thumb", ["0"])[0] == "1":
                        import cv2
                        h, w = frame.shape[:2]
                        frame = cv2.resize(frame, (160, max(1, round(h * 160 / w))))
                    self._send(200, encode_jpeg(frame), "image/jpeg")
                except (ValueError, TypeError) as exc:
                    self._send(400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                return
            if path.startswith("/static/"):
                rel = path[len("/static/"):]
                target = (static_root / rel).resolve()
                if static_root.resolve() not in target.parents or not target.is_file():
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self._send(200, target.read_bytes(), ctype, cache="no-store")
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/video_source":
                size = self.headers.get("Content-Length", "0")
                try:
                    from .video_selection import register_video
                    print(f"[stage9-web] Video upload: {size} bytes")
                    result = register_video(self.rfile, int(size), video_root)
                    self._send(200, json.dumps(result).encode("utf-8"), "application/json")
                except Exception as exc:
                    print(f"[stage9-web] Video upload failed ({size} bytes): {type(exc).__name__}: {exc}")
                    self._send(400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                return
            if parsed.path == "/api/analyze_live":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                request_id = None
                acquired = False
                try:
                    req_data = json.loads(body.decode("utf-8"))
                    request_id = str(req_data.get("request_id") or uuid.uuid4().hex)
                    if len(request_id) > 80:
                        raise ValueError("Request ID không hợp lệ.")
                    acquired = analysis_lock.acquire(blocking=False)
                    if not acquired:
                        self._send(409, 'Máy chủ đang xử lý một yêu cầu khác. Vui lòng chờ hoàn tất.'.encode('utf-8'), 'text/plain; charset=utf-8')
                        return
                    with progress_lock:
                        while len(progress) >= 50:
                            del progress[next(iter(progress))]
                        progress[request_id] = {'status': 'running', 'stage': 0, 'stages': {}, 'started_at': time.time(), 'message': 'Đang chuẩn bị ảnh nguồn và nạp mô hình'}
                    frame_index = req_data.get("estimated_frame", 0)
                    print(f"[stage9-web] LIVE ANALYSIS REQUEST: frame={frame_index}")
                    
                    import subprocess
                    import sys
                    import base64
                    
                    # Every request gets a new directory. This is also the hard
                    # boundary preventing Stage 4 from finding artifacts from a
                    # previous web run.
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    run_id = f"{stamp}_f{int(frame_index):08d}_{uuid.uuid4().hex[:8]}"
                    outdir = root / "outputs" / "live_runs" / run_id
                    outdir.mkdir(parents=True, exist_ok=False)
                    update_progress(request_id, run_id=run_id, frame_index=int(frame_index))
                    
                    import cv2
                    import hashlib
                    from .video_selection import analysis_source_frame
                    img, source_kind = analysis_source_frame(video_root, req_data)
                    # Preserve the chosen pixels without another lossy JPEG pass.
                    frame_path = outdir / f"frame_{frame_index}.png"
                    if not cv2.imwrite(str(frame_path), img):
                        raise ValueError("Không ghi được frame nguồn.")
                    selection_info = {
                        'kind': source_kind, 'frame_index': int(frame_index),
                        'time_sec': req_data.get('time_sec'), 'video_id': req_data.get('video_id'),
                        'frame_index_basis': 'MEDIA_TIME_ESTIMATE' if source_kind == 'DISPLAYED_VIDEO_FRAME' else 'EXACT_INDEX',
                        'image_path': str(frame_path),
                        'pixel_sha256': hashlib.sha256(img.tobytes()).hexdigest(),
                    }
                    (outdir / 'frame_selection.json').write_text(json.dumps(selection_info, indent=2), encoding='utf-8')
                    
                    cmd = [
                        sys.executable,
                        str(root / "live_orchestrator.py"),
                        "--image", str(frame_path),
                        "--frame", str(frame_index),
                        "--outdir", str(outdir)
                    ]
                    
                    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
                    tail = deque(maxlen=100)
                    with (outdir / 'orchestrator.log').open('w', encoding='utf-8') as logfile:
                        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              text=True, encoding='utf-8', errors='replace', env=env, bufsize=1) as proc:
                            for line in proc.stdout:
                                logfile.write(line)
                                logfile.flush()
                                tail.append(line)
                                if line.startswith('__STAGE_PROGRESS__'):
                                    event = json.loads(line[len('__STAGE_PROGRESS__'):])
                                    with progress_lock:
                                        record = progress[request_id]
                                        record['stage'] = event['stage']
                                        record['stages'][str(event['stage'])] = event['status']
                                        record['message'] = 'Đang xử lý' if event['status'] == 'running' else 'Đã chạy xong stage'
                            returncode = proc.wait()
                    if returncode != 0:
                        raise RuntimeError('Orchestrator failed:\n' + ''.join(tail))
                    update_progress(request_id, message='Đang tổng hợp kết quả để hiển thị')
                    
                    # Reload context and mutate global ctx
                    if type(ctx).__name__ == "PipelineContext":
                        from stage9_offside_position.pipeline import build_pipeline_context
                        new_paths = {f"stage{i}": outdir / f"stage{i}.json" for i in range(1, 10)}
                        new_ctx = build_pipeline_context(paths=new_paths, image_path=str(frame_path))
                        ctx.state = new_ctx.state
                        ctx.frame = new_ctx.frame
                        ctx.frame_source = new_ctx.frame_source
                        # Keep the UI on the exact frame selected by the user.
                        ctx.state["frame_index"] = int(frame_index)
                        ctx.state["has_analysis_result"] = True
                        ctx.state["timestamp_sec"] = req_data.get('time_sec')
                        ctx.state["frame_selection"] = selection_info

                    stage5_handoff = json.loads((outdir / "stage5.json").read_text(encoding="utf-8"))
                    track_team = stage5_handoff.get("track_team", {})
                    assigned_count = sum(
                        1 for record in track_team.values()
                        if record.get("team_id") is not None
                    )
                    unknown_count = sum(
                        1 for record in track_team.values()
                        if record.get("team_status") == "UNKNOWN"
                    )
                    stage9_state = json.loads((outdir / "stage9.json").read_text(encoding="utf-8"))
                    stage9_attackers = stage9_state.get("attackers", [])
                    
                    update_progress(request_id, status='completed', finished_at=time.time(), message='Pipeline đã hoàn tất')
                    self._send(200, json.dumps({
                        "status": "ok", "run_id": run_id, "outdir": str(outdir),
                        "frame_selection": selection_info,
                        "stage3_execution": "EXISTING_STAGE3_PIPELINE_FRESH_SINGLE_FRAME_INFERENCE",
                        "stage3_invocation": str(outdir / "stage3_invocation.json"),
                        "stage4_execution": "EXISTING_STAGE4_PIPELINE_FRESH_INFERENCE",
                        "stage4_invocation": str(outdir / "stage4_invocation.json"),
                        "stage5_execution": "EXISTING_STAGE5_PIPELINE_FRESH_SINGLE_FRAME_INFERENCE",
                        "stage5_invocation": str(outdir / "stage5_invocation.json"),
                        "stage5_result": {
                            "selected_frame": stage5_handoff.get("selected_frame"),
                            "assigned_count": assigned_count,
                            "unknown_count": unknown_count,
                            "attacking_team_not_resolved": stage5_handoff.get("attacking_team_not_resolved"),
                            "track_team": track_team,
                        },
                        "stage6_execution": "EXISTING_STAGE6_CONTACT_AWARE_PIPELINE_FRESH_SINGLE_FRAME_INFERENCE",
                        "stage6_invocation": str(outdir / "stage6_invocation.json"),
                        "stage7_execution": "EXISTING_STAGE7_PIPELINE_FRESH_CONTEXT_RESOLUTION",
                        "stage7_invocation": str(outdir / "stage7_invocation.json"),
                        "stage8_execution": "EXISTING_STAGE8_PIPELINE_FRESH_REFERENCE_GEOMETRY",
                        "stage8_invocation": str(outdir / "stage8_invocation.json"),
                        "stage9_execution": "EXISTING_STAGE9_PIPELINE_FRESH_STRICT_CLASSIFICATION",
                        "stage9_invocation": str(outdir / "stage9_invocation.json"),
                        "stage9_result": {
                            "status": stage9_state.get("status"),
                            "mode": stage9_state.get("mode"),
                            "reference": stage9_state.get("reference"),
                            "offside_position_count": sum(1 for row in stage9_attackers if row.get("label") == "OFFSIDE_POSITION"),
                            "onside_count": sum(1 for row in stage9_attackers if row.get("label") == "ONSIDE"),
                            "attackers": stage9_attackers,
                            "note": "Position classification only; not an offside-offence decision.",
                        },
                    }).encode("utf-8"), "application/json")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    if acquired:
                        update_progress(request_id, status='failed', finished_at=time.time(), message='Xử lý thất bại', error=str(e)[-6000:])
                    self._send(400, str(e).encode("utf-8"), "text/plain; charset=utf-8")
                finally:
                    if acquired:
                        analysis_lock.release()
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def log_message(self, fmt, *args):
            print("[stage9-web] " + (fmt % args))

    return Handler


def serve_demo(ctx: DemoContext, host: str = "127.0.0.1", port: int = 8099) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, int(port)), make_handler(ctx))
    return server
