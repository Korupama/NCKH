from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json
import numpy as np

from ...camera import CameraTimelineLite
from ...stage3_adapter import Stage3State, load_stage3_state
from .camera_adapter import stage1_to_fc_camera
from .contracts import FC_EXPORT_SCHEMA
from .sam3d_cache import Sam3DCache


def eligible_tracks(s3: Stage3State, include_roles: Sequence[str]) -> list[str]:
    allowed = {str(x) for x in include_roles}
    return [t.track_id for t in s3.tracks if str(t.role) in allowed]


def source_frames_for_tracks(s3: Stage3State, track_ids: Sequence[str]) -> list[int]:
    wanted = set(track_ids)
    frames = sorted({
        int(o.frame_index)
        for t in s3.tracks if t.track_id in wanted
        for o in t.observations
    })
    return frames


def build_boxes_from_stage3(s3: Stage3State, frames: Sequence[int], track_ids: Sequence[str]) -> np.ndarray:
    frame_to_i = {int(f): i for i, f in enumerate(frames)}
    track_to_j = {str(t): j for j, t in enumerate(track_ids)}
    boxes = np.full((len(frames), len(track_ids), 4), np.nan, dtype=np.float32)
    by_id = s3.track_by_id()
    for tid in track_ids:
        track = by_id.get(tid)
        if track is None:
            continue
        j = track_to_j[tid]
        for obs in track.observations:
            i = frame_to_i.get(int(obs.frame_index))
            if i is None:
                continue
            if np.isfinite(obs.bbox_xyxy).all():
                boxes[i, j] = np.asarray(obs.bbox_xyxy, dtype=np.float32)
    return boxes


def export_field_converter_raw(
    *,
    stage3_state: str | Path,
    camera_dir: str | Path,
    sam3d_cache: str | Path,
    output_root: str | Path,
    sequence_name: str,
    include_roles: Sequence[str] = ("player", "goalkeeper"),
) -> dict:
    s3 = load_stage3_state(stage3_state)
    cams = CameraTimelineLite.load_dir(camera_dir)
    cache = Sam3DCache.load(sam3d_cache)
    track_ids = eligible_tracks(s3, include_roles)
    if not track_ids:
        raise ValueError("No Stage-3 player/goalkeeper tracks selected for Field Converter")
    frames = source_frames_for_tracks(s3, track_ids)
    if not frames:
        raise ValueError("No Stage-3 observations for selected Field Converter tracks")

    if tuple(cache.track_ids) != tuple(track_ids):
        raise ValueError(
            "SAM3D cache track order mismatch. Expected Stage-3 order "
            f"{track_ids}, got {list(cache.track_ids)}"
        )
    if not np.array_equal(cache.frame_indices, np.asarray(frames, dtype=np.int64)):
        raise ValueError(
            "SAM3D cache frame timeline mismatch. Rebuild the cache from the current Stage-3 state."
        )

    stage3_boxes = build_boxes_from_stage3(s3, frames, track_ids)
    cache_box_finite = np.isfinite(cache.boxes_xyxy).all(axis=-1)
    source_box_finite = np.isfinite(stage3_boxes).all(axis=-1)
    both = cache_box_finite & source_box_finite
    if np.any(both):
        max_box_delta = float(np.max(np.abs(cache.boxes_xyxy[both] - stage3_boxes[both])))
        if max_box_delta > 1e-3:
            raise ValueError(f"SAM3D cache was created from different boxes; max delta={max_box_delta:.6f}px")

    T = len(frames)
    K = np.full((T, 3, 3), np.nan, dtype=np.float64)
    R = np.full((T, 3, 3), np.nan, dtype=np.float64)
    t = np.full((T, 3), np.nan, dtype=np.float64)
    k = np.full((T, 2), np.nan, dtype=np.float64)
    missing_camera = []
    for i, frame in enumerate(frames):
        cam = cams.by_frame(frame)
        if cam is None or cam.status == "INVALID":
            missing_camera.append(frame)
            continue
        K[i], R[i], t[i], k[i] = stage1_to_fc_camera(cam)
    if missing_camera:
        raise ValueError(f"Field Converter export requires complete usable camera timeline; missing/invalid={missing_camera[:20]}")

    root = Path(output_root).expanduser().resolve()
    for dirname in ("boxes", "cameras", "skel_2d", "skel_3d_relative", "frames"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    np.save(root / "boxes" / f"{sequence_name}.npy", stage3_boxes.astype(np.float32))
    np.savez_compressed(root / "cameras" / f"{sequence_name}.npz", K=K, R=R, t=t, k=k)
    np.save(root / "skel_2d" / f"{sequence_name}.npy", cache.skel_2d_px.astype(np.float32))
    np.save(root / "skel_3d_relative" / f"{sequence_name}.npy", cache.skel_3d_relative_m.astype(np.float32))

    # Field Converter uses filenames only to recover source frame numbers.
    frame_dir = root / "frames" / sequence_name
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        marker = frame_dir / f"frame_{int(frame):08d}.frame"
        marker.touch(exist_ok=True)

    manifest = {
        "schema_version": FC_EXPORT_SCHEMA,
        "sequence_name": sequence_name,
        "stage3_state": str(Path(stage3_state).expanduser().resolve()),
        "camera_dir": str(Path(camera_dir).expanduser().resolve()),
        "sam3d_cache": str(Path(sam3d_cache).expanduser().resolve()),
        "frame_indices": [int(x) for x in frames],
        "track_ids": list(track_ids),
        "shape": {"T": T, "N": len(track_ids), "J": 25},
        "image_size": [int(s3.replay_context.get("image_width")), int(s3.replay_context.get("image_height"))],
        "source_fps": float(s3.replay_context.get("fps")),
        "selected_frame": int(s3.selected_frame),
        "sam3d_semantic_mapping_validated": bool(cache.semantic_mapping_validated),
        "files": {
            "boxes": str(root / "boxes" / f"{sequence_name}.npy"),
            "cameras": str(root / "cameras" / f"{sequence_name}.npz"),
            "skel_2d": str(root / "skel_2d" / f"{sequence_name}.npy"),
            "skel_3d_relative": str(root / "skel_3d_relative" / f"{sequence_name}.npy"),
        },
    }
    manifest_path = root / "stage4_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
