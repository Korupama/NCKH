from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# The worker is intentionally runnable from a separate SAM3D CUDA environment.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from stage4_metric3d.backends.sam3d_pitch_refined.cache import (
    MHR70_NAMES,
    normalize_mhr_name,
    save_sam3d_native_cache,
)
from stage4_metric3d.camera import CameraTimelineLite
from stage4_metric3d.stage3_adapter import load_stage3_state


def _eligible_tracks(s3, requested_track_ids: list[str] | None = None) -> list[str]:
    eligible = [t.track_id for t in s3.tracks if str(t.role) in {"player", "goalkeeper"}]
    if not requested_track_ids:
        return eligible
    requested = set(str(track_id) for track_id in requested_track_ids)
    missing = sorted(requested.difference(eligible))
    if missing:
        raise ValueError(f"Requested track IDs are absent or ineligible in Stage-3: {missing}")
    return [track_id for track_id in eligible if track_id in requested]


def _source_frames(s3, track_ids: list[str]) -> list[int]:
    wanted = set(track_ids)
    return sorted({int(o.frame_index) for t in s3.tracks if t.track_id in wanted for o in t.observations})


def _boxes(s3, frames: list[int], track_ids: list[str]) -> np.ndarray:
    fi = {f: i for i, f in enumerate(frames)}
    tj = {t: j for j, t in enumerate(track_ids)}
    out = np.full((len(frames), len(track_ids), 4), np.nan, dtype=np.float32)
    for tr in s3.tracks:
        if tr.track_id not in tj:
            continue
        for obs in tr.observations:
            i = fi.get(int(obs.frame_index))
            if i is not None and np.isfinite(obs.bbox_xyxy).all():
                out[i, tj[tr.track_id]] = np.asarray(obs.bbox_xyxy, dtype=np.float32)
    return out


def _frame_file_map(frames_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames-dir not found: {frames_dir}")
    for path in sorted(frames_dir.iterdir()):
        if not path.is_file():
            continue
        matches = re.findall(r"\d+", path.stem)
        if matches:
            result[int(matches[-1])] = path
    return result


def _open_video(path: Path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def _read_frame(*, frame: int, frames_dir: Path | None, file_map: dict[int, Path] | None, video_cap):
    import cv2
    if frames_dir is not None:
        assert file_map is not None
        path = file_map.get(int(frame))
        if path is None:
            raise FileNotFoundError(f"No image corresponding to frame {frame} in {frames_dir}")
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img_bgr = video_cap.read()
    if not ok or img_bgr is None:
        raise RuntimeError(f"Failed to read video frame {frame}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _build_estimator(checkpoint: str, mhr_path: str, *, device: str, sam3d_root: str | None, inference_type: str = "body"):
    import torch
    if sam3d_root:
        root = Path(sam3d_root).expanduser().resolve()
        if not (root / "sam_3d_body").is_dir():
            raise FileNotFoundError(f"sam3d-root does not contain sam_3d_body: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "SAM3D_CUDA_UNAVAILABLE: --device cuda was requested but torch.cuda.is_available() is false."
        )
    if resolved_device == "cpu":
        if inference_type != "body":
            raise ValueError("CPU compatibility currently supports --inference-type body only; full hand refinement still requires CUDA.")
        print("[SAM3D-v0.5] CPU compatibility mode enabled; inference can be very slow.", flush=True)
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from sam_3d_body.metadata.mhr70 import mhr_names

    installed_names = tuple(normalize_mhr_name(x) for x in mhr_names)
    if installed_names != MHR70_NAMES:
        raise RuntimeError(
            "SAM3D_MHR70_SCHEMA_MISMATCH: installed sam-3d-body metadata differs from the Stage4 v0.5 verified MHR70 schema."
        )
    torch_device = torch.device(resolved_device)
    model, model_cfg = load_sam_3d_body(checkpoint, device=torch_device, mhr_path=mhr_path)
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )
    return estimator, resolved_device


def main() -> int:
    p = argparse.ArgumentParser(description="Run official SAM 3D Body on tracked players and write Stage4 v0.5 native MHR70 cache")
    p.add_argument("--stage3-state", required=True)
    p.add_argument("--camera-dir", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames-dir")
    source.add_argument("--video")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mhr-path", required=True)
    p.add_argument("--sam3d-root", default=None, help="Root of the official sam-3d-body repository when it is not installed as a package")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--cpu-threads", type=int, default=None, help="Limit PyTorch CPU threads, for example 4, to leave capacity for other applications")
    p.add_argument("--person-batch-size", type=int, default=None, help="Number of tracked people per SAM3D call; use 1 on CPU to limit RAM")
    p.add_argument("--track-id", action="append", default=None, help="Debug/smoke test: process only this track ID; repeat for multiple tracks")
    p.add_argument("--output-cache", required=True)
    p.add_argument("--inference-type", choices=("body", "full"), default="body")
    frame_selection = p.add_mutually_exclusive_group()
    frame_selection.add_argument(
        "--frame-index",
        type=int,
        action="append",
        default=None,
        help="Run one exact zero-based source frame. Repeat to select multiple frames.",
    )
    frame_selection.add_argument("--limit-frames", type=int, default=None, help="Debug only: run the first N available frames")
    args = p.parse_args()

    import torch

    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            p.error("--cpu-threads must be at least 1")
        torch.set_num_threads(args.cpu_threads)
    if args.person_batch_size is not None and args.person_batch_size < 1:
        p.error("--person-batch-size must be at least 1")

    s3 = load_stage3_state(args.stage3_state)
    cams = CameraTimelineLite.load_dir(args.camera_dir)
    track_ids = _eligible_tracks(s3, args.track_id)
    frames = _source_frames(s3, track_ids)
    if args.frame_index:
        requested = sorted(set(int(frame) for frame in args.frame_index))
        available = set(frames)
        missing = [frame for frame in requested if frame not in available]
        if missing:
            raise ValueError(f"Requested frame indices are absent from Stage-3: {missing}")
        frames = requested
    elif args.limit_frames is not None:
        frames = frames[: max(0, int(args.limit_frames))]
    boxes = _boxes(s3, frames, track_ids)
    T, N, J = len(frames), len(track_ids), len(MHR70_NAMES)
    out2d = np.full((T, N, J, 2), np.nan, dtype=np.float32)
    out3d = np.full((T, N, J, 3), np.nan, dtype=np.float32)
    cam_t = np.full((T, N, 3), np.nan, dtype=np.float32)
    focal = np.full((T, N), np.nan, dtype=np.float32)
    valid = np.zeros((T, N), dtype=bool)

    estimator, resolved_device = _build_estimator(
        args.checkpoint,
        args.mhr_path,
        device=args.device,
        sam3d_root=args.sam3d_root,
        inference_type=args.inference_type,
    )
    person_batch_size = (1 if resolved_device == "cpu" else len(track_ids)) if args.person_batch_size is None else int(args.person_batch_size)
    if person_batch_size < 1:
        raise ValueError("--person-batch-size must be at least 1")
    frames_dir = None if args.frames_dir is None else Path(args.frames_dir).expanduser().resolve()
    file_map = None if frames_dir is None else _frame_file_map(frames_dir)
    cap = None if args.video is None else _open_video(Path(args.video).expanduser().resolve())

    failures: list[dict] = []
    try:
        for ti, frame in enumerate(frames):
            camera = cams.by_frame(frame)
            if camera is None or camera.status == "INVALID":
                failures.append({"frame_index": frame, "reason": "camera_missing_or_invalid"})
                continue
            finite_idx = np.flatnonzero(np.isfinite(boxes[ti]).all(axis=1))
            if finite_idx.size == 0:
                continue
            image = _read_frame(frame=frame, frames_dir=frames_dir, file_map=file_map, video_cap=cap)
            # SAM3D accepts externally supplied intrinsics. This prevents its optional FOV estimator
            # from introducing a second camera model inconsistent with Stage 1.
            # Upstream expects a batch of intrinsics (B,3,3), even for one image.
            cam_int = torch.as_tensor(np.asarray(camera.K, dtype=np.float32), dtype=torch.float32, device=resolved_device).unsqueeze(0)
            for start in range(0, len(finite_idx), person_batch_size):
                chunk_idx = finite_idx[start : start + person_batch_size]
                print(f"[SAM3D-v0.5] frame={frame} processing tracks={[track_ids[i] for i in chunk_idx.tolist()]} device={resolved_device}", flush=True)
                bboxes = boxes[ti, chunk_idx].astype(np.float32)
                outputs = estimator.process_one_image(
                    image,
                    bboxes=bboxes,
                    cam_int=cam_int,
                    use_mask=False,
                    inference_type=args.inference_type,
                )
                if len(outputs) != len(chunk_idx):
                    raise RuntimeError(
                        f"SAM3D_OUTPUT_COUNT_MISMATCH frame={frame}: requested {len(chunk_idx)} boxes, got {len(outputs)} outputs"
                    )
                for output, person_idx in zip(outputs, chunk_idx.tolist()):
                    p2d = np.asarray(output.get("pred_keypoints_2d"), dtype=np.float32)
                    p3d = np.asarray(output.get("pred_keypoints_3d"), dtype=np.float32)
                    trans = np.asarray(output.get("pred_cam_t"), dtype=np.float32).reshape(-1)
                    foc = np.asarray(output.get("focal_length"), dtype=np.float32).reshape(-1)
                    if p2d.shape != (J, 2) or p3d.shape != (J, 3) or trans.shape != (3,):
                        raise RuntimeError(
                            "SAM3D_NATIVE_SCHEMA_MISMATCH: Stage4 v0.5 expects official MHR70 outputs "
                            f"2D=(70,2), 3D=(70,3), pred_cam_t=(3,), got {p2d.shape}, {p3d.shape}, {trans.shape}."
                        )
                    if foc.size < 1 or not (np.isfinite(p2d).all() and np.isfinite(p3d).all() and np.isfinite(trans).all()):
                        continue
                    out2d[ti, person_idx] = p2d
                    out3d[ti, person_idx] = p3d
                    cam_t[ti, person_idx] = trans
                    focal[ti, person_idx] = float(foc[0])
                    valid[ti, person_idx] = True
            print(f"[SAM3D-v0.5] {ti + 1}/{T} frame={frame} valid={int(valid[ti].sum())}/{N}", flush=True)
    finally:
        if cap is not None:
            cap.release()

    metadata = {
        "producer": "run_sam3d_body_worker.py",
        "producer_version": "0.5.0",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "mhr_path": str(Path(args.mhr_path).expanduser().resolve()),
        "stage3_state": str(Path(args.stage3_state).expanduser().resolve()),
        "camera_dir": str(Path(args.camera_dir).expanduser().resolve()),
        "frames_dir": None if frames_dir is None else str(frames_dir),
        "video": None if args.video is None else str(Path(args.video).expanduser().resolve()),
        "inference_type": args.inference_type,
        "device": resolved_device,
        "runtime_mode": "LOCAL_CPU_BODY_COMPATIBILITY" if resolved_device == "cpu" else "CUDA",
        "cpu_threads": torch.get_num_threads(),
        "person_batch_size": person_batch_size,
        "requested_track_ids": args.track_id,
        "requested_frame_indices": None if not args.frame_index else frames,
        "joint_schema": "SAM3D_MHR70_OFFICIAL",
        "camera_intrinsics_source": "STAGE1_K",
        "failures": failures,
    }
    cache_path = save_sam3d_native_cache(
        args.output_cache,
        frame_indices=frames,
        track_ids=track_ids,
        boxes_xyxy=boxes,
        skel_2d_px=out2d,
        skel_3d_relative_m=out3d,
        pred_cam_t_m=cam_t,
        focal_length_px=focal,
        valid_mask=valid,
        joint_names=MHR70_NAMES,
        metadata=metadata,
    )
    print(json.dumps({
        "cache": str(cache_path), "schema": "stage4-sam3d-native-cache-1.0",
        "T": T, "N": N, "J": J,
        "valid_player_frames": int(valid.sum()),
        "coverage": float(valid.mean()) if valid.size else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
