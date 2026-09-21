from __future__ import annotations

"""Run the official KASportsFormer checkpoint from Stage-3 COCO17 tracks.

The official demo first runs its own detector/HRNet stack. Stage 4 already has
identity-consistent, quality-labelled COCO17 observations, so this worker uses
those directly. KASportsFormer output is scale ambiguous without WorldPose's
per-sample scale metadata: it is valid as a relative-depth initializer and for
PA-MPJPE, but never as pitch-world metric XYZ.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import json
import pickle
import subprocess
import sys

import numpy as np

from ..stage3_adapter import Stage3State, Stage3Track, load_stage3_state
from .kasportsformer import (
    H36M17_NAMES,
    H36M17_TO_WHOLEBODY133,
    write_kasportsformer_cache,
)


H36M_COCO_DESTINATIONS = (9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3)
H36M_COCO_SOURCES = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
LEFT_JOINTS = (1, 2, 3, 14, 15, 16)
RIGHT_JOINTS = (4, 5, 6, 11, 12, 13)
OFFICIAL_WORLDPOSE_DET_CHECKPOINT_SHA256 = (
    "a4e0b9018e4755676a8421edbf8063375d7d22eb2f423f6aa4ba3883ee90a767"
)


@dataclass(frozen=True)
class KASportsFormerRunConfig:
    stage3_state: Path
    repository_root: Path
    model_config: Path
    checkpoint: Path
    output_raw: Path
    output_cache: Optional[Path]
    device: str = "auto"
    batch_size: int = 2
    window_radius_frames: int = 13
    selected_only: bool = False


def coco17_to_h36m17(
    keypoints: np.ndarray,
    confidence: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the repository's COCO17 -> H36M17 geometry and score mapping."""

    coco = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(confidence, dtype=np.float32)
    if coco.shape[-2:] != (17, 2):
        raise ValueError(f"Expected (...,17,2) COCO keypoints, got {coco.shape}")
    if scores.shape != coco.shape[:-1]:
        raise ValueError(f"Expected confidence shape {coco.shape[:-1]}, got {scores.shape}")

    output = np.zeros_like(coco, dtype=np.float32)
    output_scores = np.zeros_like(scores, dtype=np.float32)
    output[..., H36M_COCO_DESTINATIONS, :] = coco[..., H36M_COCO_SOURCES, :]
    output_scores[..., H36M_COCO_DESTINATIONS] = scores[..., H36M_COCO_SOURCES]

    # These equations intentionally mirror demo/lib/preprocess.py in the
    # official repository, including its head/thorax refinements.
    output[..., 10, 0] = np.mean(coco[..., 1:5, 0], axis=-1)
    output[..., 10, 1] = np.sum(coco[..., 1:3, 1], axis=-1) - coco[..., 0, 1]
    output[..., 8, :] = np.mean(coco[..., 5:7, :], axis=-2)
    output[..., 8, :] += (coco[..., 0, :] - output[..., 8, :]) / 3.0
    output[..., 0, :] = np.mean(coco[..., 11:13, :], axis=-2)
    output[..., 7, :] = np.mean(coco[..., (5, 6, 11, 12), :], axis=-2)

    shoulder_mid = np.mean(coco[..., 5:7, :], axis=-2)
    output[..., 9, :] -= (output[..., 9, :] - shoulder_mid) / 4.0
    output[..., 7, 0] += 2.0 * (
        output[..., 7, 0] - np.mean(output[..., (0, 8), 0], axis=-1)
    )
    output[..., 8, 1] -= (
        np.mean(coco[..., 1:3, 1], axis=-1) - coco[..., 0, 1]
    ) * (2.0 / 3.0)

    output_scores[..., 0] = np.mean(scores[..., (11, 12)], axis=-1)
    output_scores[..., 8] = np.mean(scores[..., (5, 6)], axis=-1)
    output_scores[..., 7] = np.mean(output_scores[..., (0, 8)], axis=-1)
    output_scores[..., 10] = np.mean(scores[..., 1:5], axis=-1)
    return output, output_scores


def normalize_screen_coordinates(xyc: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalize x/y exactly as the official demo while preserving confidence."""

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid replay dimensions: {width}x{height}")
    result = np.asarray(xyc, dtype=np.float32).copy()
    if result.shape[-1] != 3:
        raise ValueError(f"Expected input (...,3), got {result.shape}")
    result[..., :2] = result[..., :2] / float(width) * 2.0
    result[..., 0] -= 1.0
    result[..., 1] -= float(height) / float(width)
    return result


def _observation_confidence(track: Stage3Track) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.asarray([obs.frame_index for obs in track.observations], dtype=np.int64)
    xy = np.stack([obs.uv23[:17] for obs in track.observations], axis=0).astype(np.float64)
    state_weight = np.stack(
        [obs.state_weights23[:17] for obs in track.observations], axis=0
    ).astype(np.float64)
    raw_score = np.stack([obs.raw_scores23[:17] for obs in track.observations], axis=0)
    detector_score = np.where(np.isfinite(raw_score), np.clip(raw_score, 0.0, 1.0), 1.0)
    confidence = np.clip(state_weight * detector_score, 0.0, 1.0)
    confidence[~np.isfinite(xy).all(axis=-1)] = 0.0
    return frames, xy, confidence


def interpolate_track_coco17(
    track: Stage3Track,
    query_frames: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate/extrapolate one identity track without inventing confidence.

    Coordinates are linearly interpolated and edge-held. Exact valid Stage-3
    observations retain their semantic confidence. Filled gaps are capped at
    the Stage-3 TEMPORAL_IMPUTED weight (0.30). A joint never observed on the
    track falls back to the tracked-box centre with confidence zero.
    """

    query = np.asarray(query_frames, dtype=np.int64)
    if query.ndim != 1 or query.size == 0:
        raise ValueError("query_frames must be a non-empty one-dimensional sequence")
    source_frames, source_xy, source_conf = _observation_confidence(track)
    if source_frames.size == 0:
        raise ValueError(f"Track {track.track_id} has no Stage-3 observations")

    bbox_centres = np.stack(
        [
            (obs.bbox_xyxy[:2] + obs.bbox_xyxy[2:]) / 2.0
            if np.isfinite(obs.bbox_xyxy).all()
            else np.asarray([0.0, 0.0])
            for obs in track.observations
        ],
        axis=0,
    )
    fallback = np.stack(
        [
            np.interp(query, source_frames, bbox_centres[:, axis])
            for axis in range(2)
        ],
        axis=-1,
    )

    output_xy = np.empty((query.size, 17, 2), dtype=np.float32)
    output_conf = np.zeros((query.size, 17), dtype=np.float32)
    for joint in range(17):
        valid = np.isfinite(source_xy[:, joint]).all(axis=-1) & (source_conf[:, joint] > 0.0)
        if not np.any(valid):
            output_xy[:, joint, :] = fallback
            continue
        valid_frames = source_frames[valid]
        valid_xy = source_xy[valid, joint]
        valid_conf = source_conf[valid, joint]
        for axis in range(2):
            output_xy[:, joint, axis] = np.interp(query, valid_frames, valid_xy[:, axis])
        interpolated_conf = np.interp(query, valid_frames, valid_conf)
        exact_lookup = {int(frame): float(conf) for frame, conf in zip(valid_frames, valid_conf)}
        output_conf[:, joint] = np.asarray(
            [exact_lookup.get(int(frame), min(float(conf), 0.30)) for frame, conf in zip(query, interpolated_conf)],
            dtype=np.float32,
        )
    return output_xy, output_conf


def _wanted_targets(
    state: Stage3State,
    radius: int,
    selected_only: bool,
) -> List[Tuple[Stage3Track, int]]:
    if radius < 0:
        raise ValueError("window_radius_frames must be non-negative")
    lo = state.selected_frame if selected_only else state.selected_frame - radius
    hi = state.selected_frame if selected_only else state.selected_frame + radius
    return [
        (track, obs.frame_index)
        for track in state.tracks
        for obs in track.observations
        if lo <= obs.frame_index <= hi
    ]


def build_model_inputs(
    state: Stage3State,
    *,
    window_length: int,
    window_radius_frames: int,
    selected_only: bool,
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """Build centered 27-frame model windows for the Stage-4 cohort."""

    if window_length <= 0 or window_length % 2 != 1:
        raise ValueError("KASportsFormer n_frames must be a positive odd number")
    width = int(state.replay_context.get("image_width", 0))
    height = int(state.replay_context.get("image_height", 0))
    half = window_length // 2
    samples: List[np.ndarray] = []
    keys: List[Tuple[str, int]] = []
    for track, target_frame in _wanted_targets(state, window_radius_frames, selected_only):
        temporal_frames = np.arange(target_frame - half, target_frame + half + 1)
        coco_xy, coco_conf = interpolate_track_coco17(track, temporal_frames)
        h36m_xy, h36m_conf = coco17_to_h36m17(coco_xy, coco_conf)
        xyc = np.concatenate((h36m_xy, h36m_conf[..., None]), axis=-1)
        samples.append(normalize_screen_coordinates(xyc, width, height))
        keys.append((track.track_id, int(target_frame)))
    if not samples:
        raise ValueError("No Stage-3 observations fall inside the requested Stage-4 window")
    return np.stack(samples, axis=0).astype(np.float32), keys


def _load_yaml(path: Path) -> Mapping[str, object]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("KASportsFormer worker requires PyYAML") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Model config must contain a YAML object: {path}")
    return value


def _load_checkpoint_payload(torch, checkpoint: Path, checkpoint_sha256: str):
    """Load the audited official asset without weakening unknown-file safety."""

    try:
        return torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        # The official checkpoint contains a NumPy scalar in its training
        # metadata, which older weights-only unpicklers reject even though the
        # model state itself is valid. Full pickle loading is permitted only
        # for the exact audited official asset; an unknown or modified file is
        # never silently loaded with code execution enabled.
        if checkpoint_sha256.lower() != OFFICIAL_WORLDPOSE_DET_CHECKPOINT_SHA256:
            raise RuntimeError(
                "weights_only checkpoint loading failed and the file SHA-256 "
                "does not match the audited official KASportsFormer WorldPose "
                "detected-2D checkpoint; refusing unsafe fallback"
            ) from exc
        return torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError as exc:  # PyTorch before weights_only was introduced
        if checkpoint_sha256.lower() != OFFICIAL_WORLDPOSE_DET_CHECKPOINT_SHA256:
            raise RuntimeError(
                "This PyTorch version lacks weights_only loading and the "
                "checkpoint SHA-256 is not the audited official asset; "
                "refusing legacy pickle loading"
            ) from exc
        return torch.load(str(checkpoint), map_location="cpu")


def _load_model(
    repository_root: Path,
    config: Mapping[str, object],
    checkpoint: Path,
    checkpoint_sha256: str,
    device: str,
):
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("KASportsFormer worker requires PyTorch") from exc

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(config.get("model_name")) != "KASportsFormer":
        raise ValueError(f"Expected model_name=KASportsFormer, got {config.get('model_name')}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is false: {device}")
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    try:
        from model.KASportsFormer import KASportsFormer  # type: ignore
    except Exception as exc:  # pragma: no cover - third-party environment
        raise RuntimeError(
            "Could not import the official KASportsFormer model. Install its "
            "runtime dependencies (at minimum torch, timm and PyYAML) and pass "
            "--kasportsformer-root to the cloned official repository."
        ) from exc

    activation_name = str(config.get("act_layer", "gelu")).lower()
    activation = {"gelu": nn.GELU, "relu": nn.ReLU}.get(activation_name)
    if activation is None:
        raise ValueError(f"Unsupported KASportsFormer act_layer: {activation_name}")
    kwargs = {
        "n_layers": int(config.get("n_layers", 26)),
        "dim_in": int(config.get("dim_in", 3)),
        "dim_feat": int(config.get("dim_feat", 128)),
        "dim_rep": int(config.get("dim_rep", 512)),
        "dim_out": int(config.get("dim_out", 3)),
        "mlp_ratio": float(config.get("mlp_ratio", 4)),
        "act_layer": activation,
        "attn_drop": float(config.get("attn_drop", 0.0)),
        "drop": float(config.get("drop", 0.0)),
        "drop_path": float(config.get("drop_path", 0.0)),
        "use_layer_scale": bool(config.get("use_layer_scale", True)),
        "layer_scale_init_value": float(config.get("layer_scale_init_value", 1e-5)),
        "use_adaptive_fusion": bool(config.get("use_adaptive_fusion", True)),
        "num_heads": int(config.get("num_heads", 8)),
        "qkv_bias": bool(config.get("qkv_bias", False)),
        "qkv_scale": config.get("qkv_scale"),
        "hierarchical": bool(config.get("hierarchical", False)),
        "num_joints": int(config.get("num_joints", 17)),
        "use_temporal_similarity": bool(config.get("use_temporal_similarity", True)),
        "temporal_connection_len": int(config.get("temporal_connection_len", 1)),
        "use_tcn": bool(config.get("use_tcn", False)),
        "graph_only": bool(config.get("graph_only", False)),
        "neighbour_num": int(config.get("neighbour_num", 4)),
        "n_frames": int(config.get("n_frames", 27)),
    }
    if kwargs["dim_in"] != 3 or kwargs["dim_out"] != 3:
        raise ValueError("This worker requires KASportsFormer dim_in=3 and dim_out=3")
    if kwargs["num_joints"] != 17:
        raise ValueError("This worker requires the official 17-joint KASportsFormer checkpoint")
    model = KASportsFormer(**kwargs)
    payload = _load_checkpoint_payload(torch, checkpoint, checkpoint_sha256)
    state_dict = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a model state_dict")
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    metadata = dict(kwargs)
    metadata["act_layer"] = activation_name
    return torch, model, metadata, device


def _flip_tensor(tensor):
    flipped = tensor.clone()
    flipped[..., 0] *= -1
    source = flipped.clone()
    flipped[..., list(LEFT_JOINTS + RIGHT_JOINTS), :] = source[
        ..., list(RIGHT_JOINTS + LEFT_JOINTS), :
    ]
    return flipped


def _predict(torch, model, inputs: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    outputs: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            stop = min(start + batch_size, len(inputs))
            batch = torch.from_numpy(inputs[start:stop]).to(device)
            normal = model(batch)
            flipped = _flip_tensor(model(_flip_tensor(batch)))
            prediction = (normal + flipped) / 2.0
            prediction[:, :, 0, :] = 0.0
            centre = prediction[:, prediction.shape[1] // 2]
            outputs.append(centre.detach().cpu().numpy())
            batch_index = start // batch_size
            if batch_index == 0 or stop == len(inputs) or batch_index % 10 == 0:
                print(f"[KASportsFormer] inferred {stop}/{len(inputs)} pose windows", flush=True)
    result = np.concatenate(outputs, axis=0).astype(np.float64)
    if result.shape != (len(inputs), 17, 3) or not np.isfinite(result).all():
        raise RuntimeError(f"Unexpected/non-finite KASportsFormer output: {result.shape}")
    return result


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_revision(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def run_kasportsformer_worker(config: KASportsFormerRunConfig) -> Mapping[str, object]:
    for label, path in (
        ("Stage-3 state", config.stage3_state),
        ("KASportsFormer repository", config.repository_root),
        ("model config", config.model_config),
        ("checkpoint", config.checkpoint),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    expected_model_file = config.repository_root / "model" / "KASportsFormer.py"
    if not expected_model_file.is_file():
        raise FileNotFoundError(
            f"Official KASportsFormer model file not found under repository root: {expected_model_file}"
        )
    checkpoint_hash = _file_sha256(config.checkpoint)

    stage3 = load_stage3_state(config.stage3_state)
    model_config = _load_yaml(config.model_config)
    window_length = int(model_config.get("n_frames", 27))
    inputs, keys = build_model_inputs(
        stage3,
        window_length=window_length,
        window_radius_frames=config.window_radius_frames,
        selected_only=config.selected_only,
    )
    print(
        f"[KASportsFormer] prepared {len(keys)} centered {window_length}-frame windows "
        f"from {len({track_id for track_id, _ in keys})} tracks",
        flush=True,
    )
    torch, model, model_kwargs, resolved_device = _load_model(
        config.repository_root,
        model_config,
        config.checkpoint,
        checkpoint_hash,
        config.device,
    )
    print(f"[KASportsFormer] checkpoint loaded on {resolved_device}", flush=True)
    xyz = _predict(torch, model, inputs, resolved_device, config.batch_size)

    records: List[Dict[str, object]] = []
    for (track_id, frame_index), pose in zip(keys, xyz):
        records.append(
            {
                "frame_index": int(frame_index),
                "track_id": track_id,
                "xyz": pose.tolist(),
                "joint_names": list(H36M17_NAMES),
                "wholebody_indices": list(H36M17_TO_WHOLEBODY133),
            }
        )

    revision = _repository_revision(config.repository_root)
    raw = {
        "schema_version": "external-pose-records-1.0",
        "coordinate_scope": "SCALE_AMBIGUOUS",
        "units": "model",
        "joint_names": list(H36M17_NAMES),
        "model": {
            "name": "KASportsFormer",
            "variant": "WorldPose detected-2D checkpoint",
            "n_frames": window_length,
            "input_channel_number": int(model_config.get("input_channel_number", 3)),
        },
        "provenance": {
            "stage3_state": str(config.stage3_state),
            "repository_root": str(config.repository_root),
            "repository_revision": revision,
            "model_config": str(config.model_config),
            "checkpoint": str(config.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "device": resolved_device,
            "model_kwargs": model_kwargs,
            "preprocessing": "official_COCO17_to_H36M17_and_screen_normalization",
            "temporal_sampling": "centered_window_edge_hold_and_low_confidence_interpolation",
            "flip_test_time_augmentation": True,
            "worldpose_scale_metadata_available": False,
        },
        "poses": records,
    }
    config.output_raw.parent.mkdir(parents=True, exist_ok=True)
    config.output_raw.write_text(json.dumps(raw, indent=2, allow_nan=False), encoding="utf-8")

    cache_path: Optional[str] = None
    if config.output_cache is not None:
        written = write_kasportsformer_cache(
            config.output_cache,
            records,
            source_description=(
                "Official KASportsFormer WorldPose detected-2D checkpoint; "
                f"sha256={checkpoint_hash}; Stage-3 COCO17 input; scale ambiguous"
            ),
        )
        cache_path = str(written)

    return {
        "schema_version": "stage4-kasportsformer-worker-result-1.0",
        "raw_output": str(config.output_raw),
        "initializer_cache": cache_path,
        "pose_records": len(records),
        "track_count": len({record["track_id"] for record in records}),
        "selected_frame": stage3.selected_frame,
        "window_radius_frames": config.window_radius_frames,
        "model_window_frames": window_length,
        "coordinate_scope": "SCALE_AMBIGUOUS",
        "checkpoint_sha256": checkpoint_hash,
        "repository_revision": revision,
        "device": resolved_device,
    }
