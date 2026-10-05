from __future__ import annotations

"""Independent Stage-4 WorldPose benchmark for the frozen KASportsFormer baseline.

This lane owns its WorldPose input, model checkpoint and evaluation protocol. It
does not import Stage 1 or Stage 3 and therefore cannot silently turn an
integration artifact into a model benchmark.
"""

import argparse
import gc
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np


WORLDPOSE_DATASET = "WorldPose"
WORLDPOSE_PROTOCOL = "KASportsFormer official WorldPose evaluation protocol"
EXPECTED_CHECKPOINT_SHA256 = "a4e0b9018e4755676a8421edbf8063375d7d22eb2f423f6aa4ba3883ee90a767"
N_FRAMES = 27
N_JOINTS = 17
IMAGE_WIDTH = 1920.0
IMAGE_HEIGHT = 1080.0
ROOT_INDEX = 0
LEFT_JOINTS = (1, 2, 3, 14, 15, 16)
RIGHT_JOINTS = (4, 5, 6, 11, 12, 13)
JOINT_NAMES = (
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resample(ori_len: int, target_len: int) -> np.ndarray:
    if ori_len <= 0:
        raise ValueError("Cannot resample an empty WorldPose sequence")
    even = np.linspace(0, ori_len, num=target_len, endpoint=False)
    if ori_len < target_len:
        low = np.floor(even)
        high = np.ceil(even)
        selected = np.random.randint(2, size=even.shape)
        result = np.sort(selected * low + (1 - selected) * high)
    else:
        interval = even[1] - even[0]
        result = np.random.random(even.shape) * interval + even
    return np.clip(result, a_min=0, a_max=ori_len - 1).astype(np.uint32)


def official_test_clips(source: Iterable[str], n_frames: int = N_FRAMES) -> list[np.ndarray]:
    """Match KASportsFormer's deterministic ``mysplit_clips`` implementation."""
    source = list(source)
    result: list[np.ndarray] = []
    start = 0
    i = 0
    while i < len(source):
        if source[i] != source[start]:
            if i - start >= (n_frames / 2):
                result.append(_resample(i - start, n_frames) + start)
            start = i
            i -= 1
        elif i - start + 1 == n_frames:
            result.append(np.arange(start, i + 1, dtype=np.int64))
            start += n_frames
        i += 1
    return result


def _flip_tensor(torch, tensor):
    flipped = tensor.clone()
    flipped[..., 0] *= -1
    source = flipped.clone()
    swapped = list(LEFT_JOINTS + RIGHT_JOINTS)
    paired = list(RIGHT_JOINTS + LEFT_JOINTS)
    flipped[..., swapped, :] = source[..., paired, :]
    return flipped


def _load_model(repo_root: Path, config_path: Path, checkpoint: Path, device: str):
    import torch
    import yaml
    from torch import nn

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from model.KASportsFormer import KASportsFormer  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name != "timm":
            raise
        # The upstream KASportsFormer only needs timm's DropPath for this
        # model. Keep the benchmark runnable in the project's CPU env without
        # mutating that environment or changing the upstream model source.
        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x):
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
                return x.div(keep_prob) * random_tensor.floor()

        timm_module = types.ModuleType("timm")
        timm_models = types.ModuleType("timm.models")
        timm_layers = types.ModuleType("timm.models.layers")
        timm_layers.DropPath = DropPath
        timm_models.layers = timm_layers
        timm_module.models = timm_models
        sys.modules.update({
            "timm": timm_module,
            "timm.models": timm_models,
            "timm.models.layers": timm_layers,
        })
        from model.KASportsFormer import KASportsFormer  # type: ignore

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    activation = {"gelu": nn.GELU, "relu": nn.ReLU}[str(config.get("act_layer", "gelu")).lower()]
    kwargs = {
        "n_layers": int(config["n_layers"]), "dim_in": int(config["dim_in"]),
        "dim_feat": int(config["dim_feat"]), "dim_rep": int(config["dim_rep"]),
        "dim_out": int(config["dim_out"]), "mlp_ratio": float(config["mlp_ratio"]),
        "act_layer": activation, "attn_drop": float(config["attn_drop"]),
        "drop": float(config["drop"]), "drop_path": float(config.get("drop_path", 0.0)),
        "use_layer_scale": bool(config["use_layer_scale"]),
        "layer_scale_init_value": float(config["layer_scale_init_value"]),
        "use_adaptive_fusion": bool(config["use_adaptive_fusion"]),
        "num_heads": int(config["num_heads"]), "qkv_bias": bool(config["qkv_bias"]),
        "qkv_scale": config.get("qkv_scale"), "hierarchical": bool(config["hierarchical"]),
        "num_joints": int(config["num_joints"]),
        "use_temporal_similarity": bool(config["use_temporal_similarity"]),
        "temporal_connection_len": int(config["temporal_connection_len"]),
        "use_tcn": bool(config["use_tcn"]), "graph_only": bool(config["graph_only"]),
        "neighbour_num": int(config["neighbour_num"]), "n_frames": int(config["n_frames"]),
    }
    model = KASportsFormer(**kwargs)
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "WorldPose checkpoint hash mismatch; refusing to benchmark an "
            f"unverified asset ({checkpoint_hash})"
        )
    try:
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    model.to(resolved_device).eval()
    return torch, model, resolved_device, kwargs, checkpoint_hash


def _procrustes_mpjpe(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    mu_x = np.mean(target, axis=1, keepdims=True)
    mu_y = np.mean(predicted, axis=1, keepdims=True)
    x0 = target - mu_x
    y0 = predicted - mu_y
    norm_x = np.sqrt(np.sum(x0 * x0, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0 * y0, axis=(1, 2), keepdims=True))
    norm_x = np.maximum(norm_x, 1e-8)
    norm_y = np.maximum(norm_y, 1e-8)
    x0 /= norm_x
    y0 /= norm_y
    h = np.matmul(x0.transpose(0, 2, 1), y0)
    u, singular, vt = np.linalg.svd(h)
    v = vt.transpose(0, 2, 1)
    det_sign = np.sign(np.linalg.det(np.matmul(v, u.transpose(0, 2, 1))))
    det_sign[det_sign == 0] = 1.0
    v[:, :, -1] *= det_sign[:, None]
    singular[:, -1] *= det_sign
    rotation = np.matmul(v, u.transpose(0, 2, 1))
    scale = np.sum(singular, axis=1, keepdims=True)[..., None] * norm_x / norm_y
    translation = mu_x - scale * np.matmul(mu_y, rotation)
    aligned = scale * np.matmul(predicted, rotation) + translation
    return np.mean(np.linalg.norm(aligned - target, axis=-1), axis=1)


def _metric_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_mm": None if not array.size else float(np.mean(array)),
        "median_mm": None if not array.size else float(np.median(array)),
        "p90_mm": None if not array.size else float(np.percentile(array, 90)),
        "p95_mm": None if not array.size else float(np.percentile(array, 95)),
    }


def run_benchmark(
    *, data_path: Path, checkpoint: Path, repo_root: Path, config_path: Path,
    report_path: Path, batch_size: int, device: str, max_windows: int | None,
) -> dict[str, Any]:
    np.random.seed(114514)
    with data_path.open("rb") as handle:
        raw = pickle.load(handle)
    test = raw["test"]
    source = list(test["source"])
    clips = official_test_clips(source)
    if max_windows is not None:
        clips = clips[:max_windows]
    if not clips:
        raise RuntimeError("WorldPose test split produced no evaluation clips")

    # Keep only the test arrays needed by the official evaluation protocol.
    inputs = np.asarray(test["joint_2d"], dtype=np.float32)
    inputs /= IMAGE_WIDTH
    inputs *= 2.0
    inputs -= np.asarray([1.0, IMAGE_HEIGHT / IMAGE_WIDTH], dtype=np.float32)
    confidence = np.asarray(test["confidence"], dtype=np.float32)[..., None]
    inputs = np.concatenate((inputs, confidence), axis=-1)
    labels_scaled = np.asarray(test["joints_2.5d_image"], dtype=np.float32)
    factors = np.asarray(test["2.5d_factor"], dtype=np.float32)
    actions = np.asarray(test["action"], dtype=str)
    del raw, test
    gc.collect()

    torch, model, resolved_device, model_kwargs, checkpoint_hash = _load_model(
        repo_root, config_path, checkpoint, device
    )
    model_kwargs_report = {
        key: (value.__name__ if isinstance(value, type) else value)
        for key, value in model_kwargs.items()
    }
    all_mpjpe: list[float] = []
    all_p_mpjpe: list[float] = []
    all_acceleration: list[float] = []
    joint_errors: list[np.ndarray] = []
    action_joint_errors: dict[str, list[np.ndarray]] = {}
    action_metrics: dict[str, dict[str, list[float]]] = {}
    windows_done = 0
    frames_done = 0
    with torch.inference_mode():
        for start in range(0, len(clips), batch_size):
            batch_clips = clips[start:start + batch_size]
            indices = np.stack(batch_clips, axis=0)
            batch = torch.from_numpy(inputs[indices]).to(resolved_device)
            prediction = model(batch)
            flipped = _flip_tensor(torch, batch)
            prediction = (prediction + _flip_tensor(torch, model(flipped))) / 2.0
            prediction[:, :, ROOT_INDEX, :] = 0.0
            predicted = prediction.detach().cpu().numpy().astype(np.float32)
            target = labels_scaled[indices].copy()
            scale = factors[indices][..., None, None]
            predicted[..., :2] = (predicted[..., :2] + np.asarray([1.0, IMAGE_HEIGHT / IMAGE_WIDTH])) * IMAGE_WIDTH / 2.0
            predicted[..., 2:] = predicted[..., 2:] * IMAGE_WIDTH / 2.0
            predicted *= scale
            target *= 1.0
            predicted -= predicted[:, :, ROOT_INDEX:ROOT_INDEX + 1, :]
            target -= target[:, :, ROOT_INDEX:ROOT_INDEX + 1, :]

            frame_mpjpe = np.mean(np.linalg.norm(predicted - target, axis=-1), axis=-1)
            frame_p_mpjpe = _procrustes_mpjpe(
                predicted.reshape(-1, N_JOINTS, 3), target.reshape(-1, N_JOINTS, 3)
            ).reshape(len(batch_clips), N_FRAMES)
            frame_acc = np.mean(
                np.linalg.norm(
                    (predicted[:, 2:] - 2 * predicted[:, 1:-1] + predicted[:, :-2])
                    - (target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]),
                    axis=-1,
                ),
                axis=-1,
            )
            all_mpjpe.extend(frame_mpjpe.reshape(-1).tolist())
            all_p_mpjpe.extend(frame_p_mpjpe.reshape(-1).tolist())
            all_acceleration.extend(frame_acc.reshape(-1).tolist())
            joint_errors.append(np.linalg.norm(predicted - target, axis=-1).reshape(-1, N_JOINTS))
            for row, clip in enumerate(batch_clips):
                action = str(actions[int(clip[0])])
                bucket = action_metrics.setdefault(action, {"mpjpe": [], "p_mpjpe": [], "acceleration": []})
                bucket["mpjpe"].extend(frame_mpjpe[row].tolist())
                bucket["p_mpjpe"].extend(frame_p_mpjpe[row].tolist())
                bucket["acceleration"].extend(frame_acc[row].tolist())
                action_joint_errors.setdefault(action, []).append(
                    np.linalg.norm(predicted[row] - target[row], axis=-1)
                )
            windows_done += len(batch_clips)
            frames_done += len(batch_clips) * N_FRAMES
            if windows_done == len(clips) or windows_done % (batch_size * 20) == 0:
                print(f"evaluated_windows={windows_done}/{len(clips)} frames={frames_done}", flush=True)

    joint = np.mean(np.concatenate(joint_errors, axis=0), axis=0)
    action_report = {
        action: {
            "frame_count": len(values["mpjpe"]),
            "mpjpe_mm": float(np.mean(values["mpjpe"])),
            "p_mpjpe_mm": float(np.mean(values["p_mpjpe"])),
            "acceleration_error_mm": float(np.mean(values["acceleration"])),
        }
        for action, values in sorted(action_metrics.items())
    }
    # The upstream evaluator's headline values are macro-averages over the
    # action/game buckets, rather than a micro-average over all frames. Keep
    # that protocol result explicit and retain the micro summaries above for
    # diagnostic use.
    official_mpjpe = [row["mpjpe_mm"] for row in action_report.values()]
    official_p_mpjpe = [row["p_mpjpe_mm"] for row in action_report.values()]
    official_acceleration = [row["acceleration_error_mm"] for row in action_report.values()]
    official_joint = np.mean(
        np.stack(
            [np.mean(np.concatenate(action_joint_errors[action], axis=0), axis=0)
             for action in sorted(action_joint_errors)],
            axis=0,
        ),
        axis=0,
    )
    report = {
        "schema_version": "stage4-worldpose-benchmark-report-1.0",
        "status": "EVALUATED",
        "dataset": {
            "name": WORLDPOSE_DATASET,
            "protocol": WORLDPOSE_PROTOCOL,
            "source_pickle": str(data_path.resolve()),
            "source_sha256": sha256_file(data_path),
            "split": "test",
            "sequence_disjoint_split": "official WorldPose game split",
        },
        "model": {
            "name": "KASportsFormer",
            "variant": "WorldPose detected-2D checkpoint",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "repository": str(repo_root.resolve()),
            "config": str(config_path.resolve()),
            "device": resolved_device,
            "model_kwargs": model_kwargs_report,
        },
        "protocol": {
            "window_length": N_FRAMES,
            "test_stride": N_FRAMES,
            "seed": 114514,
            "flip_test_time_augmentation": True,
            "input": "WorldPose HRNet/GT-box 2D detections plus confidence from wp_hr_conf_cam_source_final.pkl",
            "coordinate_scope": "ROOT_RELATIVE_METRIC",
            "units": "mm",
            "root_index": ROOT_INDEX,
            "windows_evaluated": windows_done,
            "frames_evaluated": frames_done,
            "max_windows": max_windows,
        },
        "metrics": {
            "mpjpe": _metric_summary(all_mpjpe),
            "p_mpjpe": _metric_summary(all_p_mpjpe),
            "acceleration_error": _metric_summary(all_acceleration),
            "per_joint_mpjpe_mm": {name: float(value) for name, value in zip(JOINT_NAMES, joint)},
            "upper_body_mpjpe_mm": float(np.mean(joint[7:])),
            "lower_body_mpjpe_mm": float(np.mean(joint[1:7])),
            "official_upstream_macro_by_action": {
                "mpjpe_mm": float(np.mean(official_mpjpe)),
                "p_mpjpe_mm": float(np.mean(official_p_mpjpe)),
                "acceleration_error_mm": float(np.mean(official_acceleration)),
                "per_joint_mpjpe_mm": {
                    name: float(value) for name, value in zip(JOINT_NAMES, official_joint)
                },
                "upper_body_mpjpe_mm": float(np.mean(official_joint[7:])),
                "lower_body_mpjpe_mm": float(np.mean(official_joint[1:7])),
            },
        },
        "by_action": action_report,
        "gates": {
            "worldpose_relative_3d_accuracy": "EVALUATED",
            "pitch_world_global_accuracy": "NOT_EVALUATED",
            "stage1_stage3_integration": "NOT_EVALUATED",
            "accuracy_claim_allowed": False,
            "reason": "This WorldPose lane evaluates root-relative 3D pose; it does not establish Stage-4 pitch-world placement.",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Stage 4 KASportsFormer on WorldPose")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--kasportsformer-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()
    report = run_benchmark(
        data_path=args.data.expanduser().resolve(), checkpoint=args.checkpoint.expanduser().resolve(),
        repo_root=args.kasportsformer_root.expanduser().resolve(), config_path=args.config.expanduser().resolve(),
        report_path=args.report.expanduser().resolve(), batch_size=args.batch_size,
        device=args.device, max_windows=args.max_windows,
    )
    print(json.dumps({"report": str(args.report.resolve()), "metrics": report["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
