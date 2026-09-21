from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import os
import subprocess

from .contracts import FieldConverterBundle, FieldConverterV04Config
from .runtime_config import prepare_runtime_config


def build_field_converter_command(
    *,
    bundle: FieldConverterBundle,
    cfg: FieldConverterV04Config,
    raw_input_dir: str | Path,
    output_dir: str | Path,
    runtime_config_path: str | Path,
    source_fps: float,
    image_size: tuple[int, int],
) -> list[str]:
    cmd = [
        str(bundle.python_exe), "-m", "field_converter.inference",
        "--model-type", "tcn",
        "--config", str(Path(runtime_config_path).resolve()),
        "--checkpoint", str(bundle.checkpoint_path),
        "--input-dir", str(Path(raw_input_dir).resolve()),
        "--pitch-points", str(bundle.pitch_points_path),
        "--output-dir", str(Path(output_dir).resolve()),
        "--sequence", cfg.sequence_name,
        "--image-size", str(int(image_size[0])), str(int(image_size[1])),
        "--sam3d-sign", str(int(cfg.sam3d_sign)),
        "--world-alignment", str(cfg.world_alignment),
        "--box-normalization-min-size-px", str(float(cfg.box_normalization_min_size_px)),
        "--source-fps", str(float(source_fps)),
        "--target-fps", str(float(cfg.target_fps)),
        "--device", str(cfg.field_converter_device),
        "--no-csv",
    ]
    if cfg.field_converter_batch_size is not None:
        cmd.extend(["--batch-size", str(int(cfg.field_converter_batch_size))])
    if not cfg.save_field_converter_intermediate:
        cmd.append("--no-save-intermediate")
    return cmd


def run_field_converter(
    *,
    bundle: FieldConverterBundle,
    cfg: FieldConverterV04Config,
    raw_input_dir: str | Path,
    output_dir: str | Path,
    work_dir: str | Path,
    source_fps: float,
    image_size: tuple[int, int],
    timeout_s: Optional[int] = None,
) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    runtime_cfg = prepare_runtime_config(
        source_config=bundle.config_path,
        normalization_stats_path=bundle.normalization_stats_path,
        runtime_output_dir=work / "fc_runtime_outputs",
        destination=work / "field_converter_runtime.yaml",
    )
    cmd = build_field_converter_command(
        bundle=bundle,
        cfg=cfg,
        raw_input_dir=raw_input_dir,
        output_dir=output_dir,
        runtime_config_path=runtime_cfg,
        source_fps=source_fps,
        image_size=image_size,
    )
    env = os.environ.copy()
    if bundle.repo is not None:
        src = str(bundle.repo / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path = work / "field_converter_inference.log"
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    log_path.write_text(
        "COMMAND:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr,
        encoding="utf-8",
    )
    result = {
        "returncode": int(proc.returncode),
        "command": cmd,
        "runtime_config": str(runtime_cfg),
        "log_path": str(log_path),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        raise RuntimeError(
            "Official Field Converter inference failed. See log: " + str(log_path)
        )
    return result
