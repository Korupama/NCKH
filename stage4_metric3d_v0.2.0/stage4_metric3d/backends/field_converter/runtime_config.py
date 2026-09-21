from __future__ import annotations

from pathlib import Path
import json


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to prepare Field Converter runtime config") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Field Converter config must be a mapping: {path}")
    return data


def prepare_runtime_config(
    *,
    source_config: str | Path,
    normalization_stats_path: str | Path,
    runtime_output_dir: str | Path,
    destination: str | Path,
) -> Path:
    src = Path(source_config).expanduser().resolve()
    stats = Path(normalization_stats_path).expanduser().resolve()
    if stats.name != "normalization_stats.npz":
        raise ValueError("normalization stats file must be named normalization_stats.npz")
    data = _load_yaml(src)
    data["data_dir"] = str(stats.parent)
    data["output_dir"] = str(Path(runtime_output_dir).expanduser().resolve())
    # root_init_dir is not read for qualitative inference preprocessing, but
    # keeping auto avoids stale paths from a training machine.
    data["root_init_dir"] = "auto"
    dst = Path(destination).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        dst.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except Exception as exc:
        raise RuntimeError("Failed to write Field Converter runtime YAML") from exc
    return dst
