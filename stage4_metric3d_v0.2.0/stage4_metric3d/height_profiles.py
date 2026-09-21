from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict

from .wholebody import POSE23_NAMES


@dataclass(frozen=True)
class HeightProfile:
    profile_id: str
    reference_player_height_m: float
    fractions: Dict[str, float]
    source_path: Path
    raw: dict

    def z_m(self, keypoint_name: str, player_height_m: float) -> float:
        return float(self.fractions[keypoint_name]) * float(player_height_m)


def resolve_height_profile(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    name = str(value)
    if name.endswith(".json"):
        filenames = [name]
    elif name == "canonical-body-height-v1":
        filenames = ["canonical_body_v1.json"]
    else:
        filenames = [f"{name}.json"]

    roots = [
        Path.cwd() / "height_profiles",
        Path(__file__).resolve().parent.parent / "height_profiles",
    ]
    for root in roots:
        for filename in filenames:
            path = root / filename
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(f"Height profile not found: {value}")


def load_height_profile(value: str | Path) -> HeightProfile:
    path = resolve_height_profile(value)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "stage4-height-profile-1.0":
        raise ValueError(f"Unsupported height-profile schema: {raw.get('schema_version')}")
    fractions = {str(k): float(v) for k, v in (raw.get("height_fraction_by_keypoint") or {}).items()}
    missing = [name for name in POSE23_NAMES if name not in fractions]
    extra = sorted(set(fractions).difference(POSE23_NAMES))
    invalid = {name: value for name, value in fractions.items() if not 0.0 <= value <= 1.2}
    if missing or extra or invalid:
        raise ValueError(
            f"Invalid height profile: missing={missing}, extra={extra}, invalid={invalid}"
        )
    height = float(raw.get("reference_player_height_m"))
    if not 1.0 <= height <= 2.5:
        raise ValueError("height-profile reference_player_height_m must be in [1.0, 2.5]")
    return HeightProfile(
        profile_id=str(raw["profile_id"]),
        reference_player_height_m=height,
        fractions=fractions,
        source_path=path,
        raw=raw,
    )
