from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path, PurePosixPath
from typing import Any

from .soccernet_v3d import SoccerNetV3DCSV

HF_REPO_ID = "SoccerNet/SoccerNet_raw_HQ"
HF_REVISION = "frames-v3"
ARCHIVE_NAME = "Frames-v3.zip"


@dataclass(frozen=True)
class FramesV3ArchiveSpec:
    league: str
    season: str
    match: str

    @property
    def remote_path(self) -> str:
        # Hugging Face repository paths are POSIX paths even on Windows clients.
        return str(PurePosixPath(self.league, self.season, self.match, ARCHIVE_NAME))

    def local_path(self, root: str | Path) -> Path:
        return Path(root).expanduser().resolve() / self.league / self.season / self.match / ARCHIVE_NAME

    def to_dict(self, root: str | Path | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["remote_path"] = self.remote_path
        if root is not None:
            payload["local_path"] = str(self.local_path(root))
        return payload


def build_frames_v3_download_plan(
    csv_path: str | Path,
    split: str = "test",
    max_games: int | None = None,
) -> list[FramesV3ArchiveSpec]:
    """Return unique per-match frame archives needed by an SNv3D split.

    The Hugging Face ``frames-v3`` revision uses exactly:
    ``<league>/<season>/<match>/Frames-v3.zip``.
    """
    dataset = SoccerNetV3DCSV(csv_path)
    records = dataset.split(split)
    unique: dict[tuple[str, str, str], FramesV3ArchiveSpec] = {}
    for record in records:
        key = (record.league, record.season, record.match)
        unique.setdefault(key, FramesV3ArchiveSpec(*key))
    plan = list(unique.values())
    if max_games is not None:
        plan = plan[: max(0, int(max_games))]
    return plan


def summarize_frames_v3_plan(
    csv_path: str | Path,
    output_root: str | Path,
    split: str = "test",
    max_games: int | None = None,
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
) -> dict[str, Any]:
    dataset = SoccerNetV3DCSV(csv_path)
    records = dataset.split(split)
    plan = build_frames_v3_download_plan(csv_path, split=split, max_games=max_games)
    root = Path(output_root).expanduser().resolve()
    present = [spec for spec in plan if spec.local_path(root).is_file()]
    missing = [spec for spec in plan if not spec.local_path(root).is_file()]
    return {
        "repo_id": repo_id,
        "revision": revision,
        "layout": "<league>/<season>/<match>/Frames-v3.zip",
        "csv": str(Path(csv_path).expanduser().resolve()),
        "split": split,
        "records": len(records),
        "unique_match_archives": len(plan),
        "already_present": len(present),
        "missing": len(missing),
        "output_root": str(root),
        "remote_paths": [spec.remote_path for spec in plan],
        "missing_remote_paths": [spec.remote_path for spec in missing],
    }


def download_frames_v3_archives(
    csv_path: str | Path,
    output_root: str | Path,
    split: str = "test",
    *,
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
    max_games: int | None = None,
    max_workers: int = 4,
    token_env: str | None = None,
) -> dict[str, Any]:
    """Selectively download only the per-match Frames-v3.zip files used by a split.

    Authentication is delegated to the Hugging Face CLI cache by default. If
    ``token_env`` is supplied, the token is read from that environment variable;
    the token value is never printed or stored by this helper.
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it with: python -m pip install -U huggingface_hub"
        ) from exc

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = build_frames_v3_download_plan(csv_path, split=split, max_games=max_games)
    remote_paths = [spec.remote_path for spec in plan]
    if not remote_paths:
        raise ValueError(f"No Frames-v3 archives selected for split={split!r}")

    token = None
    if token_env:
        token = os.getenv(token_env)
        if not token:
            raise RuntimeError(f"Environment variable {token_env!r} is not set or is empty")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(root),
            allow_patterns=remote_paths,
            token=token,
            max_workers=max(1, int(max_workers)),
        )
    except HfHubHTTPError as exc:
        raise PermissionError(
            "Could not access SoccerNet_raw_HQ. Request/accept access on the Hugging Face dataset page, "
            "then run `hf auth login` (or provide --token-env pointing to a token environment variable) and retry."
        ) from exc

    missing_after = [spec for spec in plan if not spec.local_path(root).is_file()]
    return {
        "status": "COMPLETE" if not missing_after else "INCOMPLETE",
        "repo_id": repo_id,
        "revision": revision,
        "layout": "<league>/<season>/<match>/Frames-v3.zip",
        "csv": str(Path(csv_path).expanduser().resolve()),
        "split": split,
        "output_root": str(root),
        "requested_archives": len(plan),
        "available_after_download": len(plan) - len(missing_after),
        "missing_after_download": [spec.remote_path for spec in missing_after],
    }
