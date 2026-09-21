from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

PACKAGE_VERSION = "0.5.1"
STAGE6_VERSION = "stage6-ball-localization-0.5.1"


def runtime_provenance() -> Dict[str, Any]:
    """Small provenance block embedded in reports to prevent stale-package confusion."""
    package_root = Path(__file__).resolve().parent
    return {
        "package_version": PACKAGE_VERSION,
        "stage6_version": STAGE6_VERSION,
        "package_root": str(package_root),
        "version_module": str(Path(__file__).resolve()),
    }
