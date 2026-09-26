from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OffsidePositionState:
    schema_version: str = "stage9-offside-position-1.0"
    stage_version: str = "stage9-offside-position-0.1.0"
    frame_index: Optional[int] = None
    status: str = "DEMO_BEST_EFFORT"
    mode: str = "BEST_EFFORT_DEMO"
    attack_direction: Optional[Dict[str, Any]] = None
    reference: Optional[Dict[str, Any]] = None
    toucher_track_id: Optional[str] = None
    attackers: List[Dict[str, Any]] = field(default_factory=list)
    opponents: List[Dict[str, Any]] = field(default_factory=list)
    others: List[Dict[str, Any]] = field(default_factory=list)
    ball: Optional[Dict[str, Any]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
