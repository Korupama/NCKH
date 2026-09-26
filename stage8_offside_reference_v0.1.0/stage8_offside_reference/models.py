from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OffsideReferenceState:
    schema_version: str = "stage8-offside-reference-1.0"
    stage_version: str = "stage8-offside-reference-0.1.0"
    frame_index: Optional[int] = None
    status: str = "UNRESOLVED"
    reasons: List[str] = field(default_factory=list)
    attack_direction: Optional[Dict[str, Any]] = None
    legal_body_policy: Dict[str, Any] = field(default_factory=dict)
    opponent_ranking: List[Dict[str, Any]] = field(default_factory=list)
    second_last_opponent: Optional[Dict[str, Any]] = None
    ball: Optional[Dict[str, Any]] = None
    reference: Optional[Dict[str, Any]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
