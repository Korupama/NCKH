from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class GameStateContext:
    schema_version: str = "1.0"
    stage_version: str = "stage7-game-state-0.1.0"
    frame_index: Optional[int] = None
    status: str = "UNRESOLVED"
    reasons: List[str] = field(default_factory=list)
    toucher: Optional[Dict[str, Any]] = None
    attacking_team_id: Any = None
    attack_direction: Optional[Dict[str, Any]] = None
    sets: Dict[str, List[str]] = field(default_factory=lambda: {
        "attackers": [],
        "opponents": [],
        "referees_excluded": [],
        "unknown_team_excluded": [],
        "inactive_excluded": [],
    })
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
