from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import cv2


def render_selected_frame(frame, track_records: Sequence[Mapping[str, Any]], output_path: str | Path):
    img = frame.copy()
    palette = {0: (255, 80, 40), 1: (40, 220, 255), None: (180, 180, 180)}
    for rec in track_records:
        bbox = rec.get("selected_frame_bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(lambda x: int(round(float(x))), bbox)
        team = rec.get("team_id")
        color = palette.get(team, (180,180,180))
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        label = f"{rec.get('track_id')} {rec.get('role')} T={team if team is not None else 'UNK'}"
        cv2.putText(img, label, (x1, max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out
