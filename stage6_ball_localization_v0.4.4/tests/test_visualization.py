from pathlib import Path
import cv2
import numpy as np

from ball_localization.visualization import _render_metric_pitch, render_minimap_from_state_json


def test_metric_pitch_contains_penalty_and_goal_area_lines():
    img, pix = _render_metric_pitch(length_m=105.0, width_m=68.0, scale=8.0, padding_px=30, show_dimension_labels=False)
    def has_white_line(xw, yw):
        x, y = pix(xw, yw)
        patch = img[max(0,y-2):y+3, max(0,x-2):x+3]
        return bool((patch.mean(axis=2) > 225).any())

    # Left penalty-area inside corner: x=-52.5+16.5=-36.0, y=20.16.
    assert has_white_line(-36.0, 20.16)

    # Left goal-area inside corner: x=-52.5+5.5=-47.0, y=9.16.
    assert has_white_line(-47.0, 9.16)

    # Right-side symmetric penalty and goal area lines.
    assert has_white_line(36.0, -20.16)
    assert has_white_line(47.0, -9.16)


def test_existing_state_can_be_rerendered_without_inference(tmp_path: Path):
    state = Path('/mnt/data/ball_trajectory_state.json')
    if not state.is_file():
        return
    out = render_minimap_from_state_json(state, tmp_path / 'minimap.png')
    assert out.is_file()
    img = cv2.imread(str(out))
    assert img is not None
    assert img.shape[0] > 600
    assert img.shape[1] > 900
