import numpy as np

from stage9_offside_position.visualization import (
    _defender_visible,
    render_overlay,
    resolve_track_visual_geometry,
)


def test_missing_screen_geometry_is_not_fabricated():
    g = resolve_track_visual_geometry(
        track_id="track_missing", frame_index=104, width=640, height=360,
        stage3_tracks={}, stage4_tracks={}, stage1=None,
    )
    assert g["bbox"] is None
    assert g["source"] is None


def test_overlay_does_not_draw_fake_bbox_when_geometry_missing():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    state = {
        "frame_index": 104, "mode": "BEST_EFFORT_DEMO",
        "reference": {},
        "attackers": [{"track_id":"track_missing","label":"OFFSIDE_POSITION","delta_q_m":0.4}],
        "opponents": [], "others": [],
    }
    out = render_overlay(frame, state, stage4={}, stage3={}, stage1=None, show_reference=False)
    # Header occupies rows 0..57. Nothing may be painted below it for a track
    # with no grounded screen geometry.
    assert np.count_nonzero(out[60:]) == 0


def test_only_critical_defenders_visible_by_default():
    assert _defender_visible({"rank":1}, False)
    assert _defender_visible({"rank":2}, False)
    assert not _defender_visible({"rank":3}, False)
    assert _defender_visible({"rank":6}, True)
