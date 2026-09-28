import json
from pathlib import Path
from stage9_offside_position.webapp import build_demo_context, render_frame_bytes, make_handler
from .helpers import stage4_basic, stage7_basic, stage8_basic


def dump(p,obj):
    p.write_text(json.dumps(obj),encoding="utf-8")
    return str(p)


def test_web_context_and_jpeg(tmp_path: Path):
    p4=dump(tmp_path/"s4.json",stage4_basic(1))
    p7=dump(tmp_path/"s7.json",stage7_basic(1))
    p8=dump(tmp_path/"s8.json",stage8_basic(1))
    ctx=build_demo_context(stage4_path=p4,stage7_path=p7,stage8_path=p8)
    assert ctx.state["mode"]=="BEST_EFFORT_DEMO"
    data=render_frame_bytes(ctx,{"overlay":["1"]})
    assert data[:2] == b"\xff\xd8"
    assert make_handler(ctx) is not None
