from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage9_offside_position.core import build_offside_position_state


def _joints(xs):
    names = ["nose", "left_shoulder", "right_hip", "left_big_toe"]
    return [{"name": n, "xyz_world_m": [x, 0.0, 1.0 if "toe" not in n else 0.02], "valid": True} for n, x in zip(names, xs)]


def fixtures():
    base7 = {"frame_index": 104, "status": "VALID", "attack_direction": {"s": 1}, "toucher": {"track_id": "a0"}, "sets": {"attackers": ["a0", "a1", "a2"], "opponents": ["d1", "d2"]}}
    base8 = {"frame_index": 104, "status": "VALID", "attack_direction": {"s": 1}, "reference": {"goalward_q_m": 47.0, "X_world_m": 47.0, "source": "SECOND_LAST_OPPONENT"}}
    tracks = [
        {"track_id": "a0", "role": "player", "observations": [{"frame_index": 104, "joints_world": _joints([40,40,40,40])}]},
        {"track_id": "a1", "role": "player", "observations": [{"frame_index": 104, "joints_world": _joints([48,47.5,47.2,47.8])}]},
        {"track_id": "a2", "role": "player", "observations": [{"frame_index": 104, "joints_world": _joints([46,45.5,45.2,45.8])}]},
    ]
    stage4 = {"selected_frame": 104, "tracks": tracks}
    return [
        {"name": "ltr_basic", "s4": stage4, "s7": base7, "s8": base8, "gt": {"a0":"TOUCHER_EXCLUDED", "a1":"OFFSIDE_POSITION", "a2":"ONSIDE"}},
        {"name": "ignore_upstream_unresolved", "s4": stage4, "s7": {**base7, "status":"UNRESOLVED"}, "s8": {**base8, "status":"UNRESOLVED"}, "gt": {"a0":"TOUCHER_EXCLUDED", "a1":"OFFSIDE_POSITION", "a2":"ONSIDE"}},
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("oracle", nargs="?")
    p.add_argument("--output-dir", default="outputs/stage9_oracle")
    args = p.parse_args()
    rows=[]
    for f in fixtures():
        state=build_offside_position_state(f["s4"],f["s7"],f["s8"],best_effort=True).to_dict()
        pred={r["track_id"]:r["label"] for r in state["attackers"]}
        ok=pred==f["gt"]
        rows.append({"fixture":f["name"],"pass":ok,"prediction":pred,"gt":f["gt"]})
    report={"status":"PASS" if all(r["pass"] for r in rows) else "FAIL","oracle_logic_accuracy":sum(r["pass"] for r in rows)/len(rows),"target":1.0,"rows":rows}
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); (out/"stage9_oracle_benchmark.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
