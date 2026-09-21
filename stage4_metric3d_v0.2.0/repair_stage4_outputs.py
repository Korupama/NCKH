from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stage4_metric3d.output_semantics import build_stage5_handoff, recompute_selected_frame_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair Stage-4 selected-frame semantics without rerunning optimization")
    parser.add_argument("--input-state", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source = Path(args.input_state).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = json.loads(source.read_text(encoding="utf-8"))
    recompute_selected_frame_metrics(state)
    state["stage4_version"] = "stage4-metric3d-0.1.6"
    state["output_semantics"] = {
        "version": "1.1",
        "selected_frame_status_is_independent_of_track_optimizer_status": True,
        "repair_source_state": str(source),
        "repair_source_sha256": _sha256(source),
        "optimizer_rerun": False,
    }

    state_path = output_dir / "metric_pose_3d_state_fixed.json"
    handoff_path = output_dir / "stage5_handoff_v1_1.json"
    state["artifacts"] = {
        "metric_pose_3d_state": str(state_path),
        "stage5_handoff": str(handoff_path),
    }
    handoff = build_stage5_handoff(state, str(state_path))
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    handoff_path.write_text(json.dumps(handoff, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"metric_pose_3d_state": str(state_path), "stage5_handoff": str(handoff_path), "metrics": state["metrics"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
