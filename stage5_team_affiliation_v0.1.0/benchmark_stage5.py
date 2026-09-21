from __future__ import annotations

import argparse, json
from pathlib import Path
from stage5_team_affiliation.evaluation import evaluate_team_assignments


def main():
    ap = argparse.ArgumentParser(description="Evaluate Stage5 team assignments from track-level GT JSON")
    ap.add_argument("--state", required=True, help="team_affiliation_state.json")
    ap.add_argument("--gt", required=True, help="JSON mapping track_id -> 0/1")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    pred = {str(r["track_id"]): r.get("team_id") for r in state.get("tracks", [])}
    metrics = evaluate_team_assignments({str(k): int(v) for k,v in gt.items()}, pred)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")

if __name__ == "__main__":
    main()
