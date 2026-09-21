from __future__ import annotations

import argparse

from ball_localization.visualization import render_minimap_from_state_json


def main() -> None:
    p = argparse.ArgumentParser(description="Re-render Stage-6 ball trajectory minimap without rerunning detection/localization")
    p.add_argument("--state-json", required=True, help="Existing ball_trajectory_state.json")
    p.add_argument("--output", default="ball_trajectory_minimap.png")
    p.add_argument("--scale", type=float, default=9.0)
    p.add_argument("--padding-px", type=int, default=30)
    p.add_argument("--no-dimension-labels", action="store_true")
    a = p.parse_args()
    out = render_minimap_from_state_json(
        a.state_json,
        a.output,
        scale=a.scale,
        padding_px=a.padding_px,
        show_dimension_labels=not a.no_dimension_labels,
    )
    print(out)


if __name__ == "__main__":
    main()
