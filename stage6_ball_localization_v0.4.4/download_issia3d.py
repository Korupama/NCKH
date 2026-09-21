from __future__ import annotations
import argparse
import urllib.request
from pathlib import Path

ASSETS = {
    "ISSIA-3D.csv": "https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/ISSIA-3D.csv",
    "issia_calibration.json": "https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/issia_calibration.json",
}

def main() -> int:
    parser = argparse.ArgumentParser(description="Download public SoccerNet-v3D ISSIA-3D oracle geometry assets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for name, url in ASSETS.items():
        target = out / name
        if target.is_file() and target.stat().st_size > 0 and not args.force:
            print(f"[EXISTS] {target}")
            continue
        print(f"[DOWNLOAD] {name}")
        urllib.request.urlretrieve(url, target)
        print(f"[READY] {target} ({target.stat().st_size} bytes)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
