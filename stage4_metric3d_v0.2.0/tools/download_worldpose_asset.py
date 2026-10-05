from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests


def download_google_drive_file(file_id: str, output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    endpoint = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download"}
    response = session.get(endpoint, params=params, timeout=120)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("text/html"):
        match = re.search(r'name="uuid" value="([^"]+)"', response.text)
        if not match:
            raise RuntimeError("Google Drive download confirmation UUID was not found")
        params.update({"confirm": "t", "uuid": match.group(1)})
        response = session.get(endpoint, params=params, stream=True, timeout=120)
        response.raise_for_status()
    if response.headers.get("content-type", "").startswith("text/html"):
        raise RuntimeError("Google Drive returned an HTML warning instead of the binary asset")
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                handle.write(chunk)
    temporary.replace(output)
    print(f"downloaded={output}")
    print(f"bytes={output.stat().st_size}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    download_google_drive_file(args.file_id, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
