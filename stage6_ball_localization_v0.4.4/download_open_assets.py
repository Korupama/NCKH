from __future__ import annotations
import argparse
from pathlib import Path
from urllib.request import Request,urlopen
from ball_localization.assets import ASSETS

def download(name:str,out_dir:Path)->Path:
    if name not in ASSETS:raise KeyError(name)
    asset=ASSETS[name]; out_dir.mkdir(parents=True,exist_ok=True); dest=out_dir/asset["filename"]
    req=Request(asset["url"],headers={"User-Agent":"stage6-ball-localization/0.2"})
    with urlopen(req) as r,dest.open("wb") as f:
        total=int(r.headers.get("Content-Length") or 0); done=0
        while True:
            chunk=r.read(1024*1024)
            if not chunk:break
            f.write(chunk); done+=len(chunk)
            if total:print(f"\r[{name}] {100*done/total:6.2f}% {done/1e6:.1f}/{total/1e6:.1f} MB",end="",flush=True)
            else:print(f"\r[{name}] {done/1e6:.1f} MB",end="",flush=True)
    print(); return dest

def main():
    p=argparse.ArgumentParser(description="Download official public SoccerNet-v3D release assets"); p.add_argument("asset",choices=sorted(ASSETS)); p.add_argument("--output-dir",default="assets"); a=p.parse_args(); print(download(a.asset,Path(a.output_dir)).resolve())
if __name__=="__main__":main()
