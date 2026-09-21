from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mmpose-root", required=True)
    args = p.parse_args()

    root = Path(args.mmpose_root).expanduser().resolve()
    project = root / "projects" / "rtmpose3d"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(project))

    import torch
    import mmcv
    import mmengine
    import mmpose
    import rtmpose3d  # noqa: F401
    from mmpose.apis import inference_topdown, init_model  # noqa: F401
    from mmpose.utils import register_all_modules

    register_all_modules()

    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mmcv": mmcv.__version__,
        "mmengine": mmengine.__version__,
        "mmpose": mmpose.__version__,
        "mmpose_root": str(root),
        "rtmpose3d_project": str(project),
        "rtmpose3d_import": "OK",
    }
    print(json.dumps(result, indent=2))
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available in this environment; use --device cpu or fix the PyTorch/CUDA installation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
