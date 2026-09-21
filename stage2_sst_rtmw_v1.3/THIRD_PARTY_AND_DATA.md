# Third-party and data policy

> Local handoff: [dataset locations and download policy](../docs_ban_giao/DATASET_PATHS.md). Check existing datasets outside `D:\NCKH` before downloading; `D:\GSR` and `D:\SoccerNet` are existing roots, not folders to recreate inside the project.

This Stage-2 package does not bundle model weights or restricted datasets.

## Runtime/reference components

- **SST**: football-specific object detector used as the primary entity detector.
- **RTMW WholeBody**: pretrained 2D whole-body pose model used as a shared raw tracking cue and handed to Stage 3.
- **SciPy Hungarian assignment**: target-window association.
- **OpenCV**: raw-frame extraction, image/video I/O and RTMW ONNX execution through the retained backend implementation.

## Evaluation data

SoccerNet is preferred for domain evaluation when the user's authorized local data is available. Credentials must be supplied only through the user's local secure mechanism when a download is actually required and must never be persisted in this package.

## Reproducibility

Record model file hashes and thresholds for every benchmark run. Do not infer that a checkpoint is the same merely from its filename.
