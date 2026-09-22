# Pose23@t0 data inventory — Phase 2

Status: `CONTRACT_READY_DATA_SOURCE_PENDING`

| Item | Current status | Decision |
|---|---|---|
| Versioned Pose23 manifest | Present, `0` samples | Keep as the only local contract artifact |
| Authorized broadcast/offside Pose23 annotations | Not mounted/selected | Do not invent labels or download restricted data |
| 3DSP | Downloaded locally; train `4000` images + `4000` posture JSON, test `200` images + `0` posture JSON | Use only as football-pose diagnostic B1; do not ingest as the task benchmark |
| COCO-WholeBody | Downloaded locally; validation `5000` images + `11004` person annotations | Use only as generic WholeBody diagnostic B2; do not ingest as the task benchmark |
| SoccerNet/GSR `gamestate-2024` valid | Downloaded locally after authorized access; `58` sequences / `43,500` frames / `707,766` human annotations; all label files report version `1.3` | Stored outside source control at `/Volumes/Lexar/Github/NCKH/third_party/SoccerNetGS`; use GameState boxes/roles as a broadcast-football diagnostic only until Pose23 annotation scope is authorized |
| Raw video/frame media | Not copied into this repository | Keep outside the repo; manifest stores reference paths only |

The current contract requires a source decision before samples are added:

- `source_name`;
- `license_or_permission`;
- `authorization_status`;
- source release/split and annotation ownership;
- whether the source permits derivative keypoint annotations.

The validator intentionally allows an empty template only with `--allow-empty`.
A benchmark run must reject an empty manifest.

## SoccerNet/GSR acquisition record

Downloaded release: `gamestate-2024`, split `valid`.

```text
/Volumes/Lexar/Github/NCKH/third_party/SoccerNetGS/
├── gamestate-2024/valid.zip  # downloaded archive
└── valid/
    └── SNGS-*/
        ├── Labels-GameState.json
        └── img1/
```

The Stage-2 GameState loader successfully read every sequence. Its labels provide image-space game-state annotations (including human boxes, roles and tracks), not the project-specific 23-keypoint ground truth required by Pose23@t0. No access credential is recorded here.

## Downloaded diagnostic assets

```text
data/3dsp/
data/coco_wholebody/annotations/coco_wholebody_val_v1.0.json
data/coco_wholebody/val2017/
data/coco_wholebody/archives/val2017.zip
```

COCO SHA256:

```text
coco_wholebody_val_v1.0.json
f8272e9c12f3a42457033ebc75da1167546edf1be5e2ffaf586d6ee97541ff6e

val2017.zip
4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05
```
