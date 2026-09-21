# Model and data provenance

## Production model

Primary model: **RTMW-L WholeBody, 384×288**, reused from Stage 2.

Official OpenMMLab ONNX SDK archive:

```text
https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/
rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip
```

The project does not redistribute the model weight. Use:

```powershell
python download_open_assets.py rtmw-l-384x288 --output-dir .\weights
```

Stage 3's OpenCV-DNN runner decodes the two SimCC axes, takes the x/y argmax locations and records the average x/y maxima as `raw_model_score`. It does **not** apply sigmoid/softmax calibration or call the value a probability.

## Stage-2 cache reuse

The preferred runtime path does not re-run RTMW. Stage 2 already computed WholeBody133 as a shared association cue. Stage 3 treats that cache as immutable upstream evidence and adds its own quality semantics.

## Optional same-model fallback

Fallback may run the exact same RTMW-L ONNX on controlled re-crops. It is intentionally not a second hidden model. Original Stage-2 raw evidence is retained when fallback is selected.

## Public datasets

### 3D Shot Posture Dataset (3DSP)

Repository:

```text
https://github.com/calvinyeungck/3D-Shot-Posture-Dataset
```

The repository contains `3dsp.zip`. Its README describes an annotated train portion with 20 crops × 200 shots and a test portion with 20 crops × 10 shots/tracklets. The images originate from SoccerNet broadcast video clips. The repository license is Apache-2.0.

The public JSON code contains the historical key typo `keypont_2d`; the Stage-3 adapter accepts both this spelling and `keypoint_2d`.

### COCO-WholeBody

Dataset/project:

```text
https://github.com/jin-s13/COCO-WholeBody
```

Official extended evaluator reference:

```text
https://github.com/jin-s13/xtcocoapi
```

Stage 3 does not redistribute COCO data.

## Research alternatives, not runtime dependencies

Potential later ablations:

- RTMW-X;
- RTMPose-L;
- DWPose;
- Sapiens WholeBody.

They are not introduced into the critical path until the RTMW-L Stage-3 baseline is measured and failure cases justify the additional complexity.
