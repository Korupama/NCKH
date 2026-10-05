# Stage 4 Phase 0 — Baseline provenance and acceptance

- Generated (UTC): `2026-10-04T06:06:15.422088+00:00`
- Stage 4: `0.5.2`
- Git HEAD: `0c7058d1069e74dc2e268d620142e11dcde6ab87`
- Scope: `stage4_only`; upstream read-only: `True`

## Acceptance status

| Gate | Result |
|---|---|
| Implementation/tests | PASS_IMPLEMENTATION |
| Stage-4-only model development | ALLOWED |
| Pretrained inference assets | BLOCKED_MISSING_STAGE4_MODEL_ASSETS |
| Independent model accuracy | NOT_EVALUATED_NO_INDEPENDENT_BENCHMARK_RECORDED |
| Optional Stage-1/3 frame-104 integration | NOT_AVAILABLE_INTEGRATION_INPUTS |
| Metric accuracy | NOT_EVALUATED |
| Self-consistency diagnostics | NOT_AVAILABLE |
| Research accuracy frozen | False |

Accuracy is not claimed from synthetic tests, ground/contact residuals, pitch bounds, or reprojection residuals.
Stage-1/Stage-3 artifacts are required only for the separate integration baseline; they do not gate Stage-4-only backend development or dataset-based model evaluation.

## Test evidence

- Command: `PYTHONPATH=. C:\Users\Admin\anaconda3\envs\nckh-env\python.exe -m pytest -q tests tests_v04 tests_v05`
- Result: `PASS`; passed `91`, failed `0`

## Input and output availability

| Artifact | Exists | SHA256 / manifest SHA256 |
|---|---:|---|
| `stage3_state` | False | `NOT_AVAILABLE` |
| `stage1_camera_dir` | False | `NOT_AVAILABLE` |
| `source_video` | False | `NOT_AVAILABLE` |
| `frames_dir` | False | `NOT_AVAILABLE` |
| `sam3d_native_cache` | False | `NOT_AVAILABLE` |
| `sam3d_checkpoint` | False | `NOT_AVAILABLE` |
| `mhr_model` | False | `NOT_AVAILABLE` |
| `sam3d_repository` | False | `NOT_AVAILABLE` |
| `sam3d_pitch_refined` | False | `NOT_AVAILABLE` |
| `sam3d_direct` | False | `NOT_AVAILABLE` |

Missing optional frame-104 integration artifacts:
- `stage3_state`
- `stage1_camera_dir`
- `sam3d_native_cache`

## Selected-frame compatibility

- Requested baseline frame: `104`
- Stage-3 selected frame: `NOT_AVAILABLE`
- Stage-3 schema: `NOT_AVAILABLE`

A state from another sequence/frame is not accepted as a replacement for the frame-104 baseline.

## Recovery readiness

| Route | Ready | Missing |
|---|---:|---|
| Replay existing native cache | False | `stage3_state, stage1_camera_dir, sam3d_native_cache` |
| Stage-4-only pretrained inference | False | `sam3d_checkpoint, mhr_model, sam3d_repository_or_installed_package` |
| Generate integration cache (optional) | False | `stage3_state, stage1_camera_dir, stage4_pretrained_model_assets, source_video_or_frames_dir` |

The audit is fail-closed: it does not replace missing inputs with fixtures or rerun Stage 1/3. A missing integration baseline does not block independent Stage-4 work.

## Production source and configuration hashes

| File | SHA256 |
|---|---|
| `tools/stage4_phase0_audit.py` | `daaf69b071a5336e59d6217357fece594ba9ee484abfff03b107d1ee952e4450` |
| `run_sam3d_model_only.py` | `a6da9a2ff35a3ad647855e47b9749f6c633bce1e4d5e559e795d97d7a56df15e` |
| `evaluate_model_only.py` | `f154e3d2d20632bdf45ac36b4b7a6bd2ad52600ca504c85070a7820c8d7b142d` |
| `compare_model_only_systems.py` | `51b5029f1a5b6b885587b794a77dad56db25034b5c03edaf72bc1e23f3d38865` |
| `stage4_metric3d/model_only.py` | `dc684e6c5ea5507f0c0f067faae0ed9891c6ea2d4ef0a5b7105200f3434752c3` |
| `stage4_metric3d/model_only_refinement.py` | `fbd0d35b4b882d72c353ba0c97ebbae30edcb001e88a24f33f5b697063d6344a` |
| `stage4_metric3d/model_only_evaluation.py` | `f78e337773908bfc218396688521701f88833f8e690b16d30d6c41249f18feb6` |
| `stage4_metric3d/model_only_systems.py` | `5bf8823c431c5e08e7014ea6dc4ba0877ba69ec50551504e586e2b40abc11aa3` |
| `run_stage4.py` | `2899751157b8ba9aed9baf2546314f3bc710c86a06404f86284770d55d1dfb95` |
| `run_sam3d_body_worker.py` | `bf92a16b3da5206cf44c7a612fb1005b375c9a10bea32155c500eabd12cf095e` |
| `stage4_metric3d/camera.py` | `8e935adb2048efe274628167a621c612613fb1f114eadc39e1cf4be0a4391b25` |
| `stage4_metric3d/stage3_adapter.py` | `408defce31422fb80c8a6b9b1c122990631bfbdb78cd54581e785b61d53746b1` |
| `stage4_metric3d/schemas.py` | `840dd5812b9ecf81b033f915a01e0c7622e17d9d087bb17aa435189543fc3496` |
| `stage4_metric3d/output_semantics.py` | `ce69d5fae8d39f93be147676c42525033753bfb845e04358561d724b89009243` |
| `stage4_metric3d/backends/sam3d_pitch_refined/config.py` | `8db2fce9a964d42ba249e0b47d0b1818b05f72e8faa87e5d59452a145bc6fcb2` |
| `stage4_metric3d/backends/sam3d_pitch_refined/pipeline.py` | `807ce254aaeebe9d00f6816beac27d12319c057485e79f6bf386be661f09baf1` |
| `stage4_metric3d/backends/sam3d_pitch_refined/refiner.py` | `25e771ba3792b98f6d387827ed34f718a82e615216e987f350232f659ecdb348` |
| `stage4_metric3d/backends/sam3d_pitch_refined/ground_anchor.py` | `35ec1b13bcb83e03afd4337d7a9c82c81f968e4e915179c974f1ee51f856d3f4` |
| `stage4_metric3d/backends/sam3d_pitch_refined/geometry.py` | `a71b3273382e34db3a041b16ca371fc8fde334ca58da411d66313dc517807a20` |
| `stage4_metric3d/backends/sam3d_pitch_refined/quality.py` | `2218e9c61581280d07d9f15a5deceff3fd0ad047d3f4470de216d434bf940a96` |
| `stage4_metric3d/backends/sam3d_pitch_refined/metrics.py` | `b346f78fec1c4a8d6e4136e7761162cbb247a132fff761db32cb938dbf36f54b` |
| `stage4_metric3d/backends/sam3d_pitch_refined/visualization.py` | `a3e7634c9e0e5f4470148aef2a1059e6d1690e8118dba0e8d6cf34b75dc1dc87` |
| `stage4_metric3d/backends/sam3d_pitch_refined/cache.py` | `c2adfc6ca1703a2490758b3c9d4ef015eda0a4d4c198abbe782daba28933c8bc` |
| `stage4_metric3d/backends/sam3d_pitch_refined/joint_mapping.py` | `b2af5b7004e93bd00d66d5226dfaf889a98d1b9f0dd4d8a16786fb8d44723ee0` |
| `stage4_metric3d/backends/sam3d_pitch_refined/__init__.py` | `3822f81c856285e00f5c0965d08fb4faff8b6f75ee68fd961a818536736b39f0` |
| `tests_v05/test_refinement.py` | `aa4fce6009b188a79515d4981a0e88591b7d1ddd1ea546b2e641d2bc0442a441` |
| `tests_v05/test_ground_first.py` | `fa18b2e0771b8aaf4e6c5b6c358bc291e2e33955eec7b66dd13b8a9114554822` |
| `tests_v05/test_cache_and_mapping.py` | `31c1aa9451f4113a5143a9cfe48d90487514fb85dc3a37ad6bdffa60cdfee2fe` |
| `tests/test_output_semantics.py` | `25eacd813dc843e0cda72547dec8a88164e3b4280f3f8080761272ac0108ced2` |
| `tests/test_visualization.py` | `205d0b766ee944763b59cee597d49891aa93ead98704f849be0d210e26a436cc` |
| `tests/test_phase0_audit.py` | `f94ef5defd469211f41419212114f8e888a6245728a1ef4f0722e8004590a939` |
| `tests/test_model_only_manifest.py` | `cb4e66da5cebc122b9634ab2f21ace38978f1a83a9aacca182c3bb097043d414` |
| `tests/test_model_only_refinement.py` | `2767b6d3d1af869d6f20218e6fd1eb6d475b95834b32135cb91c0d322a141ae0` |
| `tests/test_model_only_evaluation.py` | `f05b246ef5c983ec26434ea0060bf701854b3c587bf805539cac214b123a6f67` |
| `tests/test_model_only_systems.py` | `17bad61ac87fb9167ba31298db249760564cbe47678781d849995f48951e8725` |
| `stage4_v05_config` | `9a280ddceea6b78b800e5569b426443947d7f9efd56107025abfe842eee8b886` |
| `mhr70_mapping` | `37f6fd90122f1a4269f419c7e1bb52972148bf928fbb6a7130182fd948886eb7` |

## Limitations

- The audit does not create Stage-1, Stage-3, SAM3D or metric-GT artifacts.
- Ground/contact and reprojection residuals are self-consistency diagnostics because they are optimization evidence.
- Metric accuracy and temporal refinement accuracy remain NOT_EVALUATED without independent metric ground truth and real cache inputs.
