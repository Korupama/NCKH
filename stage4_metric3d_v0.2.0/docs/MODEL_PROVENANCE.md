# Stage 4 Model Provenance

## v0.5 production candidate

**SAM 3D Body** supplies the native MHR70 relative 3D pose, its corresponding projected 2D keypoints, and `pred_cam_t`. The Stage-4 worker verifies the installed official `sam_3d_body.metadata.mhr70` ordering before inference and supplies Stage-1 camera intrinsics to the estimator.

Stage 4 performs no learned root refinement of its own. The pitch-constrained step is a deterministic robust least-squares optimization over only the global translation vector.

## v0.4 optional backend

Field Converter-TCN remains in the package for future ablation. It is not the v0.5 default because the released checkpoint requires exact training normalization statistics that are referenced by upstream inference code but are not included in the public release available during this patch.

## Retained baselines

- v0.3 fixed-height image-ray / body-height-plane proxy.
- v0.2 legacy full-3D optimization and archived relative-pose initializers.
