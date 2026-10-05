# Báo cáo benchmark Stage 3

## Kết luận

Stage 3 đã có benchmark pretrained RTMW-L với số liệu cụ thể trên 3DSP.
Kết quả được tách theo shot, không tách ngẫu nhiên theo frame:

| Split | Số shot | Số ảnh | PDJ@0.5 | AUC | Mean normalized error | Median normalized error |
|---|---:|---:|---:|---:|---:|---:|
| Development | 160 | 3.200 | 0.910022 | 0.665081 | 0.214363 | 0.118684 |
| Holdout | 40 | 800 | 0.903214 | 0.669479 | 0.214582 | 0.115142 |

Holdout là metric chính để kiểm tra non-regression. Stage 3 đạt benchmark
`PASS_WITH_RESEARCH_LIMITS`: pretrained pose accuracy đã được đo và artifact
đã được provenance hóa; hallucination/wrong-person rate trên broadcast thực tế
chưa được đo bởi 3DSP không chứa nhãn đúng/sai người theo Stage-2 track.

## Protocol

- Dataset: 3D Shot Posture Dataset (3DSP), SoccerNet broadcast crops.
- Split manifest: `splits/3dsp_internal_holdout_seed20260926.json`.
- Seed: `20260926`.
- Unit split: `shot_id`.
- Development: 160 shots / 3.200 ảnh / 44.800 joint observations.
- Holdout: 40 shots / 800 ảnh / 11.200 joint observations.
- Model: RTMW-L 384x288 ONNX, SHA256
  `bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054`.
- Runtime: CPU OpenCV DNN.
- Input: `288x384`, full-image bbox, `bbox_padding=1.25`, `crop_scale=1.0`.
- Output mapping: COCO-WholeBody133 sang H36M17 để tương thích annotation 3DSP.
- PDJ normalization: khoảng cách giữa tâm hai vai và tâm hai hông của GT.
- AUC: tích phân PDJ trong ngưỡng normalized error `[0.0, 0.5]`.
- Crop selection: chỉ dùng Stage-3 QA; GT chỉ được dùng ở bước tính metric cuối.

## Kết quả theo nhóm joint

| Nhóm | Development PDJ | Development AUC | Holdout PDJ | Holdout AUC |
|---|---:|---:|---:|---:|
| Head | 0.991250 | 0.714216 | 0.997500 | 0.730419 |
| Shoulder | 0.963438 | 0.695017 | 0.956250 | 0.697581 |
| Elbow | 0.890625 | 0.652013 | 0.868750 | 0.651850 |
| Wrist | 0.797031 | 0.563063 | 0.781875 | 0.579341 |
| Hip | 0.974063 | 0.638747 | 0.966875 | 0.621509 |
| Knee | 0.907969 | 0.707704 | 0.901250 | 0.705531 |
| Ankle | 0.845781 | 0.684810 | 0.850000 | 0.700119 |

Wrist và ankle là các nhóm yếu hơn tương đối; đây là slice ưu tiên cho các
ablation pretrained/crop/QA tiếp theo. Nhóm `Body` không có metric độc lập
trong protocol H36M17 hiện tại nên được giữ là `NaN`, không diễn giải thành
zero hoặc failure.

## So sánh holdout reference

So với artifact reference trước đó trên cùng 40 shot:

| Metric | Current holdout | Reference | Delta |
|---|---:|---:|---:|
| PDJ | 0.903214 | 0.903125 | +0.000089 |
| AUC | 0.669479 | 0.669442 | +0.000037 |
| Mean normalized error | 0.214582 | 0.214667 | -0.000085 |

Kết quả current holdout không bị regression theo cả ba gate. Đây là internal
shot holdout; aggregate train trước đó đã từng được quan sát, vì vậy không
được gọi là public generalization test.

## Artifact và provenance

- Development report:
  `benchmark_results/3dsp_phase_final_development_rtmw_l/3dsp_benchmark_summary.json`
  - SHA256: `12c1661b8c375fe79450768b8f1406de67518c2d95dd78627c51b9c15ea155c1`
- Holdout report:
  `benchmark_results/3dsp_phase_final_holdout_rtmw_l/3dsp_benchmark_summary.json`
  - SHA256: `88808460ccf8d4490bf9a28ba97dd777a0f4a8bf6e1df3976f489adfb9dce3c`
- Split manifest SHA256:
  `00cfa9c446b599db66e24f016e1ff25cb395dd67e10fdd5b66b4aef0029eec54`
- Final acceptance:
  `runs/phase8_final_acceptance.json`

## Giới hạn diễn giải

Benchmark này đo top-down pose accuracy khi ảnh/crop đã có sẵn; nó không đo:

- Stage 2 candidate recall;
- wrong-person/cross-person rate trên broadcast track;
- referee/player classification;
- hallucinated `VALID` precision của dữ liệu production;
- temporal consistency trên video thật.

Các metric trên chỉ được kết luận sau khi có benchmark độc lập tương ứng.
Không dùng human annotation hoặc pseudo-label để thay thế các metric đã nêu.
