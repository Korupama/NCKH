# Báo cáo tiến độ cải tiến Stage 3 — Pose 2D WholeBody

## 1. Tóm tắt

Stage 3 đã được cải tiến theo hướng reproducible, audit được và giữ nguyên
contract với Stage 2. Hệ thống hiện:

- sử dụng `COCO_WHOLEBODY_133` làm schema pose chuẩn;
- giữ nguyên `track_id`, frame quyết định `t0` và tọa độ ảnh raw;
- phân biệt raw RTMW SimCC score với probability;
- có QA cho độ đầy đủ, hình học và temporal consistency;
- có benchmark 3DSP, internal shot holdout và contract smoke với Stage 2;
- giữ RTMW-L làm backend production.

RTMW-X đã được thử nghiệm đầy đủ nhưng không vượt RTMW-L. Pseudo-label
self-training đã chuẩn bị được train artifact, nhưng chưa được dùng để claim
accuracy vì chưa có validation/test human-verified.

## 2. Phạm vi và invariant kỹ thuật

Các invariant sau vẫn được giữ nguyên:

| Hạng mục | Trạng thái |
|---|---|
| Canonical pose | `COCO_WHOLEBODY_133`, 133 keypoints |
| Tọa độ raw | `RAW_DISTORTED_PIXEL` |
| Identity | Không tạo identity mới; giữ `track_id` từ Stage 2 |
| Frame quyết định | Giữ nguyên `t0` từ Stage 2 |
| Raw x/y | Không bị temporal QA hoặc QA gate ghi đè âm thầm |
| Raw model score | RTMW SimCC evidence, không phải calibrated probability |
| Stage 2 handoff | Read-only; không sửa ba file handoff |
| Temporal estimate | Lưu riêng, không thay thế raw prediction |
| Production backend | RTMW-L; RTMW-X không được promote |

Stage 3 không phụ trách team affiliation, pitch/world coordinate, 3D lifting,
IFAB legal-body semantics hoặc quyết định offside.

## 3. Các cải tiến theo phase

### Phase 0–1 — Baseline và provenance

Đã đóng băng RTMW-L baseline trên toàn bộ 3DSP train split:

- 4.000 ảnh;
- model: RTMW-L 384×288;
- model SHA256: `bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054`;
- PDJ: `0.908607`;
- AUC: `0.665931`;
- mean normalized error: `0.214465`;
- median normalized error: `0.117928`.

### Phase 2 — Inference instrumentation

Đã bổ sung provenance cho mỗi inference attempt:

- input bbox;
- crop center/scale và input size;
- output tensor shapes;
- số keypoint finite/positive score;
- raw score min/median/max;
- metadata preprocessing và model provenance.

Đồng thời có test cho SimCC axis decoding, inverse coordinate mapping và
WholeBody133 schema.

### Phase 3 — Crop/preprocessing ablation

Đã đưa `bbox_padding` và `crop_scale` thành tham số benchmark. Giá trị mặc
định được giữ làm control; từng cấu hình ghi output riêng và không thay đổi
production path âm thầm.

### Phase 4 — Quality gate

Đã tách riêng:

1. pose coordinate accuracy;
2. pose acceptance/coverage cho downstream.

Quality gate hiện kiểm tra:

- body completeness;
- core/offside anatomy completeness;
- feet completeness;
- keypoint nằm trong bbox margin;
- bone-length sanity;
- trạng thái `VALID`, `DEGRADED`, `REJECTED`, `MISSING`;
- per-keypoint evidence state.

QA không được dùng để biến một prediction xấu thành ground truth tốt hơn.

### Phase 5 — Same-model re-crop ablation

Đã triển khai multi-crop QA selector với RTMW-L:

- control scale `1.0` luôn được giữ;
- candidate được chọn chỉ bằng engineering QA;
- không dùng ground truth để chọn crop từng ảnh;
- mọi attempt và selected scale được ghi provenance;
- raw upstream pose được bảo toàn nếu fallback được chọn.

Kết luận: multi-crop chưa đủ bằng chứng để thay đổi production default.

### Phase 6 — Temporal QA

Đã bổ sung:

- bbox-normalized temporal jitter;
- normalized second-difference/outlier diagnostics;
- nghi ngờ left/right swap;
- xử lý frame gap;
- missing keypoint handling;
- temporal estimate lưu riêng;
- kiểm tra boundary và frame `t0`.

Raw x/y không bị sửa khi bật temporal estimate.

### Phase 7 — Model backend ablation

RTMW-X 384×288 đã được export, kiểm tra graph và benchmark cùng protocol với
RTMW-L.

| Backend | Samples | PDJ | AUC | Mean error |
|---|---:|---:|---:|---:|
| RTMW-L full baseline | 4.000 | 0.908607 | 0.665931 | 0.214465 |
| RTMW-X full | 4.000 | 0.902571 | 0.655666 | 0.222620 |

Quyết định: **KEEP RTMW-L; DEFER RTMW-X promotion**.

RTMW-X chỉ được giữ trong báo cáo provenance; code production và requirements
đã quay về RTMW-L-only.

### Phase 8 — Pseudo-label self-training

Do chưa có WholeBody133 human annotation, chưa thể fine-tune chính thức.
Thay vào đó đã chuẩn bị Phase 8A exploratory:

- 58 SoccerNet-GSR sequences được split theo sequence:
  - train: 42;
  - validation: 9;
  - test: 8.
- 4.286 annotation tasks player/goalkeeper:
  - train: 2.986;
  - validation: 682;
  - test: 618.
- RTMW-L teacher xử lý 2.986 train tasks:
  - accepted pseudo-labels: 2.936;
  - rejected by QA: 50;
  - validation/test pseudo-labels: 0.

Pseudo-label artifact được đánh dấu:

```text
annotation_source = PSEUDO_LABEL
training_only = true
is_ground_truth = false
```

Chưa có student checkpoint và chưa có accuracy claim. Validation/test vẫn
phải dùng human-verified annotation.

### Phase 9 — Final integration và evaluation

Đã hoàn thành:

- shot-level internal holdout manifest;
- benchmark filter theo `shot_id`;
- final Stage 2 → Stage 3 contract smoke;
- full Stage 3 regression;
- final integration report.

Internal holdout:

- development: 160 shots;
- holdout: 40 shots / 800 ảnh;
- seed: `20260926`;
- RTMW-L PDJ: `0.903214` (current-code rerun);
- RTMW-L AUC: `0.669479` (current-code rerun);
- mean normalized error: `0.214582` (current-code rerun);
- median normalized error: `0.115142`.

Development split:

- 160 shots / 3,200 images;
- PDJ: `0.910022`;
- AUC: `0.665081`;
- mean normalized error: `0.214363`;
- median normalized error: `0.118684`.

The detailed Vietnamese benchmark report is
[`docs/STAGE3_BENCHMARK_REPORT.md`](STAGE3_BENCHMARK_REPORT.md).

Contract smoke:

- selected frame: `86`;
- candidate tracks: `10`;
- pose coverage tại `t0`: `1.0`;
- accepted coverage tại `t0`: `1.0`;
- valid coverage tại `t0`: `1.0`;
- foot coverage tại `t0`: `1.0`.

Regression cuối: **59 tests passed**.

## 4. Kết luận hiện tại

Stage 3 đã được cải tiến đáng kể về khả năng truy nguyên, QA, temporal
diagnostics, crop ablation, benchmark reproducibility và contract safety.

Backend production được giữ ổn định là RTMW-L. RTMW-X và pseudo-label
self-training chỉ là kết quả nghiên cứu/offline, chưa đủ cơ sở để thay đổi
production model hoặc tuyên bố fine-tuning accuracy.

## 5. Giới hạn cần nêu trong khoá luận

- 3DSP public test local không có posture JSON để làm test pose độc lập.
- Internal holdout là exploratory vì aggregate train đã được quan sát ở các
  phase trước; không gọi là public generalization test.
- COCO-WholeBody official evaluator chưa chạy được trong Windows environment;
  không dùng evaluator thay thế.
- Production fallback re-inference chưa validate vì legacy handoff không có
  source video truy cập được.
- Phase 8 chưa có human-verified WholeBody133 validation/test.
- Pseudo-label không được gọi là ground truth.

## 6. Artifact tham chiếu

- [Stage 3 implementation status](../IMPLEMENTATION_STATUS.md)
- [Final integration report](FINAL_INTEGRATION_REPORT.md)
- [Model backend ablation](MODEL_BACKEND_ABLATION.md)
- [Fine-tuning and pseudo-label status](MODEL_FINE_TUNING.md)
- [Stage 3 benchmark protocol](BENCHMARK.md)
- [Phase 9 shot holdout manifest](../splits/3dsp_internal_holdout_seed20260926.json)
