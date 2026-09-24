# Báo cáo Tiến độ Nghiên cứu & Cập nhật Thực nghiệm M2 (NCKH_SoICT)

**Ngày báo cáo:** 24/09/2026  
**Dự án:** NCKH_SoICT - Phân loại đội bóng và theo dõi thực thể thể thao (SoccerNet-GSR)  
**Phạm vi cập nhật:** Stage 2 (`stage2_sst_rtmw_v1.3`) & Stage 5 (`stage5_team_affiliation_v0.1.0`)

---

## 1. Tổng quan những nội dung vừa được cập nhật

### 1.1. Script sinh Pose Cache chuyên dụng cho GSR
* **File mới:** [`generate_gsr_pose_cache.py`](file:///home/tondaiquoc/Workspace/Project/NCKH_SoICT/stage5_team_affiliation_v0.1.0/generate_gsr_pose_cache.py)
* **Chức năng:** Trích xuất offline toàn bộ 133 điểm khung xương WholeBody bằng mô hình **RTMW-L** (chạy qua OpenCV DNN) cho tất cả các tracklet hợp lệ trong 58 video sequences thuộc tập `valid` của bộ dữ liệu SoccerNet-GSR v1.3.
* **Cấu hình trích xuất:** Lấy mẫu mỗi 2 khung hình (`sample_every_n_frames=2`), giới hạn tối đa 30 mẫu/track (`max_samples_per_track=30`).
* **Đầu ra:** Đã sinh hoàn tất 58 thư mục cache và file `manifest.json` lưu tại `/home/tondaiquoc/benchmarks/gsr_pose_cache_oracle_full/`.

### 1.2. Cải tiến thuật toán kết hợp đặc trưng vùng (M2 Fusion)
* **File cập nhật:** [`gsr_benchmark.py`](file:///home/tondaiquoc/Workspace/Project/NCKH_SoICT/stage5_team_affiliation_v0.1.0/stage5_team_affiliation/gsr_benchmark.py)
* **Nội dung thay đổi:**
  * Tích hợp hàm `fuse_region_features` vào pipeline phân cụm màu sắc Stage 5.
  * Kết hợp đặc trưng màu sắc HSV của vùng thân trên (**Torso** - áo đấu) và thân dưới (**Lower-body** - quần thi đấu) theo tỷ lệ trọng số:
    $$\text{fused} = \text{normalize}(0.75 \times \text{torso} + 0.25 \times \text{lower})$$
  * Cơ chế dự phòng thích ứng (*fail-safe renormalization*): Nếu một trong hai vùng bị mất dấu hoặc không đạt độ tin cậy do che khuất, trọng số sẽ tự động dồn 100% về vùng còn lại.

### 1.3. Hoàn tất Benchmark M2 trên 58 sequences tập `valid`
* **File kết quả:** `/home/tondaiquoc/benchmarks/stage5_valid_stage5_color_m2_20260924/benchmark_summary.json`
* **Cập nhật tài liệu:**
  * [`MODEL_IMPROVEMENT_M2.md`](file:///home/tondaiquoc/Workspace/Project/NCKH_SoICT/stage2_sst_rtmw_v1.3/MODEL_IMPROVEMENT_M2.md)
  * [`IMPLEMENTATION_STATUS.md`](file:///home/tondaiquoc/Workspace/Project/NCKH_SoICT/stage5_team_affiliation_v0.1.0/IMPLEMENTATION_STATUS.md)

---

## 2. Bảng so sánh kết quả thực nghiệm M2 so với Baseline B0

Toàn bộ thực nghiệm được đánh giá theo giao thức ẩn nhãn đội (hidden-team oracle track benchmark) trên cùng 58 sequences (`valid` split):

| Chỉ số đánh giá (Metrics) | B0: `bbox-color` (Baseline) | M2: `stage5-color` (Torso + Lower 0.75/0.25) | Mức độ cải thiện ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Độ chính xác toàn diện (Overall Accuracy)** | **90.26%** | **91.08%** | **+0.82%** |
| **Độ phủ dữ liệu (Coverage)** | 93.29% | 93.94% | +0.65% |
| **Độ chính xác chọn lọc (Selective Accuracy)** | 96.75% | 96.95% | +0.20% |
| **Macro-F1** | 96.60% | 96.76% | +0.16% |
| **Độ chính xác cầu thủ sân (Outfield Accuracy)** | **95.63%** | **96.24%** | **+0.61%** |
| **Outfield Selective Accuracy** | **98.65%** | **99.19%** | **+0.54%** |
| **Độ chính xác thủ môn (Goalkeeper Accuracy)** | 10.39% | 14.29% | +3.90% |
| **Độ phủ thủ môn (Goalkeeper Coverage)** | 38.96% | 48.05% | +9.09% |
| **Tỷ lệ lẫn tạp trọng tài (Referee Contamination)** | **0.00%** | **0.00%** | Giữ nguyên (Tuyệt đối an toàn) |

---

## 3. Phân tích ý nghĩa học thuật và thực tiễn của các cập nhật

### 3.1. Đối với cầu thủ thi đấu trên sân (Outfield Players)
* **Ý nghĩa:** Việc cô lập vùng thân trên và thân dưới nhờ pose keypoints giúp loại bỏ hoàn toàn nhiễu từ nền sân cỏ xanh và các vùng biên ngoài bounding box. Khi kết hợp thêm 25% trọng số màu quần thi đấu, mô hình phân biệt tốt hơn trong các trường hợp hai đội có áo đấu mang tone màu gần nhau nhưng quần khác màu rõ rệt.
* **Kết quả:** Độ chính xác trên tập cầu thủ sân đạt **96.24%** (tổng thể) và đạt tới **99.19%** khi chỉ xét các dự đoán mà mô hình có độ tự tin cao (*selective accuracy*).

### 3.2. Đối với bài toán phân loại thủ môn (Goalkeepers)
* **Ý nghĩa thực tế:** Thủ môn mặc trang phục có màu sắc hoàn toàn tách biệt so với cả hai đội bóng. Thuật toán hiện tại sử dụng heuristic so sánh màu thân dưới của thủ môn với cụm màu thân dưới của hai đội đã phân loại.
* **Điểm nghẽn (*Bottleneck*):** Mặc dù độ chính xác tăng từ 10.39% lên 14.29% và tỷ lệ đưa ra dự đoán tăng lên 48.05%, con số này **vẫn còn rất thấp** và là failure mode lớn nhất của hệ thống. Nguyên nhân do thủ môn thường mặc quần khác màu hoàn toàn với đồng đội, hoặc màu quần trùng ngẫu nhiên với màu đối phương.
* **Kết luận:** Màu sắc quần chưa đủ tin cậy để gán đội cho thủ môn một cách độc lập; cần bổ sung các đặc trưng không gian (vị trí gần khung thành nào hơn trên sân, hoặc khoảng cách tới các hậu vệ cùng đội).

### 3.3. Về tính toàn vẹn của dữ liệu và quy trình nghiên cứu
* **Cách ly trọng tài (Referee Isolation):** Tỷ lệ lẫn tạp trọng tài đạt **0%**, khẳng định quy trình lọc trọng tài trước khi phân cụm 2 đội hoạt động hoàn toàn chính xác.
* **Tính chất thực nghiệm (Oracle Box vs End-to-End):**
  * Thực nghiệm M2 này được tiến hành trên bounding box và track_id chuẩn (Ground-Truth/Oracle inputs từ SoccerNet-GSR), chỉ ẩn nhãn đội.
  * Điều này chứng minh module trích xuất màu sắc và phân cụm theo vùng pose hoạt động hiệu quả, nhưng chưa phải là kết quả pipeline hoàn chỉnh từ đầu đến cuối (*End-to-End* bao gồm cả module phát hiện đối tượng và bám vết đa mục tiêu của Stage 1 & Stage 2).

---

## 4. Kế hoạch các bước tiếp theo

1. **Cải tiến phương pháp gán đội cho Thủ môn (Goalkeeper Affiliation):**
   * Tận dụng thông tin hình học sân và tọa độ khung thành (thủ môn thường di chuyển trong khu vực 16m50 của đội nhà trong phần lớn thời gian).
   * Sử dụng liên kết bám vết và tương quan không gian với cụm phòng ngự thay vì chỉ dựa vào màu quần.
2. **Đóng băng tham số (Parameter Freezing):**
   * Các siêu tham số (ngưỡng margin, trọng số kết hợp) cần được tinh chỉnh và đóng băng trên tập `train`, sau đó chỉ đánh giá kiểm chứng 1 lần duy nhất trên tập `valid` và `test` theo đúng chuẩn mực nghiên cứu.
3. **Liên kết Stage 2 & Stage 5 End-to-End:**
   * Sau khi kiểm tra tracker chuyển động của Stage 2 (motion-aware target-window association), đưa trực tiếp các tracklet dự đoán sang Stage 5 để đánh giá độ suy giảm hiệu năng trong môi trường thực tế không có oracle box.
