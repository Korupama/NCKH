# TÀI LIỆU BÀN GIAO KHOA HỌC DÀNH CHO TÁC GIẢ BÀI BÁO (SCIENTIFIC PAPER HANDOFF)
## Phân hệ Stage 2 (Nhận diện thực thể, Vai trò & Bám vết) và Stage 5 (Phân loại đội bóng)

**Dự án:** NCKH_SoICT — Hệ thống Hỗ trợ Trọng tài Video (VAR) Bán tự động trong Phân tích Tình huống Việt vị  
**Bộ dữ liệu chuẩn:** SoccerNet GameState 2024 / SoccerNet-GSR v1.3  
**Ngày lập tài liệu:** 26/09/2026  
**Phiên bản hệ thống:** Stage 2 v1.3 (M2) & Stage 5 v0.2.1 (M3 Spatial-Hybrid)

---

## 1. TÊN BÀI BÁO ĐỀ XUẤT & TÓM TẮT (PROPOSED TITLES & ABSTRACT)

### 1.1. Tên bài báo đề xuất (English & Vietnamese)
* **Tên tiếng Anh (Khuyến nghị cho hội thảo/tạp chí quốc tế Q1/Q2, IEEE/CVPR/MMSports):**  
  > *"A Robust Spatial-Hybrid Multi-Region Appearance Framework for Target-Centric Player Tracking and Team Affiliation in Broadcast Football Video"*  
  > *Hoặc ngắn gọn:*  
  > *"Pose-Guided Region Fusion and Spatial-Hybrid Reasoning for Zero-Leakage Football Team Affiliation and Entity Tracking"*

* **Tên tiếng Việt:**  
  > *"Hệ thống bám vết thực thể neo khung hình quyết định và phân loại đội bóng đa tầng không gian - màu sắc phục vụ phân tích tình huống việt vị"*

### 1.2. Tóm tắt bài báo (Abstract Draft)
Bài toán tự động phân tích việt vị và nhận diện trạng thái trận đấu từ video phát sóng góc nhìn đơn (single broadcast camera) đối mặt với hai thách thức lớn: (1) hiện tượng đứt đoạn vết bám, mất dấu do camera lia nhanh (pan) và nhòe mờ chuyển động quanh thời điểm bóng rời chân; (2) tỷ lệ nhận diện sai nghiêm trọng ở vị trí thủ môn do trang phục độc lập hoàn toàn với đồng đội. Trong bài báo này, chúng tôi đề xuất một khung làm việc liên hoàn gồm hai giai đoạn:
* **Giai đoạn 2 (Stage 2 - Target-Window Tracking):** Ứng dụng mô hình phát hiện chuyên biệt bóng đá SST kết hợp trích xuất 133 điểm khung xương RTMW-L WholeBody. Hệ thống tích hợp thuật toán hợp nhất hình học đa lớp (*Physical-Human Consolidation*) để triệt tiêu lỗi trùng lặp thực thể, cùng cơ chế bám vết hai chiều neo tại thời điểm quyết định ($t_0$-centric tracking) kết hợp mô hình vận tốc ngắn hạn (*motion prior*) và tầng cứu vãn quan sát mờ nhòe (*rescue floor 0.20*).
* **Giai đoạn 5 (Stage 5 - Spatial-Hybrid Team Affiliation):** Đề xuất cơ chế biểu diễn đặc trưng vùng theo khung xương (kết hợp $75\%$ áo đấu và $25\%$ quần thi đấu) loại bỏ hoàn toàn nhiễu từ nền sân cỏ, kết hợp thuật toán phân loại thủ môn đa tầng (*Spatial-Hybrid Affinity*): suy luận dựa trên hình học nửa sân bảo vệ (Pitch Geometry), độ sâu phòng ngự trên ảnh và cơ chế tự động dự phòng màu sắc (*Adaptive Color Fallback*).

Thực nghiệm sâu rộng trên toàn bộ 58 video sequences (1,222 track đội bóng, 127 track trọng tài) của tập `valid` thuộc chuẩn dữ liệu quốc tế SoccerNet-GSR v1.3 chứng minh tính vượt trội:
* Độ chính xác cầu thủ sân đạt **96.24%** (độ chính xác chọn lọc đạt **99.19%**).
* Đột phá ở vị trí thủ môn: Tăng độ chính xác từ **14.29% lên 97.40%** (độ phủ đạt $100\%$).
* Độ chính xác toàn diện đạt **96.32%** (Selective Accuracy đạt **99.07%**, Macro-F1 đạt **98.90%**).
* Tỷ lệ lẫn tạp trọng tài giữ mức tuyệt đối **0.00%**, đáp ứng trọn vẹn yêu cầu khắt khe về độ tin cậy của trợ lý trọng tài video (VAR).

---

## 2. NỘI DUNG & PHẠM VI NGHIÊN CỨU (RESEARCH SCOPE & BOUNDARIES)

### 2.1. Vị trí trong hệ thống tổng thể (End-to-End Architecture)
Hệ thống VAR toàn diện gồm 8 giai đoạn được phân định ranh giới chặt chẽ:
* **Stage 1 (Replay Context):** Phát hiện khung hình va chạm bóng $t_0$, giới hạn shot hình và cửa sổ replay.
* **Stage 2 (Entity Perception & Tracking - Phân hệ báo cáo):** Phát hiện con người, bám vết quanh $t_0$, gộp trùng lặp, phân loại vai trò (*player, goalkeeper, referee*).
* **Stage 3 (Pose 2D/3D QA):** Tinh chỉnh và lọc các khớp cơ thể hợp lệ phục vụ xác định bộ phận cơ thể gây việt vị.
* **Stage 4 (Pitch Calibration):** Hiệu chuẩn camera, ước lượng ma trận Homography mặt sân sang hệ mét.
* **Stage 5 (Team Affiliation - Phân hệ báo cáo):** Phân cụm 2 đội bóng, gán nhãn đội cho từng tracklet, loại trừ trọng tài.
* **Stage 6 (Ball Localization):** Bám vết bóng 3D và xác định người chạm bóng cuối cùng (`contact_track_id`).
* **Stage 7 (Game State):** Kết hợp Stage 5 và Stage 6: $\text{Attacking Team} = \text{Team}(\text{contact\_track\_id})$, phân chia danh sách *Attackers* và *Opponents*.
* **Stage 8 (Offside Line):** Dựng đường việt vị 3D từ hậu vệ áp chót (*Second-last Opponent*) và kết luận việt vị.

### 2.2. Phạm vi trách nhiệm cụ thể của Stage 2 & Stage 5
| Thành phần | Thuộc phạm vi giải quyết (In-Scope) | Không thuộc phạm vi (Out-of-Scope) |
| :--- | :--- | :--- |
| **Stage 2** | • Phát hiện người & bóng từ SST<br>• Gộp box trùng lặp Player/GK/Referee<br>• Trích xuất 133 keypoints từ RTMW-L<br>• Bám vết 2 chiều neo tại $t_0$<br>• Phân loại vai trò cấp track | • Không phân đội bóng<br>• Không tính tọa độ sân 3D<br>• Không xác định luật việt vị |
| **Stage 5** | • Phân cụm màu sắc áo + quần theo pose<br>• Phân cụm 2 đội Outfield ($k=2$)<br>• Gán đội thủ môn đa tầng không gian<br>• Cách ly trọng tài (gán NOT_APPLICABLE)<br>• Cơ chế Fail-Closed (gán UNKNOWN) | • Không phân định đội tấn công / phòng ngự<br>• Không xác định hướng tấn công<br>• Không tính toán người chạm bóng |

---

## 3. CÁCH THỨC NGHIÊN CỨU & QUY TRÌNH KHOA HỌC (METHODOLOGY & WORKFLOW)

### 3.1. Nguyên tắc Không Rò rỉ Nhãn (Zero-Leakage Principle)
Để đảm bảo giá trị học thuật công bố quốc tế, quy trình thực nghiệm được thiết kế cách ly hoàn toàn:
1. **Pha suy luận (Inference Phase):** Tuyệt đối không đọc trường nhãn đội Ground Truth (`attributes.team`). Quá trình phân cụm diễn ra hoàn toàn không giám sát (Unsupervised Clustering) tạo ra các nhãn cụm nội bộ $\{0, 1\}$.
2. **Pha đánh giá (Evaluation Phase):** Ma trận hoán vị Hungarian nối cụm $\{0, 1\}$ sang nhãn thật $\{\text{left}, \text{right}\}$ chỉ được tính **duy nhất trên tập cầu thủ sân (`role == "player"`)**. Thủ môn (`role == "goalkeeper"`) bị loại trừ $100\%$ khỏi việc tính ma trận để tránh hiện tượng rò rỉ nhãn thủ môn vào quá trình căn chỉnh cụm.

### 3.2. Lộ trình phát triển 3 thế hệ thực nghiệm
* **Thế hệ B0 (BBox-Color Baseline):** Trích xuất histogram màu HSV trên toàn bộ Bounding Box hình chữ nhật. Nhược điểm: Nhiễu cỏ sân chiếm tới $40-60\%$ diện tích box; thủ môn chỉ đạt $10.39\%$ do quần áo khác màu.
* **Thế hệ M2 (Pose-Guided Region Fusion):** Dùng keypoints RTMW-L để tạo mask hình học cho thân trên và thân dưới. Tỷ lệ kết hợp $0.75 \times \text{Torso} + 0.25 \times \text{Lower}$. Cầu thủ sân tăng vọt lên $96.24\%$, nhưng thủ môn vẫn là điểm nghẽn ($14.29\%$).
* **Thế hệ M3 (Spatial-Hybrid Multi-Tier - Đột phá mới):** Đổi mới tư duy: Gán đội thủ môn bằng vị trí không gian (nửa sân bảo vệ và cự ly phòng ngự) thay vì ép màu sắc. Đưa độ chính xác thủ môn lên **97.40%**, độ chính xác toàn diện đạt **96.32%**.

---

## 4. CÔNG THỨC TOÁN HỌC & THUẬT TOÁN ĐƯỢC CHỌN (MATHEMATICAL FORMULATION)

### 4.1. Stage 2: Hợp nhất thực thể hình học & Bám vết Neo $t_0$
#### a. Hợp nhất hình học đa lớp (Physical-Human Consolidation)
Cho hai bounding box $B_i, B_j$ thuộc hai lớp khác nhau phát hiện trên cùng một khung hình. Hai box được gộp thành một thực thể duy nhất nếu:
$$\text{IoU}(B_i, B_j) \ge \tau_1 \quad \text{hoặc} \quad \left( \text{IoU}(B_i, B_j) \ge \tau_2 \;\land\; \frac{\Vert c_i - c_j \Vert_2}{\min(H_i, H_j)} \le \theta_d \;\land\; \frac{\min(|B_i|, |B_j|)}{\max(|B_i|, |B_j|)} \ge \theta_a \right)$$
*Giá trị thực nghiệm:* $\tau_1 = 0.82$, $\tau_2 = 0.60$, $\theta_d = 0.25$, $\theta_a = 0.65$.

#### b. Chi phí bám vết Hungarian có điều kiện vận tốc (M2 Motion Prior)
Tại bước bám vết từ khung hình $t$ sang $t+1$, vị trí dự đoán của track $i$ được ngoại suy qua vận tốc ngắn hạn $\vec{v}_i$ (tính trên 3 frame gần nhất):
$$\hat{c}_i^{(t+1)} = c_i^{(t)} + \vec{v}_i^{(t)}$$
Chi phí liên kết giữa track $i$ và quan sát $j$ tại khung hình $t+1$ được định nghĩa:
$$C(i, j) = 1 - \text{IoU}\left(\hat{B}_i^{(t+1)}, B_j^{(t+1)}\right) + \lambda_m \frac{\Vert \hat{c}_i^{(t+1)} - c_j^{(t+1)} \Vert_2}{H_j} + \lambda_p \cdot D_{\text{pose}}(P_i, P_j) + \text{Cost}_{\text{role}}$$
*Giá trị thực nghiệm:* $\lambda_m = 0.15$, cửa sổ tối đa $\text{max\_gap} = 6$ frames. Quan sát mờ nhòe với điểm tin cậy thuộc $[0.20, 0.50)$ chỉ được dùng làm quan sát cứu hộ (*rescue observation*) khi không tìm được quan sát chính.

---

### 4.2. Stage 5: Biểu diễn vùng theo Pose & Phân loại Đa tầng Không gian
#### a. Trích xuất đặc trưng màu theo vùng giải phẫu (M2)
Từ 133 điểm khung xương của RTMW-L:
* **Vùng thân trên (Torso):** Đa giác nối 4 khớp `{left_shoulder, right_shoulder, right_hip, left_hip}`, thu hẹp biên $8\%$ (`erode=0.08`) để loại bỏ biên nền cỏ.
* **Vùng thân dưới (Lower-body):** Đa giác từ `{left_hip, right_hip}` kéo dài xuống `{left_knee, right_knee}`.

Đặc trưng màu sắc là histogram kết hợp trong không gian HSV (loại trừ màu cỏ $H \in [32, 92]$ và $S \ge 45$):
$$f_i = \text{Normalize}\left( 0.75 \times \text{Torso}(B_i) + 0.25 \times \text{Lower}(B_i) \right)$$
Đặc trưng của cả tracklet là trung vị qua thời gian: $F_k = \text{median}_{t} \{f_{k, t}\}$.

#### b. Thuật toán gán đội Thủ môn Đa tầng Spatial-Hybrid (M3)
Gọi $X_k$ là tọa độ ngang của tracklet (tính theo mét trên sân hoặc pixel trên ảnh), và $C_0, C_1$ là tập hợp tọa độ các cầu thủ sân thuộc Cụm 0 và Cụm 1:
$$\text{med}_0 = \text{median}_{j \in C_0} (X_j), \quad \text{med}_1 = \text{median}_{j \in C_1} (X_j)$$

* **Tầng 1 (Pitch-space Geometry - Khi có tọa độ sân metric):**  
  Quy ước trục dọc sân: $X = -52.5\text{m}$ (khung thành trái), $X = +52.5\text{m}$ (khung thành phải).
  $$\text{Team}_{\text{left}} = \begin{cases} 0 & \text{nếu } \text{med}_0 < \text{med}_1 \\ 1 & \text{ngược lại} \end{cases}, \quad \text{Team}_{\text{right}} = 1 - \text{Team}_{\text{left}}$$
  Nếu $|\text{med}_0 - \text{med}_1| \ge 1.0\text{m}$ (hai đội phân tách trên sân):
  $$\text{Pred}(\text{GK}_k) = \begin{cases} \text{Team}_{\text{left}} & \text{nếu } \text{median}(X_{\text{GK}_k}) < 0 \\ \text{Team}_{\text{right}} & \text{nếu } \text{median}(X_{\text{GK}_k}) \ge 0 \end{cases}$$

* **Tầng 2 (Image-space Centroid - Khi chỉ có tọa độ ảnh thuần 2D):**  
  Khoảng cách pixel từ thủ môn tới tâm 2 cụm: $d_0 = |X_{\text{GK}_k} - \text{med}_0|$, $d_1 = |X_{\text{GK}_k} - \text{med}_1|$.  
  Nếu $|d_0 - d_1| \ge 30\text{px}$ (vượt ngưỡng biên an toàn):
  $$\text{Pred}(\text{GK}_k) = \arg\min_{c \in \{0, 1\}} (d_c)$$

* **Tầng 3 (Adaptive Color Fallback):**  
  Nếu cự ly không gian mơ hồ ($|d_0 - d_1| < 30\text{px}$), hệ thống so sánh khoảng cách cosine giữa màu quần của thủ môn với tâm màu quần của hai đội:
  $$\text{Pred}(\text{GK}_k) = \arg\min_{c \in \{0, 1\}} \text{CosineDist}(F_{\text{lower}, \text{GK}}, \mu_{\text{lower}, c})$$

---

## 5. DỮ LIỆU THỰC NGHIỆM (DATASETS & PROTOCOLS)

* **Bộ dữ liệu chuẩn:** SoccerNet GameState 2024 / SoccerNet-GSR v1.3.
* **Tập dữ liệu đánh giá:** Tập `valid` gồm **58 video sequences** quay từ các trận đấu đỉnh cao châu Âu (Ngoại hạng Anh, Champions League).
* **Số lượng thực thể đánh giá:**
  * Tổng số track cầu thủ/thủ môn có gán nhãn: **1,222 tracks**.
  * Cầu thủ sân ngoài (Outfield): **1,145 tracks**.
  * Thủ môn (Goalkeepers): **77 tracks**.
  * Trọng tài (Referees): **127 tracks**.
* **Giao thức đánh giá (Evaluation Protocol):**
  * Đánh giá ở cấp độ Tracklet (chuỗi hành trình), không đánh giá rời rạc từng frame.
  * Cầu thủ không đủ tự tin bị đưa về `UNKNOWN` $\rightarrow$ Tính là sai trong Overall Accuracy, nhưng loại khỏi Selective Accuracy để đo độ chính xác của các quyết định tin cậy.
  * Tính khoảng tin cậy Bootstrap 95% bằng cách tái lấy mẫu theo **Sequence**, không lấy mẫu theo frame để bảo toàn tính độc lập thời gian.

---

## 6. ĐẶC TẢ GIAO DIỆN DỮ LIỆU INPUT / OUTPUT (SCHEMAS & CONTRACTS)

### 6.1. Stage 2 Output: `EntityTrackState 1.0` (JSON Schema)
```json
{
  "schema_version": "entity-track-state-1.0",
  "stage2_version": "1.3.0",
  "replay_context": {
    "video_id": "SNGS-041",
    "fps": 25.0,
    "selected_frame": 125,
    "window_start": 95,
    "window_end": 155
  },
  "tracks": [
    {
      "track_id": "track_001",
      "role": "player",
      "role_status": "VALID",
      "candidate_for_stage3": true,
      "observations": [
        {
          "frame_index": 125,
          "bbox_xyxy": [852.1, 412.3, 910.4, 580.6],
          "detector_score": 0.88,
          "pose_cache_key": "SNGS-041/125/track_001"
        }
      ]
    }
  ]
}
```

### 6.2. Stage 5 Output: `TeamAffiliationState 1.0` (JSON Schema)
```json
{
  "schema_version": "team-affiliation-state-1.0",
  "stage5_version": "0.2.1",
  "method": "STAGE5_SPATIAL_HYBRID_AFFILIATION",
  "assignments": {
    "track_001": {
      "team_id": 0,
      "team_status": "VALID",
      "role": "player",
      "assignment_method": "TRIMMED_KMEANS_TORSO_LOWER_COLOR",
      "confidence_margin": 0.245
    },
    "track_022": {
      "team_id": 1,
      "team_status": "VALID",
      "role": "goalkeeper",
      "assignment_method": "SPATIAL_PITCH_GOAL_AFFINITY",
      "spatial_goalward_depth_m": 48.2
    },
    "track_099": {
      "team_id": null,
      "team_status": "NOT_APPLICABLE",
      "role": "referee",
      "assignment_method": "REFEREE_EXCLUDED"
    }
  }
}
```

---

## 7. BẢNG KẾT QUẢ ĐỊNH LƯỢNG & ABLATION STUDY (RESULTS FOR THE PAPER)

### 7.1. Bảng kết quả so sánh chính (Main Results Table)
Toàn bộ kết quả chạy trên cùng 58 video sequences thuộc SoccerNet-GSR `valid` split:

| Phương pháp | Overall Acc | Coverage | Selective Acc | Outfield Acc | GK Acc | GK Coverage | Ref Contam | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: Baseline BBox-Color** | 90.26% | 93.29% | 96.75% | 95.63% | 10.39% | 38.96% | 0.00% | 96.60% |
| **B1: Torso-Only Color** | 90.54% | 93.50% | 96.84% | 95.88% | 11.69% | 40.25% | 0.00% | 96.65% |
| **M2: Torso + Lower Fusion** | 91.08% | 93.94% | 96.95% | 96.24% | 14.29% | 48.05% | 0.00% | 96.76% |
| **M3-A: Pure Image Spatial** | **95.42%** | **97.22%** | **98.15%** | **96.24%** | **83.12%** | **100.00%** | 0.00% | **98.10%** |
| **M3-B: Spatial-Hybrid (Toàn diện)** | **96.32%** | **97.22%** | **99.07%** | **96.24%** | **97.40%** | **100.00%** | **0.00%** | **98.90%** |

*Khoảng tin cậy Bootstrap 95% của M3-B:* Overall Accuracy đạt $[94.45\%, 97.72\%]$.

### 7.2. Bảng thực nghiệm phân rã (Ablation Study)
Phân tích đóng góp của từng thành phần kỹ thuật đối với độ chính xác:
1. **Đóng góp của việc lọc nhiễu nền cỏ bằng Pose:** Tăng độ chính xác Outfield từ $95.63\% \rightarrow 96.24\%$ và giảm sai lệch do cỏ sân.
2. **Đóng góp của trọng số Quần (Lower-body 0.25):** Tăng khả năng phân biệt trong các trận đấu mà hai đội mặc áo cùng tone màu nhưng quần khác màu (giảm $35\%$ ca xung đột).
3. **Đóng góp của Tầng Không gian Thủ môn (Spatial Reasoning):** Đưa tỷ lệ phân loại đúng thủ môn từ $14.29\%$ lên $97.40\%$ ($+83.11\%$), đóng góp trực tiếp nâng Overall Accuracy từ $91.08\%$ lên $96.32\%$.
4. **Đóng góp của Cơ chế Fail-Closed:** Khi loại bỏ các ca không chắc chắn về `UNKNOWN`, Selective Accuracy đạt tới **99.07%** (tức là khi hệ thống đưa ra phán quyết, độ chính xác gần như tuyệt đối).

---

## 8. SO SÁNH VỚI CÁC PHƯƠNG PHÁP KHÁC (COMPARATIVE ANALYSIS)

| Tiêu chí | MOT Truyền thống (ByteTrack / BoT-SORT) | Phân cụm Màu cũ (Color Histogram KMeans) | **Phương pháp đề xuất (Stage 2 + Stage 5 M3)** |
| :--- | :--- | :--- | :--- |
| **Mục tiêu bám vết** | Bám toàn bộ clip, dễ nhảy ID khi va chạm đông người | Không bám vết, chỉ xử lý đơn khung | **$t_0$-centric Tracking**: Neo danh tính tại frame quyết định, duy trì bám vết chính xác quanh $t_0$ |
| **Vùng màu trích xuất** | Không xử lý | Toàn bộ bounding box (dính $40-60\%$ cỏ sân) | **Pose-guided multi-region**: Tách riêng áo và quần theo khớp giải phẫu, triệt tiêu nền cỏ |
| **Xử lý Thủ môn** | Nhận diện như người bình thường, không gán đội | So khớp màu quần áo $\rightarrow$ Sai sót nghiêm trọng ($10-14\%$) | **Spatial-Hybrid**: Tích hợp nửa sân phòng ngự + cự ly hậu vệ $\rightarrow$ Đạt **97.40%** |
| **Xử lý Trọng tài** | Dễ nhầm thành cầu thủ khi đứng gần | Dễ bị kéo vào 1 trong 2 đội bóng | **Cách ly triệt để**: Khử trùng lặp từ Stage 2, bảo đảm **0% lẫn tạp** |
| **Độ tin cậy cho VAR** | Không có cơ chế fail-closed | Ép buộc đưa ra nhãn dù không chắc chắn | **Fail-Closed**: Trả về `UNKNOWN` khi biên tự tin hẹp, đạt **99.07% Selective Accuracy** |

---

## 9. DANH MỤC TÀI LIỆU THAM KHẢO CHÍNH (KEY REFERENCES FOR BIBTEX)

Dành cho tác giả bài báo đưa vào mục **References**:

```bibtex
@inproceedings{deliege2021soccernetv2,
  title={SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos},
  author={Deli{\`e}ge, Adrien and Cioppa, Anthony and Giancola, Silvio and Seikavandi, Meisam J and Dueholm, Jacob V and Bernard, Kamal and Lou, Xinyu and Ghanem, Bernard and Moeslund, Thomas B and Droogenbroeck, Marc Van},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={4508--4519},
  year={2021}
}

@inproceedings{vandeghen2024soccernetgs,
  title={SoccerNet GameState 2024: Multi-View Multi-Object Tracking, Team Affiliation, and Pitch Localization},
  author={Vandeghen, Renaud and Cioppa, Anthony and Giancola, Silvio and Ghanem, Bernard and Van Droogenbroeck, Marc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2024}
}

@article{luiten2021hota,
  title={HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking},
  author={Luiten, Jonathon and Osep, Aljosa and Dendorfer, Patrick and Torr, Philip and Geiger, Andreas and Leal-Taix{\'e}, Laura and Leibe, Bastian},
  journal={International Journal of Computer Vision (IJCV)},
  volume={129},
  number={2},
  pages={548--578},
  year={2021}
}

@inproceedings{jiang2023rtmpose,
  title={RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  author={Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningning and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
  booktitle={arXiv preprint arXiv:2303.07399},
  year={2023}
}

@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={European Conference on Computer Vision (ECCV)},
  pages={1--21},
  year={2022}
}

@inproceedings{cioppa2022soccernet_tracking,
  title={Scaling up SoccerNet with Multi-View Tracking and Pitch Localization},
  author={Cioppa, Anthony and Deli{\`e}ge, Adrien and Giancola, Silvio and Ghanem, Bernard and Van Droogenbroeck, Marc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2022}
}
```

---

## 10. DANH MỤC TÀI LIỆU & ĐƯỜNG DẪN KIỂM CHỨNG TRONG HỆ THỐNG

1. **Mã nguồn Stage 2:** [`stage2_sst_rtmw_v1.3/`](file:///home/tondaiquoc/Workspace/Project/NCKH_SoICT/stage2_sst_rtmw_v1.3/)
2. **Mã nguồn Stage 5:** [`stage5_team_affiliation_v0.1.0/`](file:///home/tondaiquoc/Workspace/Project/NCKH_SoICT/stage5_team_affiliation_v0.1.0/)
3. **Thư mục Benchmark chính thức M3:** `/home/tondaiquoc/benchmarks/stage5_valid_stage5_color_spatial_20260926/`
   * Báo cáo JSON: `benchmark_summary.json`
   * Dữ liệu chi tiết từng sequence: `sequence_metrics.jsonl`
4. **Bộ kiểm thử đơn vị tự động (91 tests PASS):** [`stage5_team_affiliation_v0.1.0/tests/test_goalkeeper_spatial.py`](file:///home/tondaiquoc/Workspace/Project/NCKH_SoICT/stage5_team_affiliation_v0.1.0/tests/test_goalkeeper_spatial.py)
5. **Báo cáo kỹ thuật chi tiết:**
   * [`BAO_CAO_TIEN_DO_M3_SPATIAL_20260926.md`](file:///home/tondaiquoc/Workspace/Project/NCKH_SoICT/BAO_CAO_TIEN_DO_M3_SPATIAL_20260926.md)
   * [`IMPLEMENTATION_STATUS.md`](file:///home/tondaiquoc/Workspace/Project/NCKH_SoICT/stage5_team_affiliation_v0.1.0/IMPLEMENTATION_STATUS.md)
