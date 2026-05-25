# 📋 Review báo cáo — Check yêu cầu GV + lỗi văn chương + đề xuất

**File review:** `report/BaoCao.docx` (bản commit `7a94f76`).
**Yêu cầu trang:** 20-25 trang.
**Trang hiện tại theo mục lục:** ~23 trang (chính). Cộng bìa + tóm tắt + phân công + mục lục = **~27 trang tổng** — **CẦN GIẢM 2-3 trang**.

---

## 1. CHECK YÊU CẦU GV THEO TỪNG CHƯƠNG

### ✅ Chương 1 — Giới thiệu (1 trang)

**Yêu cầu GV**: *"giới thiệu các bài toán cần giải quyết và đề xuất giải pháp tổng quan về dữ liệu dưới dạng sơ đồ khối (pipeline)"*

| Tiêu chí | Status |
|---|---|
| Phát biểu bài toán | ✅ Có |
| 3 yêu cầu output (số cụm, thuộc tính, nhãn) | ✅ Có |
| Sơ đồ khối pipeline | ✅ Có Hình 1.1 |

→ **OK, không cần sửa.**

---

### ⚠️ Chương 2 — Thu thập và mô tả dữ liệu (6 trang)

**Yêu cầu GV 2.1**: *"nguồn dữ liệu, công cụ thu thập, cách thức, đầu vào/đầu ra, ví dụ minh hoạ; chốt tỉ lệ 90/10; cảnh báo data leakage"*
**Yêu cầu GV 2.2**: *"thống kê tổng quan (số mẫu, đặc trưng, kiểu, missing), trực quan (boxplot, histogram), phân bố biến quan trọng, distribution shift train/test"*

| Tiêu chí | Status |
|---|---|
| Nguồn + công cụ + I/O | ✅ Có |
| Chốt 90/10 + cảnh báo data leakage | ⚠️ **THIẾU cảnh báo data leakage** — GV ghi rõ trong template "SV vô tình hay cố ý bị rò rỉ dữ liệu thì điểm CK rất thấp". Thêm 1 câu ở cuối mục 2.1: *"Để tránh rò rỉ dữ liệu (data leakage), mọi tham số fit của pipeline (junk_buckets, TOP_INDUSTRIES, scaler, TF-IDF vocabulary, SVD components, KMeans centroids) đều được fit trên tập train và áp dụng nguyên xi cho test."* |
| Schema 14 cột | ✅ Có Bảng 2.1 |
| Missing analysis | ✅ Có |
| Phân bố các biến | ✅ Có nhiều ảnh |
| Distribution shift | ✅ Có Hình 2.11 |

**Vấn đề**: **11 ảnh trong 6 trang quá dày**. Nhiều ảnh overlap nhau:
- Hình 2.5 (province top 15) — sẽ lặp lại ở Chương 3 cleaning
- Hình 2.9 (salary by industry) + Hình 2.10 (salary by province) — overlap với Hình 2.8 (salary by years_exp), không cần cả 3
- Hình 2.6 (year), 2.7 (text length) — phụ trợ, ít quan trọng

**Đề xuất giảm còn 6-7 ảnh, bỏ:**
- ❌ Hình 2.5 (province top 15) — trùng nội dung với Hình 3.x (province sau cleaning)
- ❌ Hình 2.6 (year đăng) — gộp vào 1 câu prose
- ❌ Hình 2.7 (text length) — gộp vào 1 câu prose
- ❌ Hình 2.9 (salary by industry) — đã đủ với 2.8
- ❌ Hình 2.10 (salary by province) — đã đủ với 2.8

→ **Giảm 5 ảnh = bớt ~1.5-2 trang.** Văn bản giữ nguyên.

---

### ✅ Chương 3 — Trích xuất đặc trưng (3 trang)

**Yêu cầu GV**: *"lựa chọn đặc trưng, làm sạch, chuẩn hoá, giảm chiều, trực quan **trước và sau khi xử lý**"*

| Tiêu chí | Status |
|---|---|
| Cleaning rules đầy đủ | ✅ Có 6 bước |
| Pipeline 1 (TF-IDF + SVD) | ✅ Có bảng + giải thích |
| Pipeline 2 (SBERT) | ✅ Có |
| **Trực quan trước/sau xử lý** | ⚠️ Chỉ có Hình 3.1 (salary trước/sau). **Yếu** — GV yêu cầu rõ "trước và sau". |

**Đề xuất bổ sung**:
- Thêm 1 câu ngắn ở Hình 3.1: *"Sau filter, max salary giảm từ 994tr → 100tr, phân phối tập trung 0-50tr, không còn outlier nguy hiểm cho K-Means."*
- Hình 3.2 (SVD variance) đã cho thấy "tác dụng" của giảm chiều — đủ.

→ **OK, chỉ thêm 1 câu prose, không phải sửa lớn.**

---

### ⚠️ Chương 4 — Mô hình hoá (7 trang — DÀI nhất)

**Yêu cầu GV (6 ý)**:
1. ≥ 2 mô hình + cơ sở lý thuyết + siêu tham số ✅
2. Chia train/test ✅
3. Tham số huấn luyện ✅
4. **Đồ thị hiệu suất** ✅ (Hình 4.1 K-search)
5. Metrics + so sánh bằng bảng/đồ thị ✅ (Bảng 4.1, 4.2 + Hình 4.4)
6. **Đánh giá, bình luận, giải thích** ✅ (Mục 4.4 dài)

**Vấn đề**: **7 trang là dài**, có thể trim còn 5-6 trang để báo cáo không vượt 25.

**Đề xuất rút gọn:**

1. **Bảng 4.1 và Bảng 4.2 hơi dày**. Có thể gộp thành **1 bảng so sánh side-by-side** (cluster × {Pipeline 1, Pipeline 2}) — tiết kiệm 1/2 trang.

2. **Đoạn Optuna** ở mục 4.2 có thể rút gọn 1 đoạn:
   - Hiện: *"Optuna fine-tune: Mở rộng search sang (k, init, whitening_strength)... Optuna chọn k = 21... Áp dụng logic 'grid wins' → giữ k = 5..."* (4-5 dòng)
   - Rút gọn: *"Optuna 30 trials TPE mở rộng search sang (k, init, whitening). Kết quả: grid k=5 vẫn tốt hơn Optuna k=21; whitening tuned ≈ 0 không có tác dụng."* (2 dòng)

3. **Mục 4.4 phần "Vai trò GMM"** dài. Có thể rút 1 dòng.

→ **Tiết kiệm 1-1.5 trang.**

---

### ✅ Chương 5 — Kết luận (1 trang)

OK, đầy đủ tổng kết + 4 hướng phát triển.

---

### ✅ Chương 6 — TLTK (1 trang)

OK, đã sửa sang IEEE format. **NHƯNG còn THIẾU trích dẫn `[N]` trong body báo cáo!**

GV ghi: *"liệt kê các TLTK đã trích dẫn (cite) trong báo cáo tại đây"* — TLTK phải được **cite trong văn bản** trước khi list. Hiện báo cáo của user **chưa có `[N]` nào** trong body.

**Đề xuất thêm** (xem section 3 phía dưới).

---

## 2. VĂN CHƯƠNG KHÓ HIỂU CẦN SỬA

### 🔴 Lỗi văn — sửa tại các vị trí:

**Vị trí 1** — Chương 4.2 (về Optuna):
- ❌ *"Áp dụng logic 'grid wins' → giữ k = 5"*
- ✅ Sửa thành: *"Vì grid search cho silhouette tốt hơn, nhóm giữ k = 5 (kết quả grid) thay vì kết quả Optuna."*

**Vị trí 2** — Chương 4.1 (HDBSCAN):
- ❌ *"không khả thi trên 545k × 768D với CPU consumer"*
- ✅ Sửa thành: *"không khả thi trên 545k × 768D với CPU phổ thông"*

**Vị trí 3** — Chương 4.1 (HDBSCAN tiếp):
- ❌ *"ngoài tài nguyên hiện có"*
- ✅ Sửa thành: *"nằm ngoài phạm vi tài nguyên hiện có"*

**Vị trí 4** — Chương 4.4:
- ❌ *"Pipeline 1 thắng metric numeric"*, *"Pipeline 2 thua metric"*
- ✅ Sửa thành: *"Pipeline 1 vượt trội về metric numeric"*, *"Pipeline 2 kém hơn về metric"*

**Vị trí 5** — Chương 4.4:
- ❌ *"embedding các JD cùng 'category' (đều là tin tuyển dụng) collapse lại gần nhau"*
- ✅ Sửa thành: *"các JD cùng 'category' (đều là tin tuyển dụng) bị co cụm lại gần nhau trong không gian embedding"*

**Vị trí 6** — Chương 5 (Kết quả số 5):
- ❌ *"đáp ứng yêu cầu defense 5 phút"*
- ✅ Sửa thành: *"đáp ứng yêu cầu bảo vệ 5 phút"*

**Vị trí 7** — Chương 4.4 cuối (GMM):
- ❌ *"Mở rộng hướng phát triển có thể dùng GMM k = 14 để tách sub-cluster"*
- ✅ Sửa thành: *"Có thể mở rộng hướng phát triển bằng GMM với k = 14 để phân tích các cụm con bên trong (sub-cluster)"*

**Vị trí 8** — Chương 2.2 (cuối phần ngành):
- ❌ *"Top 50 ngành cover 99.97% tất cả tokens phù hợp để áp dụng MultiLabelBinarizer."* — kỹ thuật MLB chen vào EDA hơi sớm.
- ✅ Sửa thành: *"Top 50 ngành cover 99.97% tất cả tokens — cardinality đủ thấp để encoding đa nhãn ở stage feature engineering."*

**Vị trí 9** — Chương 3.1 (cuối):
- ❌ *"Kết quả: Train 546.190 → 545.480 dòng (giữ 99.87%); Test 60.688 → 60.606 dòng (giữ 99.86%)"*
- ✅ Sửa thành câu hoàn chỉnh hơn: *"Sau cleaning, train còn 545.480 dòng (giữ lại 99.87%) và test còn 60.606 dòng (giữ 99.86%). Schema dữ liệu sạch gồm 17 cột."*

---

## 3. THIẾU TRÍCH DẪN `[N]` TRONG BODY — BẮT BUỘC

Theo IEEE style (yêu cầu của BK), TLTK phải được cite trong văn bản bằng `[N]`. Hiện chưa có. **Bắt buộc thêm** ở các vị trí:

| Vị trí trong body | Sửa thành |
|---|---|
| Mục 4.1 *"MiniBatchKMeans: K-Means cập nhật centroid theo mini-batch..."* | "MiniBatchKMeans [7]: K-Means cập nhật..." |
| Mục 3.2 Pipeline 1 *"TruncatedSVD 150D..."* | "TruncatedSVD [6] 150D..." |
| Mục 3.2 Pipeline 2 *"Sentence-BERT encode ngữ nghĩa câu"* | "Sentence-BERT [5] encode..." |
| Mục 3.2 Pipeline 2 *"Mô hình: paraphrase-multilingual-mpnet-base-v2"* | "Mô hình: paraphrase-multilingual-mpnet-base-v2 [5]..." |
| Mục 4.1 *"Optuna hyperparameter tuning"* | "Optuna [8] hyperparameter tuning" |
| Mục 4.2 *"Silhouette ∈ [−1, 1]"* | "Silhouette [10] ∈ [−1, 1]" |
| Mục 4.2 *"Davies-Bouldin: thấp = tốt"* | "Davies-Bouldin [11]: thấp = tốt" |
| Mục 4.2 *"Calinski-Harabasz: cao = tốt"* | "Calinski-Harabasz [12]: cao = tốt" |
| Mục 2.1 *"Sử dụng thư viện requests..."* | "Sử dụng thư viện sklearn [4] để split..." |
| Chương 1 — đoạn nói về clustering | "Bài toán phân cụm... [1][2][3] là bài toán..." (cite bài giảng + GeeksforGeeks + sklearn) |
| Mục 2.1 *"Dataset Tinix Vietnam Job Descriptions"* | "Dataset Tinix Vietnam Job Descriptions [9]..." |

---

## 4. TÓM TẮT ĐỀ XUẤT — TỐI ƯU 20-25 TRANG

| Việc | Thay đổi | Tiết kiệm |
|---|---|---|
| Bỏ 5 ảnh thừa ở Chương 2 (Hình 2.5, 2.6, 2.7, 2.9, 2.10) | Gộp prose ngắn | ~1.5-2 trang |
| Gộp Bảng 4.1 + 4.2 thành 1 bảng side-by-side | Layout hẹp hơn | ~0.5 trang |
| Rút gọn đoạn Optuna ở mục 4.2 (4 dòng → 2 dòng) | Câu ngắn | ~0.2 trang |
| Rút gọn mục 4.4 "Vai trò GMM" | 1 dòng bớt | ~0.1 trang |
| **TỔNG TIẾT KIỆM** | | **~2.5-3 trang** |

→ Báo cáo còn **~22-23 trang**, vào đúng khoảng 20-25.

---

## 5. ĐIỀU CẦN BỔ SUNG (KHÔNG BỎ ĐƯỢC)

1. **Cảnh báo data leakage** (1 câu ở cuối mục 2.1) — yêu cầu rõ của GV.
2. **Trích dẫn `[N]`** trong body (xem section 3 ở trên) — bắt buộc theo IEEE.
3. **Câu prose nối Hình 3.1** giải thích "tác dụng của cleaning trước/sau" — yêu cầu của GV chương 3.

---

## 6. SIDE NOTES

- **Lớp học phần**: trong bảng phân công ghi "2315" — đúng không? Template gốc dùng "2011A". Kiểm tra lại với GV.
- **Trang bìa**: dạng bảng pandoc convert chưa đẹp. Bạn có thể format lại tay trong Word cho đúng kiểu template (logo trường + tên đề tài chính giữa).
- **Hình 1.1 pipeline**: bạn mới thêm — kiểm tra ảnh có rõ nét + không bị mờ khi in.
- **Mục lục**: tự generate qua Word *References → Table of Contents* để có số trang đúng + bookmark hoạt động (GV check kỹ).
