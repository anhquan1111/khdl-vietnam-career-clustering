# Nội dung Slide trình bày — 10 phút + 5 phút demo testing.ipynb

> **Tổng**: 14 slide trình bày, chia đều cho 3 thành viên:
> - **Hữu Tuân**: 4 slide (~3 phút) — EDA sâu
> - **Việt Toàn**: 4 slide (~3 phút) — Cleaning sâu
> - **Anh Quân**: 6 slide (~4 phút) — Modeling + So sánh + Kết luận
>
> Cuối cùng chạy `testing.ipynb` trong 5 phút.

---

## 🎬 Slide 1 — Trang bìa + Mở đầu (30s — Hữu Tuân vào đề)

**TIỂU LUẬN CUỐI KỲ — KHOA HỌC DỮ LIỆU**

# PHÂN CỤM BỘ DỮ LIỆU VIỆC LÀM VIỆT NAM

**Nhóm 09**
- Võ Trần Anh Quân (102230131)
- Phạm Hữu Tuân (102230139)
- Trần Việt Toàn (102230137)

*Đà Nẵng, 05/2026*

> **Người nói**: chào thầy cô, giới thiệu tên đề tài + 3 thành viên + ai phụ trách phần nào (1 câu).

---

# PHẦN 1 — EDA (Hữu Tuân, ~3 phút, 4 slide)

---

## 🎬 Slide 2 — Bài toán & Dataset (45s — Hữu Tuân)

### Bài toán
**Phân cụm 606.878 bài đăng tuyển dụng Việt Nam** — unsupervised, không có ground truth.

### Output GV yêu cầu
1. **Số cụm + hiệu suất** (silhouette, Davies-Bouldin, Calinski-Harabasz)
2. **Thuộc tính chung** mỗi cụm
3. **Nhãn tự gán** ý nghĩa kinh tế

### Dataset TINIX Vietnam Job Descriptions
- 📊 **606.878 dòng × 14 cột** (Parquet ~293MB nén, ~1.5GB giải nén)
- 📅 Năm đăng **2022–2026**
- 🗣️ Chủ yếu tiếng Việt, một phần Anh
- 🔀 Split **90/10** (train 546k / test 60k), `random_state = 42`

> **Người nói**: giới thiệu bài toán clustering + quy mô data lớn. Nhấn vào đây không phải regression — không có "đáp án đúng" để so sánh.

---

## 🎬 Slide 3 — EDA: Tổng quan + Phân phối lương (50s — Hữu Tuân)

### Chất lượng dữ liệu tốt
- Missing rất nhỏ: cao nhất **`benefits` 0.65%**, `requirements` 0.62%
- Không có cột nào thiếu trên 1%
- Không trùng `id`, không leak giữa train/test

### Lương — text dạng "10.000.000 - 15.000.000 VND"
Cần parse bằng **4 regex pattern**:
| Pattern | % | Ví dụ |
|---|---|---|
| `vnd_range_dotted` | 88% | "10.000.000 - 15.000.000 VND" |
| `vnd_range_month` | 6.5% | "10000000 - 15000000 VND MONTH" |
| `vnd_single_dotted` | <0.01% | "12.000.000 VND" |
| `usd_range` | <0.01% | "500 - 1000 USD" (×25.000) |

**Tỉ lệ parse: ~95%** — còn lại "thoả thuận", "0 VND", "đang cập nhật" → giữ với flag `salary_missing`.

![Phân phối salary (Hình 2.2)](../figures/eda_02_salary_distribution.png)

> **Người nói**: 4 pattern đủ cover 95% — phần còn lại không drop, giữ với flag (anti-pattern thường gặp).

---

## 🎬 Slide 4 — EDA: Phát hiện JUNK DATA (50s — Hữu Tuân)

### 2 loại junk phát hiện bằng rule-based — không phải vẽ chart bừa!

🔍 **Junk 1: Sentinel `999.000.000 VND`**
- 642 dòng có max range = "999.000.000 VND"
- Đi kèm các vị trí bình thường: Phục Vụ, Pha Chế, Lễ Tân → chắc chắn là giá trị mặc định
- **Drop** trong stage cleaning

🔍 **Junk 2: Bucket `years_exp = 9`**
- 228 dòng, **99.6% có `job_position = "Chưa cập nhật"`**
- So với bucket khác chỉ ~5-10% missing → bất thường
- Phát hiện bằng luật: `n_rows < 500 AND pos_missing_rate > 0.5`
- **Set NaN** trong stage cleaning

![Phân phối years_exp (Hình 2.3)](../figures/eda_03_years_exp.png)

> **Người nói**: nhấn EDA tốt phát hiện được junk → cleaning sạch sẽ → model tốt. Rule-based detect mang tính tổng quát hơn hardcode {9}.

---

## 🎬 Slide 5 — EDA: Trục cluster tiềm năng + Train/Test (50s — Hữu Tuân)

### Lương phân bố khác biệt rõ theo nhiều trục

![Lương theo số năm KN (Hình 2.8)](../figures/eda_10_salary_by_years_exp.png)

| Trục | Insight |
|---|---|
| **Years exp** | Junior 8tr → Senior 22tr, tuyến tính |
| **Ngành** (top 50 cover 99.97%) | IT/BĐS/Tài chính cao; Phổ thông thấp |
| **Tỉnh/thành** | HCM + Hà Nội chiếm 60%+ |
| **Năm đăng** | 2025–2026 chiếm phần lớn, lạm phát nhẹ |

→ **4 trục cluster có ý nghĩa** = ngành × kinh nghiệm × địa lý × lương.

### Train ↔ Test KHÔNG lệch
Phân phối các biến chính giữa 2 tập gần như đồng nhất → split 90/10 ngẫu nhiên với seed 42 không gây distribution shift → model học từ train có thể generalize sang test.

> **Người nói**: chốt 4 trục cluster — đây là motivation cho stage feature engineering. Train/test verify split tốt → bàn giao sang Việt Toàn nói về cleaning.

---

# PHẦN 2 — CLEANING (Việt Toàn, ~3 phút, 4 slide)

---

## 🎬 Slide 6 — Cleaning: Tổng quan + Salary thành 3 feature (50s — Việt Toàn)

### Mục tiêu Stage 2
Áp các quyết định EDA lên **toàn bộ 545k JD train + 61k test**, output `clean_data` 17 cột chuẩn cho FE.

### Salary: 1 cột text → 3 feature numeric + 1 flag

```
"10.000.000 - 15.000.000 VND"
            ↓
salary_min = 10.0    (triệu VNĐ)
salary_max = 15.0
salary_mid = 12.5
salary_missing = False
```

🔑 **Vì sao 3 feature, không phải 1?**
- `salary_mid` mạnh nhất cho phân cụm theo mức lương
- `salary_max - salary_min` (biên độ) = tín hiệu **tin gộp nhiều vị trí** hoặc **tin lương không rõ ràng**
- Cả 2 thông tin được giữ → cluster có chiều phong phú hơn

> **Người nói**: nhấn vào việc PARSE thông minh — không drop dòng không parse được mà giữ với flag.

---

## 🎬 Slide 7 — Cleaning: Outlier handling + salary_missing flag (50s — Việt Toàn)

### 2 quyết định outlier critical

❌ **DROP dòng có sentinel `... -999.000.000 VND`**
- 33 dòng trong train, 100% là junk
- Lý do: K-Means dùng Euclidean distance → 1 dòng 850M kéo lệch centroid mạnh

❌ **DROP dòng `salary_mid > 100M` không sentinel**
- 609 dòng — outlier scale tự nhiên
- Lý do: phá StandardScaler (std bị inflated)

✅ **GIỮ dòng "thoả thuận / đang cập nhật / 0 VND"** (5.4% data)
- Đánh dấu `salary_missing = True`
- Lý do: có thể là tín hiệu cụm riêng — "tin tuyển dụng giấu lương" thường là cấp manager/specialist

![Salary trước/sau cleaning (Hình 3.1)](../figures/cleaning_salary_distribution.png)

> **Người nói**: sau filter, max salary giảm từ **994tr → 100tr**, phân phối tập trung 0-50tr, sạch cho K-Means.

---

## 🎬 Slide 8 — Cleaning: years_exp, province, industries (50s — Việt Toàn)

### `years_exp` — Regex + Junk bucket detection
```python
"3 năm"          → 3.0
"Dưới 1 năm"     → 0.5
"Không"          → NaN
```
- Parse rate ~96%
- Junk bucket `9 năm` (228 dòng) → set NaN bằng rule auto

### `province` — Regex 2 lớp
**Lớp 1**: match 63 tỉnh/thành Việt Nam trực tiếp
**Lớp 2 (fallback)**: tra district → province
- HCM: Quận 1–12, Tân Bình, Bình Thạnh, Thủ Đức, ...
- Hà Nội: Hoàn Kiếm, Ba Đình, Đống Đa, ...

→ **Coverage 94%**, không match → `'Other'`

### `industries_list` — Multi-label
- Tách dấu `/` (vd. "Sales / CSKH / Bán lẻ")
- Top 50 ngành phổ biến + `'Other'` cho phần còn lại
- Cover **99.97%** tổng tokens
- Lưu pipe-separated: `"Sales|CSKH|Bán lẻ"`

> **Người nói**: regex 2 lớp province là kỹ thuật độc đáo — chỉ HCM/HN cần fallback district vì 2 thành phố này nhiều JD ghi địa chỉ chi tiết quận/huyện thay vì tên thành phố.

---

## 🎬 Slide 9 — Cleaning: Chunked Processing + Kết quả (50s — Việt Toàn)

### Vấn đề: train CSV 1.1 GB, RAM 16 GB
Load full vào pandas → ~3-5 GB → có thể OOM khi xử lý.

### Giải pháp: Chunked-Read + Write Incremental
```
Đọc 50.000 dòng/batch → Apply transforms → Append vào output CSV
                              ↓
                Lặp 11 lần cho 545k dòng → done
```
- Memory peak: chỉ ~500 MB (1 chunk DataFrame)
- Disk I/O nhanh, đảm bảo chạy được trên máy phổ thông

### Kết quả retain cao
| Tập | Raw | Clean | Retain |
|---|:---:|:---:|:---:|
| **Train** | 546.190 | 545.480 | **99.87%** |
| **Test** | 60.688 | 60.606 | **99.86%** |

**Schema clean: 17 cột** (thêm 3 cột salary + flag missing).

> **Người nói**: chunked-read là điểm sáng kỹ thuật — chỉ drop 0.13% data toàn junk thật → bàn giao sang Anh Quân.

---

# PHẦN 3 — MODELING + KẾT LUẬN (Anh Quân, ~4 phút, 6 slide)

---

## 🎬 Slide 10 — Feature Engineering: 2 Pipeline (45s — Anh Quân)

### Pipeline 1 — TF-IDF + TruncatedSVD (lexical features)

| Nhóm | Chiều |
|---|:---:|
| Numeric (salary × 4 + years_exp + year) | 6 |
| OneHot (province, edu, type, position) | ~95 |
| MultiLabel industries | 51 |
| TF-IDF 4 cột text (max_df=0.85 + **VN stopwords**) | ~25.500 |
| **Tổng sparse** | **~25.700** |
| **TruncatedSVD** → dense | **150D (giữ 77.7% variance)** |

### Pipeline 2 — Sentence-BERT (semantic embedding)
- Model: **`paraphrase-multilingual-mpnet-base-v2`** (768D, đa ngôn ngữ)
- Encode 606k JD trên **GPU RTX 4060** ~50 phút (~200 docs/s)
- L2-normalize → KMeans tương đương cosine clustering

→ **2 approach song song** để so sánh ở stage clustering.

> **Người nói**: nhấn (a) TF-IDF cần stop words VN vì top tokens nguyên bản toàn từ chức năng "theo", "có", "của"; (b) SBERT thay TF-IDF để bắt ngữ nghĩa.

---

## 🎬 Slide 11 — K-search + Chọn k=5 (40s — Anh Quân)

### 4 metric cho k ∈ [3, 25]

![K-search 4 metric (Hình 4.1)](../figures/clustering_k_search.png)

| Metric | Hướng | Best k |
|---|:---:|:---:|
| Inertia | ↓ luôn (elbow) | ~5–6 |
| **Silhouette** | ↑ | **5** (0.128) |
| Davies-Bouldin | ↓ | 5 (2.21) |
| Calinski-Harabasz | ↑ | 3 (4110) |

### Optuna 30 trials TPE
- Mở rộng search `(k, init, whitening_strength)`
- Optuna chọn k=21 (sil 0.096) — **tệ hơn grid k=5**
- Whitening tuned ≈ 0 → không tác dụng
→ **Grid wins** = chốt **k=5** cho cả 2 pipeline

> **Người nói**: inertia luôn giảm → không dùng được; phải dùng silhouette. Optuna xác nhận grid là tốt nhất → finding khoa học.

---

## 🎬 Slide 12 — 5 cụm Pipeline 1 + NHÃN Ý NGHĨA (KEY, 1 phút — Anh Quân)

### Pipeline 1 (TF-IDF + KMeans, k=5)

| Cụm | n (%) | Lương | KN | Nhãn |
|:---:|:---:|:---:|:---:|---|
| 0 | 21.6% | 9.8tr | 1.4y | **Nhân viên junior phổ thông** — Không bằng cấp |
| 1 | 28.6% | 10.4tr | 2.2y | **Nhân viên đa ngành mid-junior** — CĐ/TC |
| 2 | 19.9% | **23.0tr** | 3.4y | **Sales/BD lương cao + hoa hồng** |
| 3 | 24.5% | 12.0tr | 3.6y | **Kế toán/Kiểm toán & Kỹ thuật** mid |
| 4 | 5.4% | NaN | 3.1y | **Tin lương "thoả thuận"** — Manager/IT |

### 3 insight nổi bật
🌟 **Cụm 2 (~23tr)**: KHÔNG phải senior — là **Sales/BĐS/Bảo hiểm với commission cao** (KN chỉ 3.4 năm)
🌟 **Cụm 4 (5.4%)**: cụm đặc thù — tin tuyển dụng giấu lương = cấp manager + IT specialist
🌟 **Cụm 0+1** (50%): lực lượng junior chiếm nửa thị trường VN

> **Người nói**: slide **chấm điểm cao nhất** — nói chậm, giải thích từng cụm có ý nghĩa kinh tế gì.

---

## 🎬 Slide 13 — Pipeline 2 SBERT: phân ngành sạch hơn (45s — Anh Quân)

### 5 cụm SBERT — TÁCH NGÀNH RÕ HƠN

| Cụm | Nhãn |
|:---:|---|
| 0 | Service/CSKH & Giáo dục junior |
| 1 | Sales & Customer-facing mid |
| 2 | **Marketing & Education** ⭐ |
| 3 | **Kế toán/Tài chính** specialists ⭐ |
| 4 | **Engineering & Industrial** ⭐ |

![t-SNE 2D SBERT (Hình 4.3)](../figures/clustering_embed_tsne_2d.png)

### So với Pipeline 1
- TF-IDF gộp Marketing với Sales (vì cùng dùng "khách hàng", "bán hàng")
- **SBERT tách Marketing, Engineering, Industrial riêng** — hiểu ngữ cảnh
- Ví dụ JD *"Nhân Viên Editor (Marketing)"*: TF-IDF phân SAI vào Sales, SBERT phân ĐÚNG vào Marketing

> **Người nói**: SBERT thắng về diễn giải — phân ngành rõ rệt hơn, dù metric thua TF-IDF (slide tiếp theo).

---

## 🎬 Slide 14 — So sánh + Kết luận + Roadmap (1 phút — Anh Quân)

### Bảng so sánh metric

| Metric | Pipeline 1 (TF-IDF) | Pipeline 2 (SBERT) |
|---|:---:|:---:|
| Silhouette ↑ | **+0.128** | +0.068 |
| Davies-Bouldin ↓ | **2.21** | 3.31 |
| Calinski-Harabasz ↑ | **3249** | 1406 |

### 🔑 Finding chính
> *TF-IDF lexical **thắng metric** nhưng SBERT semantic **thắng diễn giải***

Trade-off: metric vs interpretability. **Advanced (SBERT) không phải lúc nào cũng tốt hơn**.

### Limitation
HDBSCAN không khả thi trên CPU consumer (>2.5h treo) — cần GPU implementation (cuML).

### Hướng phát triển (4 mục ngắn)
1. HDBSCAN trên NVIDIA RAPIDS cuML (GPU)
2. Fine-tune PhoBERT cho domain JD VN → kỳ vọng sil 0.2-0.3
3. Information Extraction kỹ năng cứng (Python, Excel, CAD, ...)
4. GMM k=14 sub-cluster analysis

---

**Cảm ơn thầy cô — Sẵn sàng demo `testing.ipynb` 5 phút!**

> **Người nói**: tóm tắt nhanh trade-off, ghi nhận limitation, đề xuất 4 hướng phát triển. Chuyển sang demo.

---

# 🎬 PHẦN DEMO `testing.ipynb` — 5 phút (Anh Quân điều khiển)

**Thời lượng**: notebook chạy < 1 phút, còn 4 phút giải thích.

### Lúc Run All → 6 cells chạy

| Cell | Nội dung | Giải thích |
|---|---|---|
| 1-2 | Load test + 2 model | Load 60k JD test + 2 KMeans (TF-IDF + SBERT) + nhãn cụm |
| 3 | **10 mẫu test demo** | Random 10 JD, hiển thị nhãn của CẢ 2 pipeline side-by-side |
| 4 | Bar chart phân bố cụm test | 5 cụm test phân bố ổn → generalize tốt từ train |
| 5 | t-SNE 5k visualization | Cụm phân biệt rõ trong 2D |
| 6 | Bảng so sánh metric | Closing — tóm tắt 3 metric chính |

### Tâm điểm 4 phút giải thích

**1 phút — 10 mẫu demo**: chỉ vào JD đặc trưng
- "JD *Nhân Viên Editor (Marketing)*: Pipeline 1 phân SAI vào Sales, Pipeline 2 phân ĐÚNG vào Marketing"
- "JD *Trade Operations Manager* lương missing → cụm 4 'Tin lương thoả thuận' — đúng bản chất"

**1 phút — Bar chart cluster sizes**:
- "Phân bố tương đồng train → model không overfit"
- "Cụm 1 (28%) lớn nhất = lực lượng lao động mid-junior chủ lực"

**1.5 phút — t-SNE**:
- "Mỗi màu = 1 cluster trong embedding space"
- "Cụm 4 (lương missing) tách riêng — flag `salary_missing` thực sự là tín hiệu"
- "Cụm 0+1 (junior) gần cụm 2 (senior) → continuum kinh nghiệm"

**0.5 phút — Tổng kết**: "Demo verify model load + predict được trên data chưa thấy → sẵn sàng Q&A".

---

# 📋 Checklist trước buổi bảo vệ

- [ ] Slide PDF luyện trước 1 lần, đếm timing đúng 10 phút
- [ ] `testing.ipynb` test chạy < 1 phút trên máy demo
- [ ] Mang **2-3 máy tính** phòng sự cố
- [ ] Bản in báo cáo khớp 100% với PDF nộp
- [ ] Folder submission `09 - Phân cụm bộ dữ liệu việc làm Việt Nam` chứa: PDF báo cáo + PDF slide + `training.ipynb` + `testing.ipynb` + README + `raw_data/` + `clean_data/`
- [ ] 3 thành viên đã luyện phần của mình

---

# 🎨 Tips làm slide PDF

1. **Font lớn**: tiêu đề ≥ 28pt, body ≥ 18pt
2. **Ít text mỗi slide**: ≤ 6 dòng + 1-2 hình
3. **Màu nhất quán**: 1 màu chính + 1 màu nhấn
4. **Ảnh chất lượng**: PNG gốc từ `figures/`, không zoom vỡ
5. **Số trang góc dưới**: dễ Q&A tham chiếu
6. **Animation tối thiểu**: chỉ "appear" cho bullet
7. **Presenter view**: bật timer để theo dõi thời gian

### Phân bố thời gian chuẩn

| Phần | Slide | Thời lượng | Người nói |
|---|:---:|:---:|---|
| Mở đầu + EDA | 1-5 | ~3 phút 15s | **Hữu Tuân** |
| Cleaning | 6-9 | ~3 phút 20s | **Việt Toàn** |
| Modeling + Kết luận | 10-14 | ~3 phút 25s | **Anh Quân** |
| **Tổng trình bày** | | **~10 phút** | |
| Demo testing.ipynb | | 5 phút | **Anh Quân** điều khiển |
