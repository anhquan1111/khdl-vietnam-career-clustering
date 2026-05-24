# CLAUDE.md — Hướng dẫn cho phiên làm việc Claude

> File này dành cho Claude (AI) đọc khi bắt đầu phiên chat mới, để nắm được bối cảnh dự án mà không cần đọc lại toàn bộ lịch sử chat.

---

## 1. Bối cảnh dự án

- **Môn học:** Khoa học Dữ liệu (KHDL), năm học 2025–2026.
- **Nhóm:** 09.
- **Đề tài (số 2 — phiên bản cập nhật ngày 2026-05-23):** *Phân cụm bộ dữ liệu việc làm Việt Nam.*
- **Loại bài toán:** **Clustering** (unsupervised). KHÔNG còn là regression như đề ban đầu.
- **Input:** **toàn bộ dataset** (kể cả cột `salary` — giờ là feature, không phải target).
- **Output yêu cầu của GV:**
  1. Số cụm và hiệu suất phân cụm (dùng các metric phổ biến: silhouette, Davies-Bouldin, Calinski-Harabasz, inertia/elbow).
  2. Các thuộc tính chung (phổ biến) của các phần tử thuộc mỗi cụm.
  3. Dựa vào (2), **tự gán nhãn** cho mỗi cụm (vd. "IT mid-level HCM", "Sales junior toàn quốc", v.v.).
- **Dataset:** [tinixai/vietnamese-job-descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions) — 606,878 dòng × 14 cột, Parquet ~293 MB. Phải dùng **TOÀN BỘ dataset** (yêu cầu của GV cho dataset TINIX có sẵn).
- **Tỉ lệ split (theo template báo cáo `Mau tieu luan KHDL_2026.docx`):** **90/10** (train/test, random_state=42).
  - Train (~546k dòng): dùng để huấn luyện và lựa chọn mô hình (số cụm, hyperparam clustering).
  - Test (~61k dòng): với bài clustering, **chỉ random 10 mẫu** trong test để demo kết quả thuật toán phân cụm (gán nhãn cụm, hiển thị nội dung job tương ứng). Không dùng test để đánh giá metric.
  - ✅ Đã re-split 90/10 (commit `dcfc266` là checkpoint regression cũ; sau đó `_download_and_split.py` đã chuyển sang chunked-read tránh OOM).
- **TLTK GV gợi ý:**
  - Phần 2.2 (Phân cụm) của bài giảng môn học (link Drive trong [đề bài cập nhật](./2026_MS%20Teams-TB%20nộp%20tiểu%20luận%20cuối%20kỳ_23.15(2).docx.md)).
  - [GeeksforGeeks — Clustering in Machine Learning](https://www.geeksforgeeks.org/machine-learning/clustering-in-machine-learning/).

> **Lịch sử:** Đề ban đầu (file [...23.15.docx.md](./2026_MS%20Teams-TB%20nộp%20tiểu%20luận%20cuối%20kỳ_23.15.docx.md)) là *"Dự đoán mức lương kỳ vọng dựa trên JD"* (regression). GV update đề ngày 23/5/2026 thành clustering. Code stage 1–4 hiện tại được thiết kế cho regression — đang **migrate sang clustering từng bước**, EDA là bước đầu.

---

## 2. Quy tắc nộp bài (BẮT BUỘC — vi phạm = 0 điểm)

Trích từ thông báo cập nhật ([2026_MS Teams-TB nộp tiểu luận cuối kỳ_23.15(2).docx.md](./2026_MS%20Teams-TB%20nộp%20tiểu%20luận%20cuối%20kỳ_23.15(2).docx.md)):

### 2.1 Hạn

| Mốc | Thời gian |
|---|---|
| Đăng ký tên đề tài trên Google Sheets | trước CN 24/5/2026 |
| Link upload bài mở | 21h00 ngày 26/5/2026 |
| Link upload bài đóng | **21h15 ngày 26/5/2026** (chỉ 15 phút) |
| Bản in báo cáo nộp | đầu buổi thi |

### 2.2 Folder bài nộp

Quy tắc đặt tên: `STTnhom - Tên đề tài` (vd. `09 - Phân cụm bộ dữ liệu việc làm Việt Nam`).

Phải chứa đủ:

1. **File PDF báo cáo** — giống 100% bản in, độ dài 15–20 trang (không kể mục lục & TLTK). **Không được đưa code vào báo cáo.**
2. **File PDF slide** — trình bày theo thứ tự nội dung mẫu báo cáo, tối đa 15 phút.
3. **File `.ipynb`** — một hoặc nhiều notebook.
4. **File README** — hướng dẫn trình tự chạy chương trình.
5. **2 file (hoặc thư mục con) raw data** — đặt tên đúng: `raw_data_train.csv`, `raw_data_test.csv`.
6. **2 file (hoặc thư mục con) clean data** — đặt tên đúng: `clean_data_train.csv`, `clean_data_test.csv`.

### 2.3 4 lỗi → 0 điểm (cấm tuyệt đối)

1. File PDF báo cáo **khác** bản in (lỗi bookmark / mục lục / định dạng).
2. Folder upload thiếu nội dung GV yêu cầu.
3. Nộp muộn hoặc tự ý sửa sau khi hết hạn.
4. Báo cáo và chương trình không khớp với thông tin đăng ký trên Google Sheets.

### 2.4 Slide

- Trình bày tối đa 15 phút.
- Tập trung vào **đặc tính dữ liệu** và **nhận xét/lý giải** kèm bảng + đồ thị.
- Đặc biệt: nhấn vào **so sánh các thuật toán phân cụm**, **các metric đánh giá**, và **diễn giải nhãn cụm**.
- Mang sẵn **2–3 máy tính** phòng sự cố.

---

## 3. Cấu trúc thư mục hiện tại

```
09 - Dự đoán mức lương kỳ vọng dựa trên bản mô tả công việc (Job Description)/
                                              ↑ tên folder sẽ đổi thành
                                              "09 - Phân cụm bộ dữ liệu việc làm Việt Nam"
                                              khi nộp bài (user tự đổi)
├── CLAUDE.md                                    ← file này
├── README.md                                    ← hướng dẫn chạy (cho người chấm) — CẦN VIẾT LẠI cho clustering
├── 2026_MS Teams-TB nộp tiểu luận cuối kỳ_23.15.docx.md      ← đề CŨ (regression) — lưu lại để tham chiếu
├── 2026_MS Teams-TB nộp tiểu luận cuối kỳ_23.15(2).docx.md   ← đề MỚI (clustering) — chuẩn
├── Mau tieu luan KHDL_2026.docx                 ← template báo cáo của GV
├── .gitignore
│
├── raw_data/                                    ← .gitignore (1.5+ GB)
│   ├── data.parquet                             (file gốc tải về — toàn bộ 606,878 dòng)
│   ├── raw_data_train.csv                       546,190 dòng / 1,105 MB (90%, đã re-split 90/10)
│   └── raw_data_test.csv                        60,688 dòng / 123 MB (10%)
│
├── clean_data/                                  ← .gitignore — đã rebuild theo schema clustering (17 cột)
│   ├── clean_data_train.csv                     545,480 dòng / 1,037 MB (giữ 99.87%, có flag salary_missing)
│   └── clean_data_test.csv                      60,606 dòng / 115 MB (giữ 99.86%)
│
├── features/                                    ← .gitignore — đã rebuild (Stage 3 clustering schema)
│   ├── X_train.npz / X_test.npz                 sparse CSR 545k×25,899 (508 MB) / 60k×25,899 (59 MB)
│   ├── X_train_svd.npy / X_test_svd.npy         dense float32 545k×100 (218 MB) / 60k×100 (24 MB)
│   ├── transformers.joblib                      dict: num_imputer, num_scaler, ohe, mlb, tfidf, svd (10 MB)
│   ├── meta.json                                groups + explained_variance + tham số
│   └── feature_names.txt                        25,899 tên feature sparse
│
├── models/                                      ← .gitignore — đã rebuild (clustering artifacts)
│   ├── kmeans_best.joblib                       MiniBatchKMeans (k=6, fit full train) — 244 KB
│   ├── gmm_best.joblib                          GaussianMixture (BIC best k=15, fit sample 30k) — 24 KB
│   ├── labels_train.npy / labels_test.npy       nhãn cụm 545k / 61k — 2.2 MB / 243 KB
│   ├── cluster_profiles.csv                     6 cluster × {n, pct, mean_salary, top industry/province/...}
│   ├── cluster_names.json                       nhãn tự gán (auto, user edit tay sau cho gọn)
│   ├── k_search_metrics.csv                     inertia/silhouette/DB/CH theo k ∈ [3, 20]
│   ├── gmm_bic_aic.csv                          BIC/AIC theo k ∈ [3, 15]
│   ├── demo_10_test_samples.csv                 10 mẫu test demo (theo yêu cầu template)
│   └── clustering_summary.json                  metadata tổng kết
│
├── pyproject.toml                               ← uv dep manifest
├── .python-version                              3.12
├── uv.lock                                      reproducibility snapshot
│
├── notebooks/
│   ├── _download_and_split.py                   ← tái tạo raw_data (đã chuyển sang chunked-read, 90/10)
│   ├── 01_EDA.ipynb                             ✅ XONG (đã migrate sang clustering, sample 150k để fit RAM)
│   ├── 02_Cleaning.ipynb                        ✅ XONG (salary thành 3 feature + flag, chunked apply full)
│   ├── 03_Feature_Engineering.ipynb             ✅ XONG (bỏ y, +TruncatedSVD 100D giữ 75.6% variance, chunked apply)
│   └── 04_Modeling.ipynb                        ✅ XONG (MiniBatchKMeans k=6 + GMM compare + profile + nhãn + demo)
│
├── report/                                      ← để PDF báo cáo cuối cùng
└── slide/                                       ← để PDF slide cuối cùng
```

---

## 4. Pipeline 4 stage (đã chuyển sang clustering)

| # | Notebook | Input | Output | Trạng thái |
|---|---|---|---|---|
| 1 | `01_EDA.ipynb` | `raw_data/raw_data_{train,test}.csv` (train sample 150k) | Biểu đồ + nhận xét + 3 hàm parse + phát hiện trục cluster tiềm năng | ✅ Done |
| 2 | `02_Cleaning.ipynb` | `raw_data/raw_data_{train,test}.csv` | `clean_data/clean_data_{train,test}.csv` (17 cột, salary là 3 feature + flag) | ✅ Done |
| 3 | `03_Feature_Engineering.ipynb` | `clean_data/clean_data_{train,test}.csv` | Sparse CSR 25,899D + dense SVD 100D (75.6% variance) | ✅ Done |
| 4 | `04_Modeling.ipynb` | `features/*` | Clustering artifacts (xem mục models/ ở Section 3) | ✅ Done |

---

## 5. Các quyết định thiết kế đã chốt cho **phiên bản clustering** (KHÔNG đổi nếu không bàn lại)

### `salary` — giờ là FEATURE, không phải target
- Parse từ cột `salary` qua **4 regex pattern** (giữ nguyên từ EDA): `vnd_range_dotted`, `vnd_range_month`, `vnd_single_dotted`, `usd_range`.
- USD → VND theo tỉ giá `25,000`.
- **Tách thành 3 feature numeric:** `salary_min`, `salary_max`, `salary_mid` (= `(min+max)/2`).
- **Outlier handling:**
  - DROP dòng có sentinel `... -999.000.000 VND` (33 dòng — junk thật).
  - DROP dòng `salary_mid > 100M` không sentinel (~609 dòng — outlier scale, sẽ phá K-Means/StandardScaler).
- **Dòng không parse được** (`"Đang cập nhật"`, `"Thoả thuận"`, `"0 VND"`, ...): **GIỮ LẠI**. Đánh dấu `salary_missing = True` (binary flag), 3 cột numeric → NaN, stage 3 impute median.

### `years_exp` (từ `experience_level`)
- Bản chất là **numeric** (số năm KN), không phải ordinal categorical.
- Regex `(\d+)\s*năm`; `"Dưới 1 năm"` → `0.5`; `"Không"`/`"Chưa cập nhật"` → `NaN`.
- **Detect junk bucket rule-based:** `n_rows < 500` AND `pos_missing_rate > 0.5` → set `years_exp = NaN`. Đã phát hiện bucket `9.0` (228 dòng, position 99.6% missing).

### `province` (từ `location`)
- Regex **2 lớp**: (1) match 63 tỉnh/thành; (2) fallback tra district→province cho HCM & Hà Nội.
- Không match → `'Other'`. Coverage hiện tại: **~94%** train, ~94% test.

### `industries_list` (từ `job_industry`)
- **Multi-label**, tách dấu `/`. Top-50 ngành base + `'Other'`. Cover 99.97%.
- Lưu vào CSV dạng chuỗi pipe-separated, vd. `"Bán hàng - Kinh doanh|Marketing"`.
- **KHÔNG tách dấu `-`**.

### Categorical ngắn (`education_level`, `job_type`, `job_position`)
- `lower().strip()`; thiếu/empty → `'unknown'`.

### Text dài (`job_title`, `company_name`, `job_description`, `requirements`, `benefits`)
- `strip()`; NaN → `''`.

### `year`
- Giữ nguyên numeric.

### Feature engineering (stage 3) — schema mới
- TF-IDF riêng từng cột text + OHE categorical + multi-hot industries + numeric impute median + StandardScaler.
- **MỚI:** sau khi `hstack` thành sparse CSR ~30k chiều, áp **TruncatedSVD** (n_components=50–100, random_state=42) để giảm xuống dense matrix. Lưu thêm `X_*_svd.npy`. Lý do: K-Means/silhouette **không scale** với sparse 30k chiều.
- KHÔNG có `y` nữa.

### Clustering (stage 4) — pipeline mới
- **Trên dữ liệu SVD-reduced.** Có thể normalize L2 để dùng cosine similarity ngầm trên K-Means.
- **Tìm `k` tối ưu:** elbow (inertia) + silhouette curve cho `k ∈ [3, 20]`. Vì 458k dòng → silhouette sample 30k–50k rows.
- **Thuật toán so sánh (chọn ≥ 2):**
  - `MiniBatchKMeans` (baseline scale tốt, deterministic với `random_state=42`).
  - `GaussianMixture` (soft clustering, BIC/AIC để chọn k).
  - `HDBSCAN` (density-based, không cần định k trước).
- **Metric đánh giá:** silhouette, Davies-Bouldin, Calinski-Harabasz, inertia.
- **Mô tả cụm:** với mỗi cluster:
  - Top-5 industries, top job_position, top province, top education.
  - Mean/median salary, mean years_exp, year distribution.
  - Top TF-IDF tokens (centroid trong SVD space → inverse_transform → top words).
- **Gán nhãn cụm:** thủ công sau khi xem profile (vd. *"IT mid-level HCM"*, *"Manufacturing Bắc Bộ junior"*).
- **Visualization:** t-SNE hoặc UMAP từ SVD space → 2D scatter colored by cluster label.
- **Demo trên test (yêu cầu của template báo cáo):** sau khi đã chọn được model + nhãn cụm cuối cùng, random 10 mẫu từ test set, predict cluster, hiển thị: cluster_id, nhãn cụm, một số trường chính (`job_title`, `salary`, `province`, `industries_list`, `years_exp`, `education_level`). Đây là phần demo trong báo cáo/slide.

---

## 6. Convention code & notebook

- **Ngôn ngữ:** mọi markdown + comment trong notebook viết **tiếng Việt có dấu**.
- **Style code:** clean code — không bịa hàm placeholder, không over-comment, function < 30 dòng nếu có thể.
- **Random state:** luôn `42` xuyên suốt.
- **Cột tạm trong EDA:** prefix `_` (`_salary_mid_M`, `_years_exp`, `_province`, `_pattern`). Cột chính thức trong clean_data: không prefix.
- **Không leak:** mọi tham số fit trên train, áp dụng nguyên cho test (junk_buckets, TOP_INDUSTRIES, scaler, SVD, KMeans centroids...).
- **Notebook structure:** mỗi notebook = section đánh số (1, 2, 3...), mỗi section có markdown header trước cell code, có in stats/diagnostic sau bước biến đổi quan trọng.

### Cách tái tạo dữ liệu
```powershell
# Cài dependencies:
pip install pandas pyarrow scikit-learn requests matplotlib seaborn jupyter nbformat nbconvert hdbscan umap-learn

# Tải raw data:
python notebooks/_download_and_split.py

# Sinh clean data:
jupyter nbconvert --to notebook --execute notebooks/02_Cleaning.ipynb --output notebooks/02_Cleaning.ipynb
```

---

## 7. Lưu ý quan trọng cho Claude (phiên chat mới)

- **Đề đã đổi từ regression sang clustering** (xem Section 1). Mọi reference đến "target", "y", "MAE", "R²", "Ridge/Lasso/LightGBM" trong code cũ đều cần xem xét lại trước khi giữ.
- **Memory caution:** load full `raw_data_train.csv` (982 MB) + nhiều cột derived rất tốn RAM (~3–5 GB). Tránh `.copy()` toàn DataFrame.
- **Encoding:** Windows console mặc định cp1252 không in được tiếng Việt — luôn set `$env:PYTHONIOENCODING = "utf-8"` trước khi chạy python qua PowerShell.
- **Không tự ý sửa data flow** đã chốt ở Section 5 — bàn lại với user nếu thấy cần đổi.
- **Confirm trước khi push/PR/destructive op** — user đã có remote `git@github.com:anhquan1111/khdl-vietnam-salary-prediction.git` (branch `master`). Lưu ý: tên repo có chữ "salary-prediction" — sẽ đổi sau khi user xong việc đăng ký.
- **Đừng commit `raw_data/` hay `clean_data/` hay `features/` hay `models/`** — đã gitignore, tổng > 2 GB.
- **`_*.py` trong notebooks/** là script tạm (builder, fix), prefix `_` để dễ nhận biết và đã được gitignore (trừ `_download_and_split.py` là script tái tạo dữ liệu chính thức).

---

## 8. Bước tiếp theo (TODO — sau khi đổi đề)

### Việc đã làm (đề CŨ — regression)
- ✅ Stage 1 EDA (regression-flavored — narrative cần đổi)
- ✅ Stage 2 Cleaning (1 quyết định cần sửa: cách xử lý dòng không parse được salary)
- ✅ Stage 3 Feature Engineering (32k features sparse — cần thêm SVD)
- ✅ Stage 4 Modeling regression (Ridge MAE=2.57, LightGBM CV MAE=2.38) — **sẽ KHÔNG dùng cho bài nộp mới**

### Migration sang clustering (đang đi từng bước)
- [x] **Stage 0 Re-split** — `_download_and_split.py` đã chuyển sang chunked-read + 90/10. Train 546,190 / Test 60,688.
- [x] **Stage 1 EDA** — đã migrate narrative + sửa quyết định outlier (drop sentinel + drop >100M không sentinel) + đã re-run thành công. EDA dùng sample 150k vì full 546k load vào pandas sẽ OOM (~3–5 GB). Kết luận (parse_rate, pattern distribution, junk bucket detection) khớp với phiên bản full trước đó.
- [x] **Stage 2 Cleaning** — đã migrate. Salary thành 3 feature (`salary_min`, `salary_max`, `salary_mid`) + flag `salary_missing`. Schema clean_data có 17 cột (thay vì 14). Junk bucket `9.0` vẫn detect (rule-based: n<500 & pos_missing>0.5). Áp dụng full data qua chunked-read 50k row/batch + write incremental. **Retain rate 99.87% / 99.86%** (cao hơn ~5% so với regression cũ vì giữ dòng `salary_missing`). Train clean = 545,480 dòng / 1,037 MB; Test clean = 60,606 dòng / 115 MB.
- [x] **Stage 3 Feature Engineering v2** — đã migrate + cải thiện. Bỏ y. Numeric 6 cột (3 salary + flag + years_exp + year). Sparse X = **25,664 chiều** (sau VN stopwords + max_df=0.85). **TruncatedSVD 150D giữ 77.72% variance** (tăng từ 100D/75.6%). Fit transformers + SVD trên sample 150k, transform full train chunked 50k/batch → vstack → SVD transform once. Output `X_{train,test}.npz` + `X_{train,test}_svd.npy` + transformers. **Cải thiện v2**: thêm VN stopwords list (~80 từ function words + pronouns) để TF-IDF không bị nhiễu bởi "theo/có/của/...", giảm max_df 0.95→0.85, tăng SVD 100→150.
- [x] **Stage 4 Modeling v2** — viết lại + Optuna study. MiniBatchKMeans k-search grid ∈ [3, 25] với 4 metric. **Optuna 30 trials TPE** tuning (k, init, whitening_strength). Logic: compare grid best vs Optuna best → chọn cái có silhouette cao hơn. Kết quả: **grid wins k=5** (sil=0.1279), Optuna chọn k=21 nhưng tệ hơn (sil=0.0959). **Whitening không có tác dụng** (best ≈ 0). GMM compare BIC/AIC → k=14. Final metrics: sil=**0.1279** (cải thiện 6.6% so v1=0.1200), DB=**2.21** (cải thiện 4.3% so v1=2.31), CH=3249. 5 cluster sạch (không trùng nhãn như v1 có 6): CSKH junior, Sales junior, CSKH senior 23tr, Kế toán mid, Other/missing. Total runtime ~15 phút.

### Việc bên ngoài code (user tự lo)
- [ ] Đăng ký lại tên đề tài trên Google Sheets — **deadline CN 24/5/2026**.
- [ ] Đổi tên folder nộp bài thành `09 - Phân cụm bộ dữ liệu việc làm Việt Nam`.
- [ ] Điền tên + MSSV vào [README.md](./README.md) — phân công nhiệm vụ.
- [ ] Viết báo cáo PDF (15–20 trang) theo [Mau tieu luan KHDL_2026.docx](./Mau%20tieu%20luan%20KHDL_2026.docx).
- [ ] Slide PDF ≤ 15 phút.
- [ ] In bản cứng báo cáo.
- [ ] Chuẩn bị 2–3 máy tính dự phòng.
