# CLAUDE.md — Hướng dẫn cho phiên làm việc Claude

> File này dành cho Claude (AI) đọc khi bắt đầu phiên chat mới, để nắm được bối cảnh dự án mà không cần đọc lại toàn bộ lịch sử chat.

---

## 1. Bối cảnh dự án

- **Môn học:** Khoa học Dữ liệu (KHDL), năm học 2025–2026.
- **Nhóm:** 09.
- **Đề tài (số 2):** *Dự đoán mức lương kỳ vọng dựa trên bản mô tả công việc (Job Description).*
- **Output (target):** `expected_salary = (lương_min + lương_max) / 2` (đơn vị triệu VNĐ).
- **Input:** tất cả các trường còn lại trong dataset.
- **Dataset:** [tinixai/vietnamese-job-descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions) — 606,878 dòng × 14 cột, Parquet ~293 MB. Đã được split 80/20 (random_state=42) thành `raw_data_{train,test}.csv`.

---

## 2. Quy tắc nộp bài (BẮT BUỘC — vi phạm = 0 điểm)

Trích từ thông báo cuối kỳ ([2026_MS Teams-TB nộp tiểu luận cuối kỳ_23.15.docx.md](./2026_MS%20Teams-TB%20nộp%20tiểu%20luận%20cuối%20kỳ_23.15.docx.md)):

### 2.1 Hạn

| Mốc | Thời gian |
|---|---|
| Đăng ký tên đề tài trên Google Sheets | trước CN 24/5/2026 |
| Link upload bài mở | 21h00 ngày 26/5/2026 |
| Link upload bài đóng | **21h15 ngày 26/5/2026** (chỉ 15 phút) |
| Bản in báo cáo nộp | đầu buổi thi |

### 2.2 Folder bài nộp (đặt tên: `09 - Dự đoán mức lương kỳ vọng dựa trên bản mô tả công việc (Job Description)`)

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
- Mang sẵn **2–3 máy tính** phòng sự cố.

---

## 3. Cấu trúc thư mục hiện tại

```
09 - Dự đoán mức lương kỳ vọng dựa trên bản mô tả công việc (Job Description)/
├── CLAUDE.md                                    ← file này
├── README.md                                    ← hướng dẫn chạy (cho người chấm)
├── 2026_MS Teams-TB nộp tiểu luận cuối kỳ_23.15.docx.md
├── Mau tieu luan KHDL_2026.docx                 ← template báo cáo của GV
├── .gitignore
│
├── raw_data/                                    ← .gitignore (1.5+ GB)
│   ├── data.parquet                             (file gốc tải về)
│   ├── raw_data_train.csv                       (485,502 dòng × 14 cột, 982 MB)
│   └── raw_data_test.csv                        (121,376 dòng × 14 cột, 246 MB)
│
├── clean_data/                                  ← .gitignore (~1 GB)
│   ├── clean_data_train.csv                     (458,568 dòng × 14 cột, 845 MB)
│   └── clean_data_test.csv                      (114,773 dòng × 14 cột, 211 MB)
│
├── features/                                    ← .gitignore (~570 MB)
│   ├── X_train.npz                              (458,568 × 32,107 sparse CSR, 456 MB)
│   ├── X_test.npz                               (114,773 × 32,107 sparse CSR, 114 MB)
│   ├── y_train.npy / y_test.npy                 log1p(expected_salary)
│   ├── transformers.joblib                      dict: num_imputer, num_scaler, ohe, mlb, tfidf
│   ├── meta.json                                groups (column ranges) + tham số
│   └── feature_names.txt                        32,107 dòng tên feature
│
├── models/                                      ← .gitignore (sinh từ stage 4)
│   ├── lgbm_best.txt, lgbm_no_year.txt          LightGBM native (Unicode-safe save)
│   ├── ridge_final.joblib, sgd_final.joblib     sklearn pickled
│   ├── predictions_test.csv                     8 cột (actual + 7 model preds)
│   ├── metrics.csv                              bảng MAE/RMSE/R²
│   ├── hyperparams.json                         best params + ensemble weights
│   └── optuna_trials.json                       full Optuna trial log
│
├── pyproject.toml                               ← uv dep manifest (pandas, sklearn, lightgbm, optuna…)
├── .python-version                              3.12
├── uv.lock                                      reproducibility snapshot
│
├── notebooks/
│   ├── _download_and_split.py                   ← tái tạo raw_data từ HuggingFace
│   ├── _build_03.py, _build_04.py               ← builder (gitignore, nbformat)
│   ├── 01_EDA.ipynb                             ✅ ĐÃ XONG
│   ├── 02_Cleaning.ipynb                        ✅ ĐÃ XONG
│   ├── 03_Feature_Engineering.ipynb             ✅ ĐÃ XONG (32k features)
│   └── 04_Modeling.ipynb                        ⏳ đang chạy (Optuna LGBM refit full)
│
├── report/                                      ← để PDF báo cáo cuối cùng
└── slide/                                       ← để PDF slide cuối cùng
```

---

## 4. Pipeline 4 stage

| # | Notebook | Input | Output | Trạng thái |
|---|---|---|---|---|
| 1 | `01_EDA.ipynb` | `raw_data/raw_data_{train,test}.csv` | Biểu đồ + nhận xét + 3 hàm parse | ✅ |
| 2 | `02_Cleaning.ipynb` | `raw_data/raw_data_{train,test}.csv` | `clean_data/clean_data_{train,test}.csv` | ✅ |
| 3 | `03_Feature_Engineering.ipynb` | `clean_data/clean_data_{train,test}.csv` | `features/X_*.npz` (32k cột), `y_*.npy`, `transformers.joblib`, `meta.json` | ✅ |
| 4 | `04_Modeling.ipynb` | `features/*` | `models/*.txt/joblib/csv/json`, plots, ensemble | ⏳ đang chạy |

---

## 5. Các quyết định thiết kế đã chốt (KHÔNG đổi nếu không bàn lại)

### Target `expected_salary`
- Parse từ cột `salary` qua **4 regex pattern**: `vnd_range_dotted`, `vnd_range_month`, `vnd_single_dotted`, `usd_range`.
- USD → VND theo tỉ giá `25,000`.
- **Drop dòng có `expected_salary > 100 triệu`** (junk; pattern `... -999.000.000 VND` là sentinel max). KHÔNG cap/clip.
- **Modeling target = `log1p(expected_salary)`** (sẽ làm ở stage 4, không lưu vào clean_data).

### `years_exp` (từ `experience_level`)
- Bản chất là **numeric** (số năm KN), không phải ordinal categorical.
- Regex `(\d+)\s*năm`; `"Dưới 1 năm"` → `0.5`; `"Không"`/`"Chưa cập nhật"` → `NaN`.
- **Detect junk bucket rule-based:** `n_rows < 500` AND `pos_missing_rate > 0.5` → set `years_exp = NaN`. Đã phát hiện và xoá bucket `9.0` tự động (225 dòng có position 99.6% missing).

### `province` (từ `location`)
- Regex **2 lớp**: (1) match 63 tỉnh/thành; (2) fallback tra district→province cho HCM & Hà Nội.
- Không match → `'Other'`. Coverage hiện tại: **93.84%** train, 93.82% test.
- **Có thể cải thiện** bằng cách mở rộng `DISTRICT_TO_PROVINCE` cho các tỉnh khác (HP, ĐN, Cần Thơ…).

### `industries_list` (từ `job_industry`)
- **Multi-label**, tách dấu `/`. Top-50 ngành base + `'Other'`. Cover 99.97%.
- Lưu vào CSV dạng chuỗi pipe-separated, vd. `"Bán hàng - Kinh doanh|Marketing"`.
- **KHÔNG tách dấu `-`** — các tên ghép như `"Khoa học - Kỹ thuật"`, `"Điện - Điện tử - Điện lạnh"` là tên ngành do publisher đặt, không phải nhiều ngành.

### Categorical ngắn (`education_level`, `job_type`, `job_position`)
- `lower().strip()`; thiếu/empty → `'unknown'`.

### Text dài (`job_title`, `company_name`, `job_description`, `requirements`, `benefits`)
- `strip()`; NaN → `''`. **Không lowercase ở stage 2** — để stage 3 vectorizer xử lý.

### `year`
- Giữ nguyên numeric. Lạm phát 2022–2026 **không deflate** trong cleaning.
- Stage 4 sẽ train **2 model pair-wise** (có `year` vs không `year`) để báo cáo nhận xét.

### Text features (stage 3)
- TF-IDF **riêng từng cột** (`job_title`, `job_description`, `requirements`, `benefits`) rồi `hstack`. Không concat trước rồi vectorize 1 lần.
- 2 bộ params: **`job_title`** (text ngắn) max=5k, **ngram=(1,2)** bigram, min_df=5; còn lại (text dài) max=10k, unigram, min_df=10.
- Numeric scaler: **`StandardScaler()`** (with_mean=True) — center cả 2 cột. Trước đây dùng `with_mean=False` gây bug `year` scale ~1730 làm SGD/Lasso diverge → đã sửa.

### Training (stage 4)
- **Stratified sample 150k** từ train (tầng theo `first_industry`) cho CV chọn hyperparam.
- **5 model:** Baseline (group-mean) | Ridge (5-fold CV, alpha grid) | Lasso-SGD (5-fold CV, alpha grid) | LightGBM (**Optuna 15-trial TPE + Hyperband pruner, 3-fold CV**) | Ensemble (weighted average 1/MAE của Ridge+Lasso+LightGBM).
- Refit best hyperparam trên FULL train (~459k) → evaluate test (114k) đúng 1 lần.
- Metric: MAE, RMSE, R² ở thang triệu VNĐ (sau `expm1`).
- Year ablation: zero cột year trong CSR (`tocsc` + zero data + `tocsr`) → refit LightGBM với cùng best_params.
- LightGBM save dùng `model_to_string()` + Python write (Unicode-safe path).

### Kết quả thực tế (đã chạy, ngày 2026-05-23)
- Ridge (α=10): Test MAE = **2.573 triệu**, R²(M) = 0.543
- Lasso-SGD (α=1e-5): Test MAE = **2.668 triệu**, nnz = 3,614 / 32,107 (11.3%)
- LightGBM Optuna (best CV MAE = **2.382 triệu**): num_leaves=125 (chạm trần range), lr=0.068, min_child_samples=98, feature_fraction=0.607
- Optuna: 15 trial / 185.7 phút / 6 pruned (40%)
- ⏳ LightGBM refit full + year ablation + ensemble: chưa xong (đang chạy cell 19)
- Kỳ vọng: LightGBM test MAE ~2.30, Ensemble ~2.25

---

## 6. Convention code & notebook

- **Ngôn ngữ:** mọi markdown + comment trong notebook viết **tiếng Việt có dấu**.
- **Style code:** clean code — không bịa hàm placeholder, không over-comment, function < 30 dòng nếu có thể.
- **Random state:** luôn `42` xuyên suốt.
- **Cột tạm trong EDA:** prefix `_` (`_salary_M`, `_years_exp`, `_province`, `_pattern`). Cột chính thức trong clean_data: không prefix.
- **Không leak:** mọi tham số fit trên train, áp dụng nguyên cho test (junk_buckets, TOP_INDUSTRIES…).
- **Notebook structure:** mỗi notebook = section đánh số (1, 2, 3…), mỗi section có markdown header trước cell code, có in stats/diagnostic sau bước biến đổi quan trọng.

### Cách tái tạo dữ liệu
```powershell
# Cài dependencies:
pip install pandas pyarrow scikit-learn requests matplotlib seaborn jupyter nbformat nbconvert

# Tải raw data:
python notebooks/_download_and_split.py

# Sinh clean data:
jupyter nbconvert --to notebook --execute notebooks/02_Cleaning.ipynb --output notebooks/02_Cleaning.ipynb
```

### Cách rebuild notebook nếu bị hỏng (chưa có script builder hiện tại)
Các notebook 01 và 02 đã được sinh bằng builder script tạm (`_build_*.py`) và đã bị xoá sau khi tạo notebook. Nếu cần edit lớn → edit trực tiếp .ipynb qua VS Code hoặc dùng nbformat.

---

## 7. Lưu ý quan trọng cho Claude (phiên chat mới)

- **Memory caution:** load full `raw_data_train.csv` (982 MB) + nhiều cột derived rất tốn RAM (~3–5 GB). Tránh `.copy()` toàn DataFrame; dùng `df.loc[mask].groupby(...)` thay vì `df.dropna(...).copy()`.
- **Encoding:** Windows console mặc định cp1252 không in được tiếng Việt — luôn set `$env:PYTHONIOENCODING = "utf-8"` trước khi chạy python qua PowerShell.
- **Không tự ý sửa data flow** đã chốt ở Section 5 — bàn lại với user nếu thấy cần đổi.
- **Confirm trước khi push/PR/destructive op** — user đã có remote `git@github.com:anhquan1111/khdl-vietnam-salary-prediction.git` (branch `master`).
- **Đừng commit `raw_data/` hay `clean_data/`** — đã gitignore, tổng > 2 GB.
- **`_*.py` trong notebooks/** là script tạm (builder, fix), prefix `_` để dễ nhận biết và đã được gitignore (trừ `_download_and_split.py` là script tái tạo dữ liệu chính thức).

---

## 8. Bước tiếp theo (TODO)

- [x] **Stage 3 — Feature Engineering** ([03_Feature_Engineering.ipynb](./notebooks/03_Feature_Engineering.ipynb)) — ✅ done:
  - `X_train (458568, 27107)` / `X_test (114773, 27107)` sparse CSR, density 0.71%.
  - Numeric impute median (years_exp=3.0) + StandardScaler(with_mean=False); `r(years_exp, y) = +0.33`, `r(year, y) = +0.07`.
  - OHE 94 cột, multi-hot industries 51 cột, TF-IDF (10k + 9087 + 7873 = 26,960 cột) với `sublinear_tf=True`, `min_df=10`, `max_df=0.95`.
  - `y = log1p(expected_salary)`, lưu cùng `features/`. Stage 4 slice cột year qua `meta['groups']['numeric']` để so sánh có/không year.
- [⏳] **Stage 4 — Modeling** ([04_Modeling.ipynb](./notebooks/04_Modeling.ipynb)) — đang chạy:
  - ✅ Baseline (group-mean `industry × years_exp`): MAE 3.93
  - ✅ Ridge CV + refit + test: MAE 2.573 (best α=10)
  - ✅ Lasso-SGD CV + refit + test: MAE 2.668 (best α=1e-5, 11.3% nonzero coef)
  - ✅ LightGBM Optuna 15 trial TPE + Hyperband: best CV MAE 2.382
  - ⏳ LightGBM refit full train (cell 19, đang chạy ~30-60')
  - ⏳ Year ablation (cell 21)
  - ⏳ Ensemble weighted (cell 23)
  - ⏳ Plots (predicted vs actual, residuals, feature importance), per-segment errors, save artifacts
- [ ] **Báo cáo PDF** (15-20 trang) theo `Mau tieu luan KHDL_2026.docx`. **Không được đưa code vào.** Dùng bảng + đồ thị + prose.
- [ ] **Slide PDF** ≤ 15 phút — focus đặc tính dữ liệu + bảng so sánh model.
- [ ] **Đăng ký tên đề tài** trên Google Sheets — **deadline CN 24/5/2026** (gấp!)
- [ ] **Điền tên + MSSV** vào [README.md](./README.md#L88) — phân công nhiệm vụ trong nhóm.
- [ ] **In bản cứng báo cáo** — nộp đầu buổi thi (phải khớp 100% với PDF nộp online).
- [ ] **Chuẩn bị 2-3 máy tính** mang theo phòng sự cố khi trình bày slide.
