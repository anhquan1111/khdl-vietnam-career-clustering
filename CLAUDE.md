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
├── features/                                    ← .gitignore (~530 MB)
│   ├── X_train.npz                              (458,568 × 27,107 sparse CSR, 425 MB)
│   ├── X_test.npz                               (114,773 × 27,107 sparse CSR, 106 MB)
│   ├── y_train.npy / y_test.npy                 log1p(expected_salary)
│   ├── transformers.joblib                      dict: num_imputer, num_scaler, ohe, mlb, tfidf
│   ├── meta.json                                groups (column ranges) + tham số
│   └── feature_names.txt                        27,107 dòng tên feature
│
├── notebooks/
│   ├── _download_and_split.py                   ← tái tạo raw_data từ HuggingFace
│   ├── _build_03.py                             ← builder của 03 (gitignore, để tái tạo notebook nếu cần)
│   ├── 01_EDA.ipynb                             ✅ ĐÃ XONG
│   ├── 02_Cleaning.ipynb                        ✅ ĐÃ XONG
│   ├── 03_Feature_Engineering.ipynb             ✅ ĐÃ XONG
│   └── 04_Modeling.ipynb                        ⏳ skeleton — chưa làm
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
| 3 | `03_Feature_Engineering.ipynb` | `clean_data/clean_data_{train,test}.csv` | `features/X_*.npz`, `y_*.npy`, `transformers.joblib`, `meta.json` | ✅ |
| 4 | `04_Modeling.ipynb` | `features/*` | Bảng so sánh model + nhận xét | ⏳ |

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
- TF-IDF **riêng từng cột** (`job_description`, `requirements`, `benefits`) rồi `hstack`. Không concat trước rồi vectorize 1 lần.

### Training (stage 4)
- **Stratified sample 150k** từ train (tầng theo `job_industry × year`) cho CV 5-fold chọn hyperparam.
- Refit trên FULL train (~459k) với hyperparam tối ưu → evaluate trên test (114k).
- Metric: MAE, RMSE, R² (và optional MAPE) ở thang triệu VNĐ.

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
- [ ] **Stage 4 — Modeling** ([04_Modeling.ipynb](./notebooks/04_Modeling.ipynb)):
  - Baseline: trung bình theo `job_industry × experience_level`.
  - Linear: Ridge/Lasso.
  - Tree: RandomForest, GradientBoosting (LightGBM nếu có).
  - 5-fold CV trên sample 150k → chốt hyperparam → fit full → evaluate test.
  - So sánh: with `year` vs without `year`.
  - Báo cáo MAE/RMSE/R² + feature importance + sai số.
- [ ] **Báo cáo + Slide** theo `Mau tieu luan KHDL_2026.docx`.
- [ ] **Đăng ký tên đề tài** trên Google Sheets (trước CN 24/5).
- [ ] **Điền tên + MSSV** vào [README.md](./README.md#L88) — phân công nhiệm vụ trong nhóm.
