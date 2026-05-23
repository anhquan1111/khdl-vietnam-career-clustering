# Nhóm 09 — Dự đoán mức lương kỳ vọng dựa trên bản mô tả công việc (Job Description)

Tiểu luận cuối kỳ môn **Khoa học Dữ liệu**, năm học 2025–2026.

- **Đề tài (số 2):** Dự đoán mức lương kỳ vọng dựa trên bản mô tả công việc.
- **Output:** `lương kỳ vọng = (lương min + lương max) / 2` (ví dụ: "23–28 triệu" ⇒ 25.5 triệu).
- **Input:** Tất cả các trường còn lại của dataset.
- **Dataset gốc:** [tinixai/vietnamese-job-descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions) — 606,878 dòng × 14 cột (Parquet, ~293 MB).

---

## 1. Cấu trúc thư mục nộp bài

```
09 - Dự đoán mức lương kỳ vọng dựa trên bản mô tả công việc (Job Description)/
├── README.md                          ← file này (hướng dẫn chạy)
├── report/
│   └── BaoCao.pdf                     ← bản PDF quyển báo cáo (phải khớp 100% bản in)
├── slide/
│   └── Slide.pdf                      ← bản PDF slide trình bày
├── notebooks/
│   ├── 01_EDA.ipynb                   ← khảo sát & trực quan hoá
│   ├── 02_Cleaning.ipynb              ← làm sạch dữ liệu thô → clean
│   ├── 03_Feature_Engineering.ipynb   ← trích đặc trưng từ clean
│   └── 04_Modeling.ipynb              ← huấn luyện & đánh giá mô hình
├── raw_data/
│   ├── raw_data_train.csv             ← dữ liệu thô tập train (80%)
│   └── raw_data_test.csv              ← dữ liệu thô tập test (20%)
└── clean_data/
    ├── clean_data_train.csv           ← dữ liệu đã làm sạch (chưa feature engineering)
    └── clean_data_test.csv
```

> Các tên `raw_data_{train,test}.csv` và `clean_data_{train,test}.csv` được đặt **đúng nguyên văn** theo yêu cầu trong thông báo nộp tiểu luận.

---

## 2. Trình tự chạy chương trình

Cần Python ≥ 3.10, < 3.13. Dự án dùng [**uv**](https://docs.astral.sh/uv/) để quản lý dependencies (cài 1 lần, snapshot trong `uv.lock` để tái lập y hệt).

```powershell
# Nếu chưa có uv:
#   PowerShell: irm https://astral.sh/uv/install.ps1 | iex
#   hoặc:       pip install uv

# Cài Python 3.12 (nếu máy chưa có) + tất cả thư viện (pandas, sklearn, lightgbm, jupyter…):
uv sync

# Mở Jupyter Notebook UI:
uv run jupyter notebook

# Hoặc chạy 1 notebook end-to-end qua CLI:
uv run jupyter nbconvert --to notebook --execute notebooks/04_Modeling.ipynb `
    --output 04_Modeling.ipynb --ExecutePreprocessor.timeout=3600
```

`uv sync` tự đọc `pyproject.toml` + tạo `.venv/` + cài đúng version. Lần đầu mất ~1-2 phút, các lần sau gần như tức thì.

Chạy lần lượt 4 notebook theo thứ tự:

| Bước | Notebook                          | Input                                   | Output                                                                |
|------|-----------------------------------|-----------------------------------------|-----------------------------------------------------------------------|
| 1    | `01_EDA.ipynb`                    | `raw_data/raw_data_{train,test}.csv`    | biểu đồ + nhận xét (in-notebook)                                      |
| 2    | `02_Cleaning.ipynb`               | `raw_data/raw_data_{train,test}.csv`    | `clean_data/clean_data_{train,test}.csv`                              |
| 3    | `03_Feature_Engineering.ipynb`    | `clean_data/clean_data_{train,test}.csv`| `features/X_{train,test}.npz`, `y_*.npy`, `transformers.joblib`, `meta.json` |
| 4    | `04_Modeling.ipynb`               | `features/*`                            | `models/lgbm_best.txt`, `predictions_test.csv`, `metrics.csv` + figures |

**Lưu ý reproducibility:** mọi bước random đều dùng `random_state=42`.

---

## 3. Nguồn dữ liệu thô

Dữ liệu thô được lấy từ Hugging Face theo link sau:

```
https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions/resolve/main/data.parquet
```

Sau khi tải, dữ liệu được chia ngẫu nhiên **80% train / 20% test** (`random_state=42`) và xuất ra hai file CSV trong `raw_data/`. Script tải/chia gốc lưu tại `notebooks/_download_and_split.py` (có thể chạy lại để tái tạo).

### Schema (14 cột)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | int | Mã định danh bài đăng |
| `job_title` | str | Chức danh công việc |
| `company_name` | str | Tên công ty |
| **`salary`** | str | **Thông tin lương dạng văn bản — dùng để tạo nhãn `expected_salary`** |
| `location` | str | Địa điểm làm việc |
| `job_type` | str | Hình thức (full-time, part-time, remote…) |
| `job_industry` | str | Ngành nghề |
| `experience_level` | str | Yêu cầu kinh nghiệm |
| `education_level` | str | Yêu cầu học vấn |
| `job_position` | str | Cấp bậc vị trí |
| `job_description` | str | Mô tả công việc |
| `benefits` | str | Quyền lợi |
| `requirements` | str | Yêu cầu ứng viên |
| `year` | int | Năm đăng (2022–2026) |

---

## 4. Thành viên nhóm & phân công

> Mỗi thành viên tự soạn phần báo cáo + slide tương ứng với mảng code mình phụ trách. Nhóm trưởng tổng hợp, biên tập và thống nhất định dạng cuối.

| STT | Họ và tên | MSSV | Nhiệm vụ |
|-----|-----------|------|----------|
| 1 | **Võ Trần Anh Quân** (nhóm trưởng) | 102230131 | Notebook `03_Feature_Engineering.ipynb` (TF-IDF + OHE + multi-hot, 32k cột) và `04_Modeling.ipynb` (Baseline / Ridge / Lasso-SGD / LightGBM-Optuna / Ensemble + year ablation). Quản lý repo, README, đóng gói nộp bài. Soạn phần báo cáo + slide cho mảng Feature Engineering và Modeling. Tổng hợp & biên tập báo cáo + slide cuối cùng. |
| 2 | Phạm Hữu Tuân | 102230139 | Notebook `01_EDA.ipynb` — khảo sát & trực quan hoá (phân phối lương, năm kinh nghiệm, ngành nghề, tỉnh thành) + nhận xét đặc tính dữ liệu. Soạn phần báo cáo + slide cho mảng EDA. |
| 3 | Trần Việt Toàn | 102230137 | Notebook `02_Cleaning.ipynb` — parse `expected_salary` (4 regex pattern), `years_exp`, `province` (2 lớp regex + district lookup), `industries_list` multi-label, xử lý junk bucket. Soạn phần báo cáo + slide cho mảng tiền xử lý & làm sạch dữ liệu. |

---

## 5. Ghi chú khi nộp bài

- Kiểm tra file `report/BaoCao.pdf` **giống 100%** bản in nộp đầu giờ thi (bookmark, mục lục, định dạng).
- **Không đưa mã nguồn vào quyển báo cáo.** Độ dài 15–20 trang (không kể mục lục & TLTK).
- Đóng gói toàn bộ thư mục theo tên `09 - Dự đoán mức lương kỳ vọng dựa trên bản mô tả công việc (Job Description)` rồi upload qua link GV gửi trên MS Teams **lúc 21h ngày 26/5/2026** (đóng link 21h15).
