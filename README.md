# Nhóm 09 — Phân cụm bộ dữ liệu việc làm Việt Nam

Tiểu luận cuối kỳ môn **Khoa học Dữ liệu**, năm học 2025–2026.

- **Đề tài (số 2 — cập nhật 23/5/2026):** Phân cụm bộ dữ liệu việc làm Việt Nam (clustering, unsupervised).
- **Input:** Toàn bộ dataset (cả `salary` cũng là feature, không phải target).
- **Output:**
  1. Số cụm và hiệu suất phân cụm (silhouette, Davies-Bouldin, Calinski-Harabasz, inertia).
  2. Thuộc tính chung (phổ biến) của các phần tử thuộc mỗi cụm.
  3. Nhãn tự gán cho mỗi cụm (vd. *"Sales/Business Development — Lương + hoa hồng cao ~23tr"*).
- **Dataset gốc:** [tinixai/vietnamese-job-descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions) — 606,878 dòng × 14 cột (Parquet, ~293 MB).

---

## 1. Cấu trúc thư mục nộp bài

```
09 - Phân cụm bộ dữ liệu việc làm Việt Nam/
├── README.md                          ← file này (hướng dẫn chạy)
├── training.ipynb                     ← orchestrator gọi 5 stages (KHÔNG cần execute lúc demo)
├── testing.ipynb                      ← demo notebook (Run All ~30s lúc bảo vệ 5 phút)
├── report/
│   └── BaoCao.pdf                     ← bản PDF báo cáo (giống 100% bản in)
├── slide/
│   └── Slide.pdf                      ← bản PDF slide 10 phút
├── notebooks/
│   ├── _download_and_split.py         ← tải parquet + split 90/10 chunked
│   ├── _embed_jds.py                  ← embed SBERT mpnet-base (GPU)
│   ├── 01_EDA.ipynb                   ← khảo sát + 14 biểu đồ
│   ├── 02_Cleaning.ipynb              ← clean → 17 cột (salary thành 3 feature + flag)
│   ├── 03_Feature_Engineering.ipynb   ← TF-IDF + SVD 150D
│   ├── 04_Modeling.ipynb              ← MiniBatchKMeans + GMM + Optuna (pipeline chính)
│   └── 05_Embedding_Clustering.ipynb  ← SBERT 768D + KMeans (so sánh)
├── raw_data/
│   ├── raw_data_train.csv             ← dữ liệu thô tập train (90%, 546k dòng)
│   └── raw_data_test.csv              ← dữ liệu thô tập test (10%, 60.6k dòng)
└── clean_data/
    ├── clean_data_train.csv           ← clean train (17 cột, 545.5k dòng)
    └── clean_data_test.csv            ← clean test (60.6k dòng)
```

> Tên `raw_data_{train,test}.csv`, `clean_data_{train,test}.csv`, `training.ipynb`, `testing.ipynb` đặt **đúng nguyên văn** theo yêu cầu GV (thông báo cập nhật 25/5/2026).

---

## 2. Cách chạy

Cần Python ≥ 3.10, < 3.13. Dự án dùng [**uv**](https://docs.astral.sh/uv/) để quản lý dependencies (snapshot trong `uv.lock`).

```powershell
# Nếu chưa có uv:
#   PowerShell: irm https://astral.sh/uv/install.ps1 | iex
#   hoặc:       pip install uv

# Cài Python 3.12 + tất cả thư viện (pandas, sklearn, sentence-transformers, torch, ...):
uv sync

# Mở Jupyter Lab:
uv run jupyter lab
```

### 2.1 Chạy demo nhanh (cho người chấm)

Trong lúc defense, GV chỉ cần chạy 1 file:

```powershell
# Mở testing.ipynb → Restart & Run All (~30 giây)
uv run jupyter nbconvert --to notebook --execute testing.ipynb `
    --output testing.ipynb
```

`testing.ipynb` load model đã train sẵn → demo 10 mẫu test + visualization t-SNE + bảng so sánh metric 2 pipeline.

### 2.2 Train từ đầu (reproduce toàn bộ, mất ~90 phút có GPU)

Chạy `training.ipynb` (Restart & Run All) — orchestrator gọi tuần tự 5 stages. Hoặc chạy thủ công từng stage:

| Bước | Stage | Time (GPU RTX 4060) | Output |
|---|---|---|---|
| 0 | `_download_and_split.py` | ~2 phút | `raw_data/*.csv` |
| 1 | `01_EDA.ipynb` | ~3 phút | 14 ảnh `figures/eda_*.png` |
| 2 | `02_Cleaning.ipynb` | ~5 phút | `clean_data/*.csv` (17 cột) |
| 3 | `03_Feature_Engineering.ipynb` | ~10 phút | `features/X_*.npz` + `X_*_svd.npy` |
| 3.5 | `_embed_jds.py` | ~50 phút (GPU) | `features/X_*_embed.npy` |
| 4 | `04_Modeling.ipynb` | ~15 phút | `models/kmeans_best.joblib` + profile + nhãn |
| 5 | `05_Embedding_Clustering.ipynb` | ~10 phút | `models/kmeans_embed_best.joblib` + so sánh |

**Lưu ý**: Stage 3.5 (SBERT embedding) cần GPU NVIDIA + CUDA. Trên CPU mất 3-5 giờ.
**Reproducibility:** mọi bước random đều dùng `random_state=42`.

---

## 3. Pipeline + kết quả

### 3.1 Pipeline 1 — TF-IDF + KMeans (chính)

```
clean_data → numeric+OHE+industries+TF-IDF (25k chiều) → TruncatedSVD 150D → MiniBatchKMeans k=5
```

| Metric | Giá trị |
|---|---|
| Silhouette | **0.128** |
| Davies-Bouldin | **2.21** |
| Calinski-Harabasz | 3249 |
| Số cụm | **5** |

**5 cụm phát hiện:**
- C0 (21.6%): Nhân viên junior phổ thông
- C1 (28.6%): Nhân viên đa ngành mid-junior
- C2 (19.9%): **Sales/Business Development lương cao + commission**
- C3 (24.5%): Kế toán/Kiểm toán & Kỹ thuật văn phòng
- C4 (5.4%): Tin lương thoả thuận (Manager/IT specialist)

### 3.2 Pipeline 2 — SBERT + KMeans (so sánh)

```
clean_data → SBERT mpnet-base 768D → L2-normalize → MiniBatchKMeans k=5
```

| Metric | Giá trị |
|---|---|
| Silhouette | 0.068 |
| Davies-Bouldin | 3.31 |
| Calinski-Harabasz | 1406 |
| Số cụm | 5 |

**5 cụm SBERT (phân ngành rõ hơn TF-IDF):**
- C0: Service/CSKH & Giáo dục junior
- C1: Sales & Customer-facing mid-level
- C2: **Marketing & Education professionals**
- C3: **Kế toán/Kiểm toán/Tài chính** (sạch hơn pipeline 1)
- C4: **Engineering & Industrial** (kỹ thuật chuyên môn)

### 3.3 Finding

TF-IDF có metric cao hơn nhưng SBERT phân cụm theo NGÀNH NGHỀ sạch hơn. Trade-off: metric numeric vs interpretability. Stage 5 cluster có ý nghĩa kinh tế phong phú hơn (có Marketing, Engineering riêng biệt).

---

## 4. Schema dataset thô (14 cột)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | int | Mã định danh bài đăng |
| `job_title` | str | Chức danh công việc |
| `company_name` | str | Tên công ty |
| `salary` | str | Thông tin lương (parse thành 3 feature numeric) |
| `location` | str | Địa điểm làm việc |
| `job_type` | str | Hình thức (full-time, part-time, ...) |
| `job_industry` | str | Ngành nghề (multi-label, tách `/`) |
| `experience_level` | str | Kinh nghiệm (parse → numeric years_exp) |
| `education_level` | str | Yêu cầu học vấn |
| `job_position` | str | Cấp bậc vị trí |
| `job_description` | str | Mô tả công việc (TF-IDF + SBERT) |
| `benefits` | str | Quyền lợi |
| `requirements` | str | Yêu cầu ứng viên |
| `year` | int | Năm đăng (2022–2026) |

---

## 5. Thành viên nhóm & phân công

> Mỗi thành viên tự soạn phần báo cáo + slide tương ứng với mảng code mình phụ trách. Nhóm trưởng tổng hợp + biên tập định dạng cuối.

| STT | Họ và tên | MSSV | Nhiệm vụ |
|---|---|---|---|
| 1 | **Võ Trần Anh Quân** (nhóm trưởng) | 102230131 | Notebook `03_Feature_Engineering.ipynb` (TF-IDF + SVD 150D), `04_Modeling.ipynb` (MiniBatchKMeans + GMM + Optuna, pipeline chính), `05_Embedding_Clustering.ipynb` (SBERT mpnet-base, so sánh), `training.ipynb` + `testing.ipynb`. Quản lý repo, README, đóng gói nộp bài. Soạn phần báo cáo + slide cho mảng Feature Engineering, Modeling, So sánh 2 pipeline. Tổng hợp & biên tập báo cáo + slide cuối cùng. |
| 2 | Phạm Hữu Tuân | 102230139 | Notebook `01_EDA.ipynb` — khảo sát + 14 biểu đồ (phân phối salary, years_exp, ngành, tỉnh thành, train vs test) + nhận xét đặc tính dữ liệu + phát hiện junk bucket. Soạn phần báo cáo + slide cho mảng EDA. |
| 3 | Trần Việt Toàn | 102230137 | Notebook `02_Cleaning.ipynb` — parse `salary` thành 3 feature (`salary_min`, `salary_max`, `salary_mid` + flag `salary_missing`), `years_exp`, `province` (regex 2 lớp), `industries_list` multi-label, junk bucket detection. Chunked apply + write incremental cho full data 545k. Soạn phần báo cáo + slide cho mảng tiền xử lý & làm sạch. |

---

## 6. Ghi chú khi nộp bài

- Kiểm tra `report/BaoCao.pdf` **giống 100%** bản in nộp đầu giờ thi (bookmark, mục lục, định dạng).
- **Không đưa mã nguồn vào quyển báo cáo.** Độ dài 15–20 trang (không kể mục lục & TLTK).
- Slide PDF **tối đa 10 phút** + chạy demo `testing.ipynb` **tối đa 5 phút**.
- Tên folder upload: `09 - Phân cụm bộ dữ liệu việc làm Việt Nam`.
- Upload qua link GV gửi trên MS Teams **lúc 21h ngày 26/5/2026** (đóng link 21h15).
- Mang sẵn **2-3 máy tính** phòng sự cố lúc bảo vệ.

---

## 7. Hardware requirements để reproduce

- CPU: ≥ 8 cores (chunked processing 545k dòng)
- RAM: ≥ 16 GB
- GPU (optional nhưng khuyến nghị): NVIDIA RTX (CUDA 12.x) — không có GPU thì stage 3.5 SBERT chạy 3-5h trên CPU
- Disk: ≥ 5 GB free
