"""
Script một lần: tải data.parquet từ Hugging Face, chia 90/10 và xuất CSV thô.
Chạy: python _download_and_split.py

Tỉ lệ 90/10 theo yêu cầu của template báo cáo "Mau tieu luan KHDL_2026":
- Train (90%, ~546k dòng): huấn luyện và lựa chọn mô hình.
- Test (10%, ~61k dòng): với bài clustering, chỉ random 10 mẫu để demo.
"""
from pathlib import Path
import sys
import requests
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions/resolve/main/data.parquet"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw_data"
PARQUET_PATH = RAW_DIR / "data.parquet"
TRAIN_CSV = RAW_DIR / "raw_data_train.csv"
TEST_CSV = RAW_DIR / "raw_data_test.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)


def download():
    if PARQUET_PATH.exists() and PARQUET_PATH.stat().st_size > 200 * 1024 * 1024:
        print(f"[skip] {PARQUET_PATH.name} đã tồn tại ({PARQUET_PATH.stat().st_size/1e6:.1f} MB)")
        return
    print(f"[download] {URL}")
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(PARQUET_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 / total
                    sys.stdout.write(f"\r  {done/1e6:7.1f} / {total/1e6:7.1f} MB ({pct:5.1f}%)")
                    sys.stdout.flush()
        print()
    print(f"[ok] saved -> {PARQUET_PATH}")


def split_and_save(batch_size: int = 50_000, test_ratio: float = 0.10):
    """Chunked split: tránh load toàn bộ parquet vào RAM (606k dòng × 14 cột ~ vài GB).

    Đọc parquet theo batch, dùng bool mask deterministic (random_state=42)
    để phân mỗi dòng vào train hoặc test, ghi append vào 2 CSV.
    """
    pf = pq.ParquetFile(PARQUET_PATH)
    n = pf.metadata.num_rows
    cols = [f.name for f in pf.schema_arrow]
    print(f"[info] tổng {n:,} dòng × {len(cols)} cột | cột: {cols}")

    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_test = int(round(n * test_ratio))
    test_mask = np.zeros(n, dtype=bool)
    test_mask[perm[:n_test]] = True
    print(f"[plan ] train = {n - n_test:,} ({(1-test_ratio)*100:.0f}%) | test = {n_test:,} ({test_ratio*100:.0f}%)")

    for p in (TRAIN_CSV, TEST_CSV):
        if p.exists():
            p.unlink()

    train_first = test_first = True
    offset = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        m = test_mask[offset:offset + len(df)]
        train_part = df.loc[~m]
        test_part = df.loc[m]
        if len(train_part):
            train_part.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig",
                              mode="w" if train_first else "a", header=train_first)
            train_first = False
        if len(test_part):
            test_part.to_csv(TEST_CSV, index=False, encoding="utf-8-sig",
                             mode="w" if test_first else "a", header=test_first)
            test_first = False
        offset += len(df)
        sys.stdout.write(f"\r  processed {offset:>9,}/{n:,} ({offset*100/n:5.1f}%)")
        sys.stdout.flush()
    print()

    print(f"[ok] -> {TRAIN_CSV.name} ({TRAIN_CSV.stat().st_size/1e6:.1f} MB)")
    print(f"[ok] -> {TEST_CSV.name}  ({TEST_CSV.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    download()
    split_and_save()
