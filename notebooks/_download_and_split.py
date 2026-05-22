"""
Script một lần: tải data.parquet từ Hugging Face, chia 80/20 và xuất CSV thô.
Chạy: python _download_and_split.py
Sau khi xong có thể xoá file này — pipeline chính nằm trong notebook.
"""
from pathlib import Path
import sys
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

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


def split_and_save():
    print(f"[read] {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[info] shape = {df.shape}, columns = {list(df.columns)}")

    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    print(f"[split] train = {len(train_df):,} | test = {len(test_df):,}")

    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")
    print(f"[ok] -> {TRAIN_CSV.name} ({TRAIN_CSV.stat().st_size/1e6:.1f} MB)")
    print(f"[ok] -> {TEST_CSV.name}  ({TEST_CSV.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    download()
    split_and_save()
