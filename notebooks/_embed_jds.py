"""Embed full clean_data train + test bằng PhoBERT/SBERT multilingual.

Output:
- features/X_train_embed.npy  (545,480 × 768 dense float32 ~ 1.6 GB)
- features/X_test_embed.npy   ( 60,606 × 768 dense float32 ~ 175 MB)

Model: paraphrase-multilingual-mpnet-base-v2 (768D, multi-lang, well-tested cho VN).
GPU: RTX 4060 ~200 docs/sec → tổng ~50 phút.

Run: python notebooks/_embed_jds.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / 'clean_data'
FEATURES = ROOT / 'features'
FEATURES.mkdir(exist_ok=True)

MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
BATCH_SIZE = 64
CHUNK_SIZE = 50_000


def build_text(df):
    """Concat 4 text cols thành 1 chuỗi mỗi JD. Handle NaN."""
    df = df.fillna('')
    return (df['job_title'] + '. ' + df['job_description'] + ' ' +
            df['requirements'] + ' ' + df['benefits']).tolist()


def embed_csv(path, out_path, model):
    """Encode full CSV chunked, vstack + save .npy."""
    print(f"\n=== Embedding {path.name} ===", flush=True)

    with open(path, encoding='utf-8-sig') as f:
        n_rows = sum(1 for _ in f) - 1
    print(f"Total rows: {n_rows:,}", flush=True)

    text_cols = ['job_title', 'job_description', 'requirements', 'benefits']
    embeddings = []
    n_done = 0
    t_start = time.time()

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, usecols=text_cols):
        texts = build_text(chunk)
        emb = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False,
                           convert_to_numpy=True)
        embeddings.append(emb.astype(np.float32))
        n_done += len(chunk)
        elapsed = time.time() - t_start
        rate = n_done / elapsed
        eta_sec = (n_rows - n_done) / rate
        print(f"  {n_done:>9,}/{n_rows:,} ({n_done*100/n_rows:5.1f}%) | "
              f"{rate:.0f} docs/sec | elapsed {elapsed/60:.1f}m | ETA {eta_sec/60:.0f}m",
              flush=True)

    all_emb = np.vstack(embeddings)
    print(f"\nFinal shape: {all_emb.shape}, dtype={all_emb.dtype}", flush=True)
    np.save(out_path, all_emb)
    print(f"Saved: {out_path} ({all_emb.nbytes / 1e6:.1f} MB)", flush=True)


def main():
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    print(f"\nLoading model: {MODEL_NAME}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device='cuda')
    print(f"Max seq length: {model.max_seq_length}", flush=True)
    print(f"Output dim    : {model.get_sentence_embedding_dimension()}", flush=True)
    print(f"Batch size    : {BATCH_SIZE}", flush=True)

    # TEST trước (60k dòng, ~5 phút) để verify pipeline
    embed_csv(CLEAN / 'clean_data_test.csv',
              FEATURES / 'X_test_embed.npy', model)

    # TRAIN (545k dòng, ~45 phút)
    embed_csv(CLEAN / 'clean_data_train.csv',
              FEATURES / 'X_train_embed.npy', model)

    print("\n✓ All done.", flush=True)


if __name__ == '__main__':
    main()
