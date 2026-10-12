# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Stage 2: store every cleaned record as fixed arrays (no sampling here).

Run from the project root after clean_data.py:
    python src/build_arrays.py --dataset cicids2017
    python src/build_arrays.py --dataset ciciot2023

Outputs in data/processed/<dataset>/arrays/:
    X_raw.npy             float32 [N, F] untransformed features; row_id = row number
    y.npy                 int8 [N] class code (class order in the manifest)
    index.parquet         row_id, source_file, source_row, source_label, class,
                          group_key (CIC-IDS2017 only)
    arrays_manifest.json  feature order, class order, counts and SHA-256 checksums

Sampling for computation, if any, happens later and only inside the training
portion (split_data.py), as Chapter 3 requires.
"""
# Why arrays: the cleaned Parquet parts are good for storage but slow to pick
# rows from. Here all records are copied, in order, into one big feature matrix
# X_raw.npy and one label vector y.npy. Record number i (row_id = i) is row i of
# both files and row i of index.parquet, which says where the record came from.
# Later steps refer to records only by row_id.
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from schema import PROCESSED, require_project_root


# SHA-256 fingerprint of a file, read in 4 MB blocks
def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    require_project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cicids2017", "ciciot2023"], required=True)
    args = ap.parse_args()
    base = PROCESSED / args.dataset
    # ---- checks: cleaning must have finished without an error
    if (base / "cleaning_error.json").exists():
        raise SystemExit(f"The last clean_data.py run stopped (see "
                         f"{base / 'cleaning_error.json'}). Fix the problem and "
                         "rerun clean_data.py first.")
    manifest = json.loads((base / "data_manifest.json").read_text())
    # column order and class order come from data_manifest.json, so they are the
    # same as in cleaning
    features, classes = manifest["feature_columns"], manifest["class_order"]
    parts = sorted((base / "clean").glob("part-*.parquet"))
    if not parts:
        raise SystemExit("No cleaned parts found. Run clean_data.py first.")
    # N = total number of cleaned records (read from the file headers only)
    n = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    if n != manifest["rows_kept"]:
        raise SystemExit(f"Cleaned parts hold {n} rows; manifest says "
                         f"{manifest['rows_kept']}. Rerun clean_data.py.")

    out = base / "arrays"
    out.mkdir(exist_ok=True)
    # ---- create X_raw.npy on disk with room for N x F numbers and fill it part by
    # part (a memory map: the whole matrix never has to fit in memory at once)
    X = np.lib.format.open_memmap(out / "X_raw.npy", mode="w+", dtype=np.float32,
                                  shape=(n, len(features)))
    # class code of every record (fits in 1 byte)
    y = np.empty(n, dtype=np.int8)
    # columns copied to index.parquet (never used as model inputs)
    id_cols = ["source_file", "source_row", "source_label", "class"]
    if args.dataset == "cicids2017":
        id_cols.append("group_key")
    # row = number of records written so far = row_id of the next record
    writer, row = None, 0
    for p in parts:
        t = pq.read_table(p).to_pandas()
        k = len(t)
        # features of this part go into rows row ... row+k-1
        X[row:row + k] = t[features].to_numpy(np.float32)
        # class name -> class number using the class order (-1 = unknown class)
        codes = pd.Categorical(t["class"], classes).codes
        if (codes < 0).any():
            raise SystemExit(f"{p.name}: class outside the class order")
        y[row:row + k] = codes
        # the index rows of this part, with row_id as the first column
        idx = t[id_cols].copy()
        idx.insert(0, "row_id", np.arange(row, row + k, dtype=np.int64))
        idx["source_row"] = idx["source_row"].astype(np.int64)
        table = pa.Table.from_pandas(idx, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out / "index.parquet", table.schema,
                                      compression="zstd")
        writer.write_table(table)
        row += k
    writer.close()
    # write everything to disk and close the files
    X.flush()
    del X
    np.save(out / "y.npy", y)
    # quick check of the first 100,000 rows: no NaN or infinite values
    if not np.isfinite(np.load(out / "X_raw.npy", mmap_mode="r")[: min(n, 100000)]).all():
        raise SystemExit("Non-finite values found in X_raw.npy")

    # ---- arrays_manifest.json: counts per class and the checksums of the three
    # files; later steps copy these checksums into their own records
    counts = np.bincount(y, minlength=len(classes))
    meta = {
        "dataset": args.dataset,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_records": int(n),
        "features": features,
        "classes": classes,
        "class_codes": {c: i for i, c in enumerate(classes)},
        "by_class": {c: int(counts[i]) for i, c in enumerate(classes)},
        "source_manifest_sha256": sha256(base / "data_manifest.json"),
        "sha256": {name: sha256(out / name)
                   for name in ("X_raw.npy", "y.npy", "index.parquet")},
    }
    (out / "arrays_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("n_records", "by_class")}, indent=2))
    print("ARRAYS_OK")


if __name__ == "__main__":
    main()
