# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Read the prepared data for training and evaluation (report Section 4.3).

All approaches load the same files:
    data/processed/<dataset>/arrays/X_raw.npy, y.npy, arrays_manifest.json
    data/processed/<dataset>/seed<S>/split_indices.parquet, preprocessing.npz
    data/processed/<dataset>/seed<S>/alpha<A>/partition.parquet
"""
# Every record has a "row_id": its row number in arrays/X_raw.npy and y.npy.
# The split and partition files only store row_ids, so loading data always
# means: (1) choose the needed row_ids with select_rows / portion_clients,
# (2) read and transform those rows with load_xy.
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

# Codes used in the "portion" column of split_indices.parquet and
# partition.parquet (3 = training record left out by subsampling, not used).
TRAIN, VAL, TEST = 0, 1, 2


# lru_cache remembers the result, so the files are opened only once per
# process even though many functions call arrays(...).
@lru_cache(maxsize=2)
def arrays(root, dataset):
    base = Path(root) / dataset / "arrays"
    # feature order, class order and counts written by build_arrays.py
    meta = json.loads((base / "arrays_manifest.json").read_text())
    # mmap_mode="r": the large feature array stays on disk and only the rows
    # that are actually used are read (CIC-IoT2023's file is about 3 GB)
    X = np.load(base / "X_raw.npy", mmap_mode="r")
    y = np.load(base / "y.npy")          # one class code per record (small)
    return X, y, meta


def seed_dir(root, dataset, seed):
    # e.g. data/processed/cicids2017/seed42
    return Path(root) / dataset / f"seed{seed}"


@lru_cache(maxsize=4)
def preprocessing(root, dataset, seed):
    # preprocessing.npz was written by split_data.py from the TRAINING rows only
    p = np.load(seed_dir(root, dataset, seed) / "preprocessing.npz")
    _, _, meta = arrays(root, dataset)
    # safety check: the arrays must not have been rebuilt after splitting
    if list(p["feature_names"]) != meta["features"]:
        raise RuntimeError("Feature order differs between arrays and preprocessing")
    # keep      = column numbers of the features used as inputs (constant ones removed)
    # mean/scale = training mean and standard deviation of each kept feature
    # transform = "log-standard" (sign(x)*log(1+|x|) then standardise) or "standard"
    # names     = names of the kept features
    return {"keep": p["kept_index"], "mean": p["mean"], "scale": p["scale"],
            "transform": str(p["transform"]), "names": [str(n) for n in p["kept_names"]]}


def dataset_info(root, dataset, seed):
    """(input features after constant removal, number of classes, class names)."""
    _, _, meta = arrays(root, dataset)
    # d_in = number of kept features (82, 81 for CIC-IDS2017 seed 1618, or 39)
    # C    = number of classes (7 for CIC-IDS2017, 8 for CIC-IoT2023)
    return len(preprocessing(root, dataset, seed)["keep"]), len(meta["classes"]), meta["classes"]


@lru_cache(maxsize=4)
def assignment(root, dataset, seed, alpha=None, suffix=""):
    """(row_id, portion, client) arrays; client is None for the global split."""
    sdir = seed_dir(root, dataset, seed)
    if alpha is None:
        # global split (no gateways): which portion each record belongs to
        t = pq.read_table(sdir / "split_indices.parquet",
                          columns=["row_id", "portion"]).to_pandas()
        return t["row_id"].to_numpy(), t["portion"].to_numpy(), None
    # gateway partition for this alpha, e.g. seed42/alpha0.1 or seed42/alpha1_dev2
    t = pq.read_table(sdir / f"alpha{float(alpha):g}{suffix}" / "partition.parquet").to_pandas()
    return t["row_id"].to_numpy(), t["portion"].to_numpy(), t["client"].to_numpy()


def select_rows(root, dataset, seed, alpha, portion, client=None, suffix=""):
    """Row ids (ascending) of one portion, optionally for one gateway."""
    rid, por, cli = assignment(root, dataset, seed, alpha, suffix)
    mask = por == portion                # e.g. only training records (portion 0)
    if client is not None:
        mask &= cli == client            # ... and only those of gateway `client`
    return rid[mask]


def portion_clients(root, dataset, seed, alpha, portion, suffix=""):
    """Row ids of a portion and the gateway each one belongs to."""
    # used for the test portion: every test record together with its gateway,
    # so results can be computed for the whole test set and per gateway
    rid, por, cli = assignment(root, dataset, seed, alpha, suffix)
    mask = por == portion
    return rid[mask], cli[mask]


def load_xy(root, dataset, seed, rows, chunk=262_144):
    """Frozen training-fitted transform applied to the given rows, in order."""
    X, y, _ = arrays(root, dataset)
    pp = preprocessing(root, dataset, seed)
    rows = np.asarray(rows, dtype=np.int64)
    # output array: one row per requested record, one column per kept feature
    out = np.empty((len(rows), len(pp["keep"])), dtype=np.float32)
    # work in blocks of 262,144 rows to limit memory use
    for i in range(0, len(rows), chunk):
        # read these rows from disk, keep only the input features, use 64-bit
        # floats for the arithmetic
        block = X[rows[i:i + chunk]][:, pp["keep"]].astype(np.float64)
        if pp["transform"] == "log-standard":
            # compress very large values: sign(x) * log(1 + |x|)
            block = np.sign(block) * np.log1p(np.abs(block))
        # standardise with the TRAINING mean and standard deviation, so no
        # information from validation or test data reaches the model
        out[i:i + chunk] = (block - pp["mean"]) / pp["scale"]
    # inputs as a float tensor [n, d_in]; labels as integer class codes [n]
    return torch.from_numpy(out), torch.from_numpy(y[rows].astype(np.int64))
