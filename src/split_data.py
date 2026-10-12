# Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town 
# 2026

"""Stage 3: global 70/10/20 split, training-only subsample and preprocessing.

Run from the project root after build_arrays.py:
    python src/split_data.py --dataset cicids2017
    python src/split_data.py --dataset ciciot2023

Defaults follow Chapters 3 and 4:
  * every source label is split 70/10/20 (stratified); CIC-IDS2017 keeps
    flows that share a Flow ID in the same portion (group-aware), CIC-IoT2023
    has no session identifier and uses a record-level split;
  * a split is accepted only if validation and test contain every class;
  * optional computational subsampling acts on the training portion only;
    validation and test are never reduced (CIC-IoT2023 default: per source
    label keep min(n, max(ceil(0.05 n), 10000)) training records);
  * constant training features are removed, then the transform is fitted on
    the final training portion only.

Outputs in data/processed/<dataset>/seed<S>/:
    split_indices.parquet  row_id, portion (0 train, 1 validation, 2 test,
                           3 training record left out by subsampling)
                           and group_id (CIC-IDS2017)
    preprocessing.npz      feature order, removed features, means and scales
    initial_state.pt       matched initial MLP parameters for this seed
    schedule.json          21 of 105 gateways per round, rounds 1..100
    split_summary.json     achieved proportions, counts, rules and checksums
"""
# What one seed produces, in short: every record gets a portion number
# (0 train, 1 validation, 2 test, 3 unused). Only the training records are then
# used to decide which features to keep and how to scale them. Each of the five
# seeds gives a different split, so results can be averaged over five repeats.
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from schema import PROCESSED, require_project_root

# let this script import fliids (in flower-ids/), so the split, the initial model
# and the schedule use exactly the same code as the Flower app
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flower-ids"))
from fliids.seeds import make_schedule, rng, torch_seed  # noqa: E402

# portion numbers stored in split_indices.parquet
PORTIONS = {0: "train", 1: "validation", 2: "test", 3: "train_unused"}
# target shares: 70 % train, 10 % validation, 20 % test
TARGETS = np.array([0.70, 0.10, 0.20])
# (fraction, floor) of training records kept per source label: CIC-IDS2017
# keeps all (1.0); CIC-IoT2023 keeps 5 % but at least 10,000 (or all if fewer)
DEFAULT_SUBSAMPLE = {"cicids2017": (1.0, 0), "ciciot2023": (0.05, 10000)}
# flow: records with the same Flow ID go to the same portion; none: each record
# is placed on its own
DEFAULT_GROUPING = {"cicids2017": "flow", "ciciot2023": "none"}


# SHA-256 fingerprint of a file, read in 4 MB blocks
def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------- split
def record_split(strata, g):
    """Exact 70/10/20 within every stratum (record level)."""
    portion = np.zeros(len(strata), dtype=np.int8)
    for s in np.unique(strata):
        # shuffle the records of this source label; the first 20 % go to test, the
        # next 10 % to validation and the rest (about 70 %) stay train (0)
        idx = g.permutation(np.flatnonzero(strata == s))
        n_test = int(round(TARGETS[2] * len(idx)))
        n_val = int(round(TARGETS[1] * len(idx)))
        portion[idx[:n_test]] = 2
        portion[idx[n_test:n_test + n_val]] = 1
    return portion


def group_split(strata, groups, g):
    """Assign whole groups to portions, stratified by each group's majority
    source label. Multi-record groups are placed at random with probabilities
    0.7/0.1/0.2; single-record groups then fill the remaining targets exactly
    where possible. Achieved proportions are recorded in split_summary.json."""
    # Step 1: give every group the source label that most of its records have
    n_strata = int(strata.max()) + 1
    n_groups = int(groups.max()) + 1
    size = np.bincount(groups, minlength=n_groups)
    pair = groups.astype(np.int64) * n_strata + strata
    uniq, cnt = np.unique(pair, return_counts=True)
    g_of, s_of = uniq // n_strata, uniq % n_strata
    order = np.lexsort((s_of, -cnt, g_of))         # per group: most records first
    lead = np.r_[0, np.flatnonzero(np.diff(g_of[order])) + 1]
    g_stratum = np.empty(n_groups, dtype=np.int64)
    g_stratum[g_of[order][lead]] = s_of[order][lead]

    # Step 2: for each label, place its groups. Groups of 2 or more records are
    # placed at random (70/10/20 chance); groups of 1 record then fill whatever
    # each portion still needs to reach its target
    group_portion = np.zeros(n_groups, dtype=np.int8)
    for s in np.unique(g_stratum):
        members = g.permutation(np.flatnonzero(g_stratum == s))
        target = TARGETS * size[members].sum()
        multi = members[size[members] > 1]
        group_portion[multi] = g.choice(3, size=len(multi), p=TARGETS)
        filled = np.bincount(group_portion[multi], weights=size[multi], minlength=3)
        singles = members[size[members] == 1]
        need = np.clip(target - filled, 0, None)
        if len(singles):
            share = need / need.sum() * len(singles) if need.sum() > 0 \
                else TARGETS * len(singles)
            # turn the shares into whole numbers that add up to len(singles)
            counts = np.floor(share).astype(int)
            for k in np.argsort(-(share - counts))[: len(singles) - counts.sum()]:
                counts[k] += 1
            cut = np.cumsum(counts)
            group_portion[singles[:cut[0]]] = 0
            group_portion[singles[cut[0]:cut[1]]] = 1
            group_portion[singles[cut[1]:]] = 2
    # every record gets the portion of its group
    return group_portion[groups]


def subsample_training(portion, strata, fraction, floor, g):
    """Keep min(n, max(ceil(fraction*n), floor)) training records per source
    label; the rest become portion 3 (unused). Validation/test untouched."""
    kept = {}
    if fraction >= 1:
        return portion, kept
    portion = portion.copy()
    for s in np.unique(strata):
        # training records of this source label; keep k of them at random and mark
        # the rest as portion 3 (they are never used)
        idx = np.flatnonzero((portion == 0) & (strata == s))
        k = min(len(idx), max(math.ceil(fraction * len(idx)), floor))
        drop = g.permutation(idx)[k:]
        portion[drop] = 3
        kept[int(s)] = (len(idx), k)
    return portion, kept


# ------------------------------------------------------------- preprocessing
def fit_preprocessing(X, rows, transform, chunk=250_000):
    """Constant-feature removal and standardisation on training rows only."""
    n_feat = X.shape[1]
    # One pass over the training rows, in blocks of 250,000, collecting per feature:
    # the smallest and largest raw value, and the sum and sum of squares of the
    # transformed value (enough to compute mean and standard deviation)
    lo = np.full(n_feat, np.inf)
    hi = np.full(n_feat, -np.inf)
    total = np.zeros(n_feat)
    squares = np.zeros(n_feat)
    for i in range(0, len(rows), chunk):
        raw = X[rows[i:i + chunk]].astype(np.float64)
        lo = np.minimum(lo, raw.min(0))
        hi = np.maximum(hi, raw.max(0))
        # log-standard: z = sign(x) * ln(1 + |x|) squeezes very large values (bytes/s
        # can reach millions) before standardising
        z = np.sign(raw) * np.log1p(np.abs(raw)) if transform == "log-standard" else raw
        total += z.sum(0)
        squares += (z ** 2).sum(0)
    keep = np.flatnonzero(hi > lo)                 # constant in training data
    # mean and standard deviation (scale) of z per feature
    mean = total / len(rows)
    var = np.maximum(squares / len(rows) - mean ** 2, 0.0)
    scale = np.sqrt(var)
    # avoid dividing by zero later
    scale[scale == 0] = 1.0
    return keep, mean[keep], scale[keep]


# ---------------------------------------------------------------------- main
def main():
    require_project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cicids2017", "ciciot2023"], required=True)
    # the five master seeds of the study
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 271, 1618, 3141])
    ap.add_argument("--grouping", choices=["auto", "flow", "none"], default="auto",
                    help="auto: Flow ID groups for CIC-IDS2017, none for CIC-IoT2023")
    # --train-fraction / --train-floor change the subsampling rule (default above)
    ap.add_argument("--train-fraction", type=float)
    ap.add_argument("--train-floor", type=int)
    ap.add_argument("--transform", choices=["log-standard", "standard"],
                    default="log-standard")
    # used only to write schedule.json (which gateways train in which round)
    ap.add_argument("--clients", type=int, default=105)
    ap.add_argument("--clients-per-round", type=int, default=21)
    ap.add_argument("--rounds", type=int, default=100)
    # a split missing a class in validation or test is redrawn, up to 20 times
    ap.add_argument("--max-attempts", type=int, default=20)
    args = ap.parse_args()

    base = PROCESSED / args.dataset
    arrays = base / "arrays"
    meta = json.loads((arrays / "arrays_manifest.json").read_text())
    features, classes = meta["features"], meta["classes"]
    # X is opened as a memory map: rows are read from disk only when needed
    X = np.load(arrays / "X_raw.npy", mmap_mode="r")
    y = np.load(arrays / "y.npy")
    if X.shape != (meta["n_records"], len(features)) or len(y) != len(X):
        raise SystemExit("Array shapes do not match arrays_manifest.json")
    fraction, floor = DEFAULT_SUBSAMPLE[args.dataset]
    fraction = fraction if args.train_fraction is None else args.train_fraction
    floor = floor if args.train_floor is None else args.train_floor
    grouping = DEFAULT_GROUPING[args.dataset] if args.grouping == "auto" else args.grouping

    # strata = source label of every record as a number (the split is stratified
    # by source label, so each fine label is split 70/10/20 on its own)
    labels = pq.read_table(arrays / "index.parquet", columns=["source_label"],
                           read_dictionary=["source_label"]).column(0).to_pandas()
    labels = pd.Categorical(labels)
    strata = labels.codes.astype(np.int32)
    label_names = list(labels.categories)
    groups = None
    # group number of every record: records with the same '<day>|<Flow ID>' share it
    if grouping == "flow":
        keys = pq.read_table(arrays / "index.parquet", columns=["group_key"]).column(0)
        groups = pd.factorize(keys.to_pandas(), sort=False)[0].astype(np.int64)
    from fliids.models import make_model, state_sha256   # torch only needed here
    import torch

    for seed in args.seeds:
        t0 = time.time()
        # ---- 1. split: try attempt 0, 1, 2, ... until validation and test both
        # contain every class (each attempt uses its own random stream)
        for attempt in range(args.max_attempts):
            g = rng(seed, "split", attempt)
            portion = (record_split(strata, g) if groups is None
                       else group_split(strata, groups, g))
            missing = {name: [classes[c] for c in range(len(classes))
                              if not np.any(y[portion == p] == c)]
                       for p, name in ((1, "validation"), (2, "test"))}
            if not missing["validation"] and not missing["test"]:
                break
            print(f"seed {seed}: attempt {attempt} rejected, missing {missing}")
        else:
            raise SystemExit(f"seed {seed}: no accepted split in {args.max_attempts} attempts")
        # ---- 2. subsample the training portion (CIC-IoT2023 only by default)
        portion, kept = subsample_training(portion, strata, fraction, floor,
                                           rng(seed, "subsample"))
        # ---- 3. fit preprocessing on the final training records only (no information
        # from validation or test is used)
        train_rows = np.flatnonzero(portion == 0)
        keep, mean, scale = fit_preprocessing(X, train_rows, args.transform)
        kept_set = set(keep.tolist())
        removed = [features[i] for i in range(len(features)) if i not in kept_set]

        out = base / f"seed{seed}"
        out.mkdir(exist_ok=True)
        # ---- 4. write the outputs of this seed
        cols = {"row_id": np.arange(len(y), dtype=np.int64), "portion": portion}
        if groups is not None:
            cols["group_id"] = groups
        pq.write_table(pa.table(cols), out / "split_indices.parquet", compression="zstd")
        np.savez(out / "preprocessing.npz",
                 feature_names=np.array(features), kept_index=keep,
                 kept_names=np.array([features[i] for i in keep]),
                 removed_names=np.array(removed, dtype=str), mean=mean, scale=scale,
                 transform=np.array(args.transform), n_fit=np.array(len(train_rows)))
        # starting weights of the MLP for this seed; every method (centralized, local,
        # FedAvg, FedProx) starts from this same file
        model = make_model(len(keep), len(classes), torch_seed(seed, "init"))
        torch.save(model.state_dict(), out / "initial_state.pt")
        # which 21 gateways train in each of the 100 rounds (same for every alpha and
        # for FedAvg and FedProx)
        schedule = make_schedule(seed, args.clients, args.clients_per_round, args.rounds)
        (out / "schedule.json").write_text(json.dumps({
            "seed": seed, "clients": args.clients,
            "clients_per_round": args.clients_per_round, "rounds": args.rounds,
            "schedule": schedule}))

        def counts(mask):
            return {classes[c]: int(np.sum(y[mask] == c)) for c in range(len(classes))}
        # ---- 5. split_summary.json: counts, achieved shares and checksums (unused
        # training records are counted as train when reporting the 70/10/20 shares)
        before_sub = np.where(portion == 3, 0, portion)
        summary = {
            "dataset": args.dataset, "seed": seed, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "accepted_attempt": attempt, "grouping": grouping,
            "n_groups": None if groups is None else int(groups.max() + 1),
            "targets": {"train": 0.7, "validation": 0.1, "test": 0.2},
            "achieved_before_subsampling": {
                PORTIONS[p]: round(float(np.mean(before_sub == p)), 6) for p in range(3)},
            "achieved_by_class_before_subsampling": {
                classes[c]: {PORTIONS[p]: int(np.sum((y == c) & (before_sub == p)))
                             for p in range(3)} for c in range(len(classes))},
            "fine_labels_missing_in_validation": [
                label_names[s] for s in np.unique(strata) if not np.any(
                    (strata == s) & (portion == 1))],
            "fine_labels_missing_in_test": [
                label_names[s] for s in np.unique(strata) if not np.any(
                    (strata == s) & (portion == 2))],
            "subsample_rule": {"fraction": fraction, "floor": floor,
                               "applies_to": "training portion only"},
            "subsample_by_source_label": {label_names[s]: {"available": a, "kept": k}
                                          for s, (a, k) in kept.items()},
            "records": {"train": int(np.sum(portion == 0)),
                        "validation": int(np.sum(portion == 1)),
                        "test": int(np.sum(portion == 2)),
                        "train_unused": int(np.sum(portion == 3))},
            "train_by_class": counts(portion == 0),
            "validation_by_class": counts(portion == 1),
            "test_by_class": counts(portion == 2),
            "preprocessing": {"transform": args.transform, "fitted_on": "final training portion",
                              "n_fit": int(len(train_rows)), "kept_features": len(keep),
                              "removed_constant_features": removed},
            "initial_state_sha256": state_sha256(model.state_dict()),
            "inputs": {"arrays_manifest_sha256": sha256(arrays / "arrays_manifest.json")},
            "outputs": {name: sha256(out / name) for name in
                        ("split_indices.parquet", "preprocessing.npz", "schedule.json")},
            "seconds": round(time.time() - t0, 1),
        }
        (out / "split_summary.json").write_text(json.dumps(summary, indent=2))
        r = summary["records"]
        print(f"seed {seed}: train {r['train']:,}  validation {r['validation']:,}  "
              f"test {r['test']:,}  unused train {r['train_unused']:,}  "
              f"features {len(keep)} (removed {removed})", flush=True)
    print("SPLIT_OK")


if __name__ == "__main__":
    main()
