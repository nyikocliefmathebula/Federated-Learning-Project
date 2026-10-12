# Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town 
# 2026

"""Check every data-pipeline output before training (report Section 4.8).

Run from the project root after partition_clients.py:
    python src/check_data.py --dataset cicids2017
    python src/check_data.py --dataset ciciot2023

Writes data/processed/<dataset>/data_check.json and prints DATA_CHECK_OK only
if every check passes.
"""
# This script only reads the data files; it changes nothing except writing
# data_check.json. Each check prints PASS or FAIL. Run it after every data
# step has finished and before any training.
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from schema import PROCESSED, require_project_root

# let this script import fliids (model and schedule code of the Flower app)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flower-ids"))


def main():
    require_project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cicids2017", "ciciot2023"], required=True)
    args = ap.parse_args()
    base = PROCESSED / args.dataset
    checks, failures = [], []

    # record one check: its name, whether it passed and an optional detail
    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        if not ok:
            failures.append(name)
        print(("PASS  " if ok else "FAIL  ") + name + (f"  ({detail})" if detail else ""))

    # ---- 1. cleaning outputs (clean_data.py)
    check("last cleaning run completed (no cleaning_error.json)",
          not (base / "cleaning_error.json").exists())
    man = json.loads((base / "data_manifest.json").read_text())
    check("cleaning rows balance",
          man["rows_read"] == man["rows_removed"] + man["rows_kept"],
          f"{man['rows_read']} = {man['rows_removed']} + {man['rows_kept']}")
    audit = pd.read_csv(base / "data_audit.csv")
    last = audit[audit["step_order"] == audit["step_order"].max()]
    check("audit final counts equal kept rows",
          int(last["remaining"].sum()) == man["rows_kept"])

    # ---- 2. arrays (build_arrays.py)
    arr = base / "arrays"
    meta = json.loads((arr / "arrays_manifest.json").read_text())
    X = np.load(arr / "X_raw.npy", mmap_mode="r")
    y = np.load(arr / "y.npy")
    check("arrays hold every kept record", len(y) == man["rows_kept"] == X.shape[0])
    check("class codes valid", y.min() >= 0 and y.max() < len(meta["classes"]))
    # check every value, 500,000 rows at a time (no NaN or infinity anywhere)
    finite = all(np.isfinite(X[i:i + 500_000]).all() for i in range(0, len(X), 500_000))
    check("all features finite", finite)
    check("feature order matches manifest", meta["features"] == man["feature_columns"])

    import torch
    from fliids.models import make_model, state_sha256
    from fliids.seeds import make_schedule

    # ---- 3. each seed folder (split_data.py)
    seed_dirs = sorted(base.glob("seed*"))
    check("at least one seed folder", len(seed_dirs) > 0)
    for sdir in seed_dirs:
        seed = int(sdir.name[4:])
        s = json.loads((sdir / "split_summary.json").read_text())
        split = pq.read_table(sdir / "split_indices.parquet").to_pandas()
        portion = split["portion"].to_numpy()
        # split_indices.parquet must list row_id 0, 1, ..., N-1 exactly once each
        check(f"{sdir.name}: every record once",
              np.array_equal(split["row_id"].to_numpy(), np.arange(len(y))))
        for p, name in ((1, "validation"), (2, "test")):
            present = np.unique(y[portion == p])
            check(f"{sdir.name}: {name} has every class",
                  len(present) == len(meta["classes"]))
        # CIC-IDS2017: all records of one Flow ID group must be in one portion
        # (unused training records count as training here)
        if "group_id" in split:
            pre = np.where(portion == 3, 0, portion)
            per_group = pd.Series(pre).groupby(split["group_id"].to_numpy()).nunique()
            check(f"{sdir.name}: no Flow ID group crosses portions",
                  int(per_group.max()) == 1)
        prep = np.load(sdir / "preprocessing.npz")
        # the number of rows the scaler was fitted on must equal the training size
        check(f"{sdir.name}: scaler fitted on training rows only",
              int(prep["n_fit"]) == int(np.sum(portion == 0)))
        check(f"{sdir.name}: preprocessing feature order",
              list(prep["feature_names"]) == meta["features"])
        # the initial weights must load into the MLP and match the recorded checksum
        state = torch.load(sdir / "initial_state.pt", weights_only=True)
        model = make_model(len(prep["kept_index"]), len(meta["classes"]), 0)
        model.load_state_dict(state)
        check(f"{sdir.name}: initial state checksum",
              state_sha256(state) == s["initial_state_sha256"])
        sch = json.loads((sdir / "schedule.json").read_text())
        # recreate the round schedule from the seed and compare it with the saved one
        check(f"{sdir.name}: schedule reproducible",
              sch["schedule"] == make_schedule(seed, sch["clients"],
                                               sch["clients_per_round"], sch["rounds"]))
        # ---- 4. each partition folder (partition_clients.py)
        for adir in sorted(sdir.glob("alpha*")):
            ps = json.loads((adir / "partition_summary.json").read_text())
            part = pq.read_table(adir / "partition.parquet").to_pandas()
            used = np.flatnonzero(portion < 3)
            # every train/validation/test record appears once, in the same portion as in
            # the split
            check(f"{sdir.name}/{adir.name}: partition covers train/val/test exactly",
                  np.array_equal(part["row_id"].to_numpy(), used) and
                  np.array_equal(part["portion"].to_numpy(), portion[used]))
            K = ps["clients"]
            # every gateway meets the minimum (512 train, 74 validation, 147 test)
            for p, name in ((0, "train"), (1, "validation"), (2, "test")):
                sizes = np.bincount(part["client"].to_numpy()[part["portion"].to_numpy() == p],
                                    minlength=K)
                check(f"{sdir.name}/{adir.name}: {name} minimum per gateway",
                      sizes.min() >= ps["minimums"][name],
                      f"min {sizes.min()} >= {ps['minimums'][name]}")
    # ---- write all results; exit code 1 if any check failed
    record = {"dataset": args.dataset, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "checks": checks, "failed": failures}
    (base / "data_check.json").write_text(json.dumps(record, indent=2))
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    print("DATA_CHECK_OK")


if __name__ == "__main__":
    main()
