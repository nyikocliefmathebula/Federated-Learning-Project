# Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town 
# 2026

"""Summarises completed runs of the grid in one table (one row per run).

Run from the project root at any time, also while the grid is running:
    python src/collect_results.py --dataset cicids2017
    python src/collect_results.py --dataset ciciot2023

Writes results/summary/<dataset>_runs.csv and prints how many of the 80 planned
runs per dataset are complete. Test metrics are on the threat-landscape test set
T_glob for centralized, FedAvg and FedProx, and for the routed deployment (each
gateway's records scored by its own model) for local-only. gateway_* columns
summarise the gateway-matched view over the 105 gateways (for centralized, at
the alpha of the row). This table is for monitoring and a first look; the
analysis itself should read the saved confusion counts and trajectories.
"""
# This script only reads the metrics.json file of each run folder; it never
# changes a run. It is safe to run as often as needed.
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# the planned grid: 5 seeds x 5 alphas for each method (centralized: 5 seeds)
RESULTS = Path("results/runs")
SEEDS = [13, 42, 271, 1618, 3141]
ALPHAS = [100.0, 10.0, 1.0, 0.5, 0.1]
METHODS = ["centralized", "local", "fedavg", "fedprox"]


# entries: one dictionary of metrics per gateway (gateway-matched view).
# Returns the mean and the worst (minimum) gateway accuracy, the mean
# macro-F1 over the classes each gateway has, and the mean false-positive rate.
# Gateways with no value (None) are left out of the averages.
def gateway_summary(entries):
    acc = [e["accuracy"] for e in entries if e.get("accuracy") is not None]
    f1 = [e["macro_f1_present"] for e in entries if e.get("macro_f1_present") is not None]
    fpr = [e["fpr"] for e in entries if e.get("fpr") is not None]
    return {"gateway_mean_accuracy": float(np.mean(acc)) if acc else None,
            "gateway_min_accuracy": float(np.min(acc)) if acc else None,
            "gateway_mean_macro_f1_present": float(np.mean(f1)) if f1 else None,
            "gateway_mean_fpr": float(np.mean(fpr)) if fpr else None}


def rows_for(dataset, method, seed):
    """One row per (method, seed, alpha); centralized gets one row per alpha."""
    out = []
    base = RESULTS / dataset / method / f"seed{seed}"
    # centralized has one model per seed (no alpha); its row is repeated for every
    # alpha so that the gateway_* columns can show each alpha's gateway-matched view
    if method == "centralized":
        path = base / "metrics.json"
        m = json.loads(path.read_text()) if path.exists() else None
        for alpha in ALPHAS:
            row = {"dataset": dataset, "method": method, "seed": seed, "alpha": alpha,
                   "complete": m is not None}
            if m:
                t = m["global_test"]
                row.update(test_accuracy=t["accuracy"], test_macro_f1=t["macro_f1"],
                           test_fpr=t["fpr"], best_step=m["best_epoch"],
                           stopped_step=m["stopped_epoch"],
                           best_val_macro_f1=m["best_val_macro_f1"], seconds=m["seconds"])
                row.update(gateway_summary(m["gateway_matched"].get(f"alpha{alpha:g}", [])))
            out.append(row)
        return out
    for alpha in ALPHAS:
        path = base / f"alpha{alpha:g}" / "metrics.json"
        row = {"dataset": dataset, "method": method, "seed": seed, "alpha": alpha,
               "complete": path.exists()}
        if path.exists():
            m = json.loads(path.read_text())
            # local-only: test_* columns = the routed deployment (each gateway's test
            # records scored by its own model)
            if method == "local":
                t = m["routed"]
                row.update(test_accuracy=t["accuracy"], test_macro_f1=t["macro_f1"],
                           test_fpr=t["fpr"],
                           local_threat_landscape_mean_accuracy=float(
                               np.mean(m["threat_landscape_accuracy"])),
                           seconds=m["seconds"])
                row.update(gateway_summary([c["gateway_matched"] for c in m["clients"]]))
            else:
                t = m["global_test"]
                row.update(test_accuracy=t["accuracy"], test_macro_f1=t["macro_f1"],
                           test_fpr=t["fpr"], best_step=m["best_round"],
                           stopped_step=m["stopped_round"],
                           best_val_macro_f1=m["best_val_macro_f1"],
                           # FedAvg/FedProx: also the communication up to the best round and up to the stop
                           serialized_bytes_at_best=m["serialized_bytes_at_best"],
                           serialized_bytes_at_stop=m["serialized_bytes_at_stop"],
                           seconds=m["seconds"])
                row.update(gateway_summary(m["gateway_matched"]))
        out.append(row)
    return out


def main():
    if not Path("src").is_dir():
        raise SystemExit("Run this from the project root:\n  cd /mnt/c/Users/Nyiko/fl-iot-ids")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["cicids2017", "ciciot2023"])
    args = p.parse_args()
    # one row per (method, seed, alpha), written to results/summary/<dataset>_runs.csv
    rows = [r for method in METHODS for seed in SEEDS
            for r in rows_for(args.dataset, method, seed)]
    table = pd.DataFrame(rows)
    out = Path("results/summary")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{args.dataset}_runs.csv", index=False)

    # progress count: centralized counted once per seed (not once per alpha)
    runs = table.drop_duplicates(["method", "seed", "alpha"])
    runs = runs[(runs.method != "centralized") | (runs.alpha == ALPHAS[0])]
    done = runs.groupby("method")["complete"].agg(["sum", "count"])
    print(f"{args.dataset}: {int(done['sum'].sum())} of {int(done['count'].sum())} runs complete")
    for method in METHODS:
        if method in done.index:
            print(f"  {method:12s} {int(done.loc[method, 'sum']):3d} / "
                  f"{int(done.loc[method, 'count'])}")
    # short table on screen: mean test accuracy and macro-F1 over the seeds that
    # are complete so far, per method and alpha
    complete = table[table.complete & table.test_accuracy.notna()] \
        if "test_accuracy" in table else table.iloc[0:0]
    if len(complete):
        shown = complete[(complete.method != "centralized") | (complete.alpha == ALPHAS[0])].copy()
        shown["alpha"] = [("all" if m == "centralized" else f"{a:g}")
                          for m, a in zip(shown.method, shown.alpha)]
        view = (shown.groupby(["method", "alpha"], sort=False)
                [["test_accuracy", "test_macro_f1"]].mean().round(4))
        print("\nMean over completed seeds (centralized does not depend on alpha):")
        print(view.to_string())
    print(f"\nWritten {out / (args.dataset + '_runs.csv')}")
    print("COLLECT_OK")


if __name__ == "__main__":
    main()
