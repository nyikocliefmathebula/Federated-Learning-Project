# Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town
# 2026

"""Cross-gateway test of the local-only models (an extra evaluation view).

Run from the project root after the local-only runs of a dataset have finished:
    python src/cross_gateway.py --dataset cicids2017
    python src/cross_gateway.py --dataset ciciot2023

Each local-only run saves its 105 gateway models in local_models.pt. This script
scores every model k on the test records of every gateway j (105 x 105 pairs per
seed and alpha): the diagonal (k = j) is the model on its own gateway, the other
10,920 pairs show how a gateway's model does on traffic of other gateways. No
model is trained or changed and only the test portion is read.

Writes, for every run, results/runs/<dataset>/local/seed<S>/alpha<A>/
cross_gateway.parquet (one row per model and gateway) and prints one summary line.
Writes results/summary/<dataset>_cross_gateway.csv (one row per run).
"""
# This script only reads saved files; it is safe to run again (it overwrites its
# own outputs). Run it when no grid is running in the same terminal session.
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# let this script import fliids, the code of the Flower app
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flower-ids"))
from fliids.data import TEST, dataset_info, load_xy, portion_clients  # noqa: E402
from fliids.evaluation import confusion_by_client, metrics, predict, save_parquet  # noqa: E402
from fliids.models import make_model  # noqa: E402

RESULTS = Path("results/runs")
SEEDS = [13, 42, 271, 1618, 3141]
ALPHAS = [100.0, 10.0, 1.0, 0.5, 0.1]


def pair_rows(k, y_true, pred, client, n_clients, n_classes):
    """Model k on the test records of each gateway j: one row per j with the
    number of records, accuracy, macro-F1 over the classes present in T_j and the
    false-positive rate (None when T_j has no benign records)."""
    # pred: model k's prediction for every test record, in row order
    cms = confusion_by_client(y_true, pred, client, n_clients, n_classes)
    rows = []
    for j in range(n_clients):
        m = metrics(cms[j])
        rows.append({"model": k, "gateway": j, "n": m["n"], "accuracy": m["accuracy"],
                     "macro_f1_present": m["macro_f1_present"], "fpr": m["fpr"]})
    return rows


def summarise(table):
    """Own-gateway (k = j) against other-gateway (k != j) means for one run."""
    own = table[table.model == table.gateway]
    other = table[table.model != table.gateway]
    return {"own_accuracy": own.accuracy.mean(), "other_accuracy": other.accuracy.mean(),
            "other_accuracy_median": other.accuracy.median(),
            "other_share_below_0.9": (other.accuracy < 0.9).mean(),
            "own_macro_f1_present": own.macro_f1_present.mean(),
            "other_macro_f1_present": other.macro_f1_present.mean(),
            "own_fpr": own.fpr.astype(float).mean(), "other_fpr": other.fpr.astype(float).mean()}


def main():
    if not Path("data/processed").is_dir():
        raise SystemExit("Run from the project root: cd /mnt/c/Users/Nyiko/fl-iot-ids")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    p.add_argument("--data-root", default="data/processed")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    root, ds, summary = args.data_root, args.dataset, []
    for seed in args.seeds:
        n_features, n_classes, _ = dataset_info(root, ds, seed)
        for alpha in args.alphas:
            run = RESULTS / ds / "local" / f"seed{seed}" / f"alpha{alpha:g}"
            if not (run / "metrics.json").exists():
                print(f"skip {run}: no finished local-only run")
                continue
            # all test records of this partition and the gateway each belongs to
            rows, client = portion_clients(root, ds, seed, alpha, TEST)
            x, y = load_xy(root, ds, seed, rows)
            y_true = y.numpy()
            states = torch.load(run / "local_models.pt", weights_only=False)["state_dicts"]
            rows_out = []
            for k in range(len(states)):
                model = make_model(n_features, n_classes, 0)
                model.load_state_dict(states[k])
                pred = predict(model, x)              # model k on every test record
                # scored per gateway at once; only the scores are kept, not the
                # predictions (they would need several GB for CIC-IoT2023)
                rows_out += pair_rows(k, y_true, pred, client, len(states), n_classes)
            table = pd.DataFrame(rows_out)
            save_parquet(table, run / "cross_gateway.parquet")
            s = {"seed": seed, "alpha": alpha, **summarise(table)}
            summary.append(s)
            print(f"seed {seed} alpha {alpha:g}: own {s['own_accuracy']:.4f}, "
                  f"other gateways {s['other_accuracy']:.4f}", flush=True)
    out = Path("results/summary")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / f"{ds}_cross_gateway.csv", index=False)
    print(f"{len(summary)} run(s) written to {out / (ds + '_cross_gateway.csv')}")
    print("CROSS_GATEWAY_OK")


if __name__ == "__main__":
    main()
