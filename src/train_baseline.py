# Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town 
# 2026

"""Centralized and local-only baselines with the shared model and training code.

Running from the project root:
    python src/train_baseline.py --method centralized --dataset cicids2017 --seed 42
    python src/train_baseline.py --method local --dataset cicids2017 --seed 42 --alpha 1

Centralized: one model on the pooled training portion, selected on V_glob,
scored on T_glob (threat landscape) and on every gateway's T_k for each alpha
that has a partition (gateway-matched view).
Local-only: one model per gateway, selected on its V_k, scored on T_glob
(threat landscape), on its own T_k (gateway-matched) and pooled over gateways
(routed deployment). Both keep their optimizer state across epochs.
"""
# The two baselines frame the federated results:
#   centralized - an upper reference: all training data in one place, one model
#   local-only  - a lower reference: each of the 105 gateways trains alone and
#                 never shares anything
# Both use the same MLP, the same initial weights, the same optimizer settings
# and the same early stopping as FedAvg and FedProx, so only the way of training
# differs. Here one 'epoch' is one pass over the training data.
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# let this script import fliids, so the baselines use the exact code of the
# Flower app (data loading, model, training loop, metrics)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flower-ids"))
from fliids.data import (TEST, TRAIN, VAL, dataset_info, load_xy,  # noqa: E402
                         portion_clients, seed_dir, select_rows)
from fliids.evaluation import (confusion, confusion_by_client,  # noqa: E402
                               confusion_rows, metrics, predict, save_parquet)
from fliids.models import load_initial_model  # noqa: E402
from fliids.records import write_run_config  # noqa: E402
from fliids.seeds import torch_seed  # noqa: E402
from fliids.training import EarlyStopping, make_optimizer, train_epochs  # noqa: E402


def fit(model, x, y, xv, yv, args, n_classes, stream_ids, select_key, verbose=False):
    """Train with per-epoch validation; return the best model and history.
    verbose prints one progress line per epoch (used for centralized runs)."""
    # x, y: training data; xv, yv: validation data. select_key is the validation
    # score used to pick the best epoch (macro_f1 or macro_f1_present).
    # stream_ids fix the random streams: (0,) for centralized, (0, k+1) for gateway k
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    generator = torch.Generator().manual_seed(torch_seed(args.seed, "minibatch", *stream_ids))
    torch.manual_seed(torch_seed(args.seed, "dropout", *stream_ids))
    stopper = EarlyStopping(args.patience, args.min_delta)
    m = metrics(confusion(yv.numpy(), predict(model, xv), n_classes))
    # epoch 0 = the untrained initial model
    stopper.update(0, m[select_key], model.state_dict())
    history = [(0, m[select_key], m["accuracy"], None, 0, 0)]
    # train one epoch, score on validation, let early stopping decide; at most
    # 200 epochs
    for epoch in range(1, args.max_epochs + 1):
        s = train_epochs(model, x, y, 1, args.batch_size, optimizer, generator)
        m = metrics(confusion(yv.numpy(), predict(model, xv), n_classes))
        stop = stopper.update(epoch, m[select_key], model.state_dict())
        history.append((epoch, m[select_key], m["accuracy"], s["loss"],
                        s["examples"], s["steps"]))
        if verbose:
            print(f"epoch {epoch}: val macro-F1 {m[select_key]:.4f} (best "
                  f"{stopper.best_score:.4f} @ {stopper.best_step}), misses {stopper.misses}",
                  flush=True)
        if stop:
            break
    # return the model of the best epoch, not of the last one
    model.load_state_dict(stopper.best_state)
    return model, stopper, history


def main():
    if not Path("data/processed").is_dir():
        raise SystemExit("Run from the project root: cd /mnt/c/Users/Nyiko/fl-iot-ids")
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["centralized", "local"], required=True)
    # note: --dataset is not checked against a list here; it must be exactly
    # cicids2017 or ciciot2023
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    # --alpha is needed only for local-only (which partition to use)
    p.add_argument("--alpha", type=float)
    p.add_argument("--clients", type=int, default=105)
    p.add_argument("--suffix", default="", help="partition suffix, e.g. _dev2")
    p.add_argument("--data-root", default="data/processed")
    p.add_argument("--results-root", default="results/runs")
    # training settings; batch size, learning rate, weight decay, patience,
    # min-delta and threads have the same values as in flower-ids/pyproject.toml
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    root, ds, seed = args.data_root, args.dataset, args.seed
    n_features, n_classes, classes = dataset_info(root, ds, seed)
    sdir = seed_dir(root, ds, seed)
    started = time.time()
    rows_out, history_rows = [], []

    # ---------------- centralized: one model on the whole training portion
    # results in results/runs/<dataset>/centralized/seed<S>/
    if args.method == "centralized":
        out = Path(args.results_root) / ds / "centralized" / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        write_run_config(out, vars(args), root, ds, seed)
        # alpha None = the global portions (all gateways together)
        x, y = load_xy(root, ds, seed, select_rows(root, ds, seed, None, TRAIN))
        xv, yv = load_xy(root, ds, seed, select_rows(root, ds, seed, None, VAL))
        model, init_sha = load_initial_model(sdir, n_features, n_classes)
        model, stopper, history = fit(model, x, y, xv, yv, args, n_classes, (0,), "macro_f1",
                                      verbose=True)
        history_rows = [(-1, *h) for h in history]
        # threat-landscape view: score the best model on the whole global test portion
        test_rows = select_rows(root, ds, seed, None, TEST)
        xt, yt = load_xy(root, ds, seed, test_rows)
        y_true, y_pred = yt.numpy(), predict(model, xt)
        cm = confusion(y_true, y_pred, n_classes)
        rows_out.append(confusion_rows("centralized", "threat_landscape", cm))
        gateway = {}
        # gateway-matched view: for every alpha with a partition, cut the same
        # predictions into each gateway's own test records (no retraining needed)
        for adir in sorted(sdir.glob("alpha*")):
            if "_" in adir.name:
                continue                                  # skip pilot partitions
            alpha = float(adir.name[5:])
            prow, pcli = portion_clients(root, ds, seed, alpha, TEST)
            if not np.array_equal(prow, test_rows):
                raise RuntimeError(f"{adir.name}: test rows differ from the global split")
            per = confusion_by_client(y_true, y_pred, pcli, int(pcli.max()) + 1, n_classes)
            rows_out += [confusion_rows("centralized", "gateway_matched", per[k], alpha, k)
                         for k in range(per.shape[0])]
            gateway[f"alpha{alpha:g}"] = [
                {"client": k, **{key: metrics(per[k])[key] for key in
                                 ("n", "accuracy", "macro_f1_present", "fpr")}}
                for k in range(per.shape[0])]
        # save predictions, the best model and the metrics
        save_parquet(pd.DataFrame({"row_id": test_rows, "y_true": y_true.astype(np.int8),
                                   "y_pred": y_pred.astype(np.int8)}),
                     out / "predictions.parquet")
        torch.save({"state_dict": model.state_dict(), "classes": classes,
                    "best_epoch": stopper.best_step, "initial_state_sha256": init_sha},
                   out / "checkpoint.pt")
        result = {"global_test": metrics(cm), "confusion": cm.tolist(),
                  "gateway_matched": gateway, "best_epoch": stopper.best_step,
                  "stopped_epoch": history[-1][0], "best_val_macro_f1": stopper.best_score}
    else:
        if args.alpha is None:
            raise SystemExit("--alpha is required for local-only")
        # ---------------- local-only: one separate model per gateway
        # results in results/runs/<dataset>/local/seed<S>/alpha<A>/
        out = (Path(args.results_root) / ds / "local" / f"seed{seed}"
               / f"alpha{args.alpha:g}{args.suffix}")
        out.mkdir(parents=True, exist_ok=True)
        write_run_config(out, vars(args), root, ds, seed, args.alpha, args.suffix)
        test_rows, test_client = portion_clients(root, ds, seed, args.alpha, TEST, args.suffix)
        xt, yt = load_xy(root, ds, seed, test_rows)
        y_true = yt.numpy()
        # routed view: each test record is predicted by the model of the gateway it
        # belongs to (-1 = not yet predicted)
        routed_pred = np.full(len(test_rows), -1, dtype=np.int64)
        per_client, states = [], {}
        for k in range(args.clients):
            # rows(portion) = gateway k's records of that portion
            rows = lambda s: select_rows(root, ds, seed, args.alpha, s, client=k,
                                         suffix=args.suffix)
            x, y = load_xy(root, ds, seed, rows(TRAIN))
            xv, yv = load_xy(root, ds, seed, rows(VAL))
            # every gateway starts from the same initial weights
            model, init_sha = load_initial_model(sdir, n_features, n_classes)
            model, stopper, history = fit(model, x, y, xv, yv, args, n_classes,
                                          (0, k + 1), "macro_f1_present")
            history_rows += [(k, *h) for h in history]
            pred = predict(model, xt)                         # global test, all rows
            cm_global = confusion(y_true, pred, n_classes)
            # the test records that belong to gateway k
            own = test_client == k
            routed_pred[own] = pred[own]
            cm_own = confusion(y_true[own], pred[own], n_classes)
            rows_out.append(confusion_rows(f"local_{k}", "threat_landscape", cm_global,
                                           args.alpha, k))
            rows_out.append(confusion_rows(f"local_{k}", "gateway_matched", cm_own,
                                           args.alpha, k))
            states[k] = model.state_dict()
            per_client.append({"client": k, "n_train": len(y), "best_epoch": stopper.best_step,
                               "val_macro_f1_present": stopper.best_score,
                               "threat_landscape": metrics(cm_global),
                               "gateway_matched": metrics(cm_own)})
            print(f"client {k}: n={len(y)} best epoch {stopper.best_step} gateway "
                  f"accuracy {per_client[-1]['gateway_matched']['accuracy']:.4f}", flush=True)
        # every test record must have been predicted by its own gateway's model
        if (routed_pred < 0).any():
            raise RuntimeError("Some test records were not routed to a gateway model")
        cm_routed = confusion(y_true, routed_pred, n_classes)
        rows_out.append(confusion_rows("local_routed", "routed", cm_routed, args.alpha))
        save_parquet(pd.DataFrame({"row_id": test_rows, "client": test_client.astype(np.int16),
                                   "y_true": y_true.astype(np.int8),
                                   "y_pred": routed_pred.astype(np.int8)}),
                     out / "predictions.parquet")
        torch.save({"state_dicts": states, "classes": classes,
                    "initial_state_sha256": init_sha}, out / "local_models.pt")
        result = {"routed": metrics(cm_routed), "routed_confusion": cm_routed.tolist(),
                  "threat_landscape_accuracy": [c["threat_landscape"]["accuracy"]
                                                for c in per_client],
                  "clients": per_client}

    # ---------------- files written by both methods
    save_parquet(pd.concat(rows_out, ignore_index=True), out / "confusion_counts.parquet")
    result["seconds"] = round(time.time() - started, 1)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    # trajectory.csv: validation score per epoch (client -1 = centralized model)
    pd.DataFrame(history_rows, columns=["client", "epoch", "val_selection_f1", "val_accuracy",
                                        "train_loss", "examples_processed",
                                        "optimizer_steps"]).to_csv(out / "trajectory.csv",
                                                                   index=False)
    print(f"BASELINE_OK {args.method} {ds} seed={seed}")


if __name__ == "__main__":
    main()
