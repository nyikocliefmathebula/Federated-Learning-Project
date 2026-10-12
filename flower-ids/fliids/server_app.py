# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""ServerApp: FedAvg or FedProx with the saved schedule and early stopping.

The round loop is written out instead of calling strategy.start, because
strategy.start in Flower 1.39 has no early stopping or schedule control. It
still uses Flower's FedAvg / FedProx classes to build messages and aggregate
(weighted by num-examples = n_k).
"""
# One FedAvg or FedProx run, from start to finish:
#   1. record the settings (run_config.json) and map Flower nodes to gateways
#   2. load the matched initial model and the validation records
#   3. for each round: send the global model to the 21 scheduled gateways,
#      collect their trained models, average them (weighted by n_k), score the
#      new global model on the validation records, apply early stopping
#   4. keep the best round's model, test it once, save all results
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flwr.app import (ArrayRecord, ConfigRecord, Context, Message, MessageType,
                      RecordDict)
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from fliids.accounting import parameter_bytes, serialized_bytes, tensor_types
from fliids.data import (TEST, VAL, dataset_info, load_xy, portion_clients,
                         seed_dir, select_rows)
from fliids.evaluation import (confusion, confusion_by_client, confusion_rows,
                               metrics, predict, save_parquet)
from fliids.models import load_initial_model
from fliids.records import write_run_config
from fliids.seeds import make_schedule
from fliids.training import EarlyStopping

app = ServerApp()     # Flower finds this object through flower-ids/pyproject.toml


class Scheduled:
    """Mixin: train exactly the nodes in self.next_nodes this round."""

    next_nodes: list = []

    # Replaces Flower's own random choice of gateways: the message is built for
    # exactly the nodes listed in next_nodes (the round's 21 scheduled gateways)
    # and carries the round number and, for FedProx, the value of mu.
    def configure_train(self, server_round, arrays, config, grid):
        config["server-round"] = server_round
        if hasattr(self, "proximal_mu"):
            config["proximal-mu"] = self.proximal_mu
        record = RecordDict({self.arrayrecord_key: arrays,
                             self.configrecord_key: config})
        return self._construct_messages(record, self.next_nodes, MessageType.TRAIN)


# Flower's FedAvg and FedProx strategies with the fixed schedule added. Their
# aggregate_train computes the weighted average of the returned models.
class ScheduledFedAvg(Scheduled, FedAvg):
    pass


class ScheduledFedProx(Scheduled, FedProx):
    pass


def map_partitions(grid, n_clients, timeout):
    """Ask every node for its partition-id; return {partition_id: node_id}."""
    # wait until all simulated nodes have started (it waits for ever if fewer
    # nodes than num-clients were configured)
    while len(nodes := list(grid.get_node_ids())) < n_clients:
        time.sleep(1)
    # ask each node which gateway it is (handled by query() in client_app.py)
    queries = [Message(content=RecordDict({"config": ConfigRecord({})}),
                       message_type=MessageType.QUERY, dst_node_id=n) for n in nodes]
    mapping = {}
    for reply in grid.send_and_receive(queries, timeout=timeout):
        if reply.has_error():
            raise RuntimeError(f"Partition query failed: {reply.error}")
        mapping[int(reply.content["metrics"]["partition-id"])] = reply.metadata.src_node_id
    # every gateway 0..N-1 must appear exactly once
    if sorted(mapping) != list(range(n_clients)):
        raise RuntimeError("Partition ids do not cover 0..N-1 exactly once")
    return mapping


def write_csv(path, rows):
    # write a list of dictionaries as a CSV file (one column per key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@app.main()
def main(grid: Grid, context: Context):
    # ---- settings: flower-ids/pyproject.toml, changed by --run-config ----------
    cfg = dict(context.run_config)
    torch.set_num_threads(int(cfg["torch-threads"]))
    root, dataset, suffix = cfg["data-root"], cfg["dataset"], cfg["partition-suffix"]
    seed, alpha = int(cfg["seed"]), float(cfg["alpha"])
    method, n_clients = cfg["method"], int(cfg["num-clients"])
    per_round, rounds = int(cfg["clients-per-round"]), int(cfg["num-server-rounds"])
    timeout = float(cfg["round-timeout"])

    # ---- output folder, e.g. results/runs/cicids2017/fedavg/seed42/alpha1 -------
    out = (Path(cfg["results-root"]) / dataset / method / f"seed{seed}"
           / f"alpha{alpha:g}{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    write_run_config(out, cfg, root, dataset, seed, alpha, suffix)

    # ---- which Flower node is which gateway, and the round schedule ------------
    node_of = map_partitions(grid, n_clients, timeout)
    (out / "node_map.json").write_text(json.dumps(node_of, indent=2))
    # the schedule is recreated from the seed and checked against the
    # schedule.json saved by split_data.py (when the sizes are the same)
    schedule = make_schedule(seed, n_clients, per_round, rounds)
    saved = seed_dir(root, dataset, seed) / "schedule.json"
    if saved.exists():
        s = json.loads(saved.read_text())
        if (s["clients"], s["clients_per_round"]) == (n_clients, per_round):
            if s["schedule"][:rounds] != schedule:
                raise RuntimeError("Schedule differs from the saved schedule.json")
    (out / "schedule.json").write_text(json.dumps(schedule))

    # ---- the strategy: Flower's FedAvg or FedProx with the fixed schedule ------
    # weighted_by_key="num-examples": each gateway's model counts in proportion
    # to its number of training records n_k; no federated evaluation is used
    common = dict(fraction_train=per_round / n_clients, fraction_evaluate=0.0,
                  min_available_nodes=n_clients, weighted_by_key="num-examples")
    if method == "fedprox":
        strategy = ScheduledFedProx(proximal_mu=float(cfg["proximal-mu"]), **common)
    elif method == "fedavg":
        strategy = ScheduledFedAvg(**common)
    else:
        raise ValueError(f"Unknown method {method!r}")

    # ---- the initial global model and the validation records -------------------
    n_features, n_classes, classes = dataset_info(root, dataset, seed)
    model, init_sha = load_initial_model(seed_dir(root, dataset, seed), n_features, n_classes)
    arrays = ArrayRecord(model.state_dict())      # the global model as a message
    # all gateways' validation records together (= the whole validation portion)
    x_val, y_val = load_xy(root, dataset, seed, select_rows(root, dataset, seed, alpha,
                                                            VAL, suffix=suffix))

    def validate(arr):
        # put the weights into the model and score it on the validation records
        model.load_state_dict(arr.to_torch_state_dict())
        return metrics(confusion(y_val.numpy(), predict(model, x_val), n_classes))

    stopper = EarlyStopping(int(cfg["patience"]), float(cfg["min-delta"]))
    first = validate(arrays)                      # round 0 = the untrained model
    stopper.update(0, first["macro_f1"], model.state_dict())
    trajectory = [{"round": 0, "val_macro_f1": first["macro_f1"],
                   "val_accuracy": first["accuracy"], "train_loss": "",
                   "examples_processed": 0, "optimizer_steps": 0,
                   "best_round": 0, "misses": 0, "seconds": 0.0}]
    ledger, cum_param, cum_ser = [], 0, 0         # communication ledger and totals
    started, stopped_at = time.time(), 0

    # ---- the federated rounds ---------------------------------------------------
    for r in range(1, rounds + 1):
        # 1. send the global model to this round's scheduled gateways and wait
        strategy.next_nodes = [node_of[pid] for pid in schedule[r - 1]]
        messages = list(strategy.configure_train(r, arrays, ConfigRecord(), grid))
        p_down, s_down = parameter_bytes(arrays), serialized_bytes(arrays)
        replies = list(grid.send_and_receive(messages, timeout=timeout))
        # 2. every scheduled gateway must reply without error, else stop the run
        failed = [m for m in replies if m.has_error()]
        returned = sorted(int(m.content["metrics"]["partition-id"])
                          for m in replies if not m.has_error())
        if failed or returned != schedule[r - 1]:
            (out / "failure.json").write_text(json.dumps(
                {"round": r, "failed": len(failed), "returned": returned,
                 "expected": schedule[r - 1],
                 "errors": [str(m.error) for m in failed]}))
            raise RuntimeError(f"Round {r}: missing or failed client replies")
        # Replies arrive in an arbitrary order; floating-point sums depend on order,
        # so aggregate in partition-id order to make every run repeatable.
        replies.sort(key=lambda m: int(m.content["metrics"]["partition-id"]))
        # 3. communication ledger: one download row per scheduled gateway ...
        for pid in schedule[r - 1]:
            cum_param += p_down
            cum_ser += s_down
            ledger.append({"round": r, "client": pid, "direction": "download",
                           "tensor_types": tensor_types(arrays),
                           "parameter_bytes": p_down, "serialized_bytes": s_down,
                           "cumulative_parameter_bytes": cum_param,
                           "cumulative_serialized_bytes": cum_ser})
        # ... and one upload row per reply; also add up the training work
        examples = steps = 0
        for m in replies:
            a = m.content["arrays"]
            cum_param += parameter_bytes(a)
            cum_ser += serialized_bytes(a)
            examples += int(m.content["metrics"]["examples-processed"])
            steps += int(m.content["metrics"]["optimizer-steps"])
            ledger.append({"round": r, "client": int(m.content["metrics"]["partition-id"]),
                           "direction": "upload", "tensor_types": tensor_types(a),
                           "parameter_bytes": parameter_bytes(a),
                           "serialized_bytes": serialized_bytes(a),
                           "cumulative_parameter_bytes": cum_param,
                           "cumulative_serialized_bytes": cum_ser})
        # 4. new global model = average of the returned models weighted by n_k
        arrays, train_metrics = strategy.aggregate_train(r, replies)
        # 5. score it on the validation records and apply early stopping
        val = validate(arrays)
        stop = stopper.update(r, val["macro_f1"], model.state_dict())
        trajectory.append({"round": r, "val_macro_f1": val["macro_f1"],
                           "val_accuracy": val["accuracy"],
                           "train_loss": float(train_metrics["train-loss"]),
                           "examples_processed": examples, "optimizer_steps": steps,
                           "best_round": stopper.best_step, "misses": stopper.misses,
                           "seconds": round(time.time() - started, 2)})
        print(f"round {r}: val macro-F1 {val['macro_f1']:.4f} (best "
              f"{stopper.best_score:.4f} @ {stopper.best_step}), misses {stopper.misses}",
              flush=True)
        stopped_at = r
        if stop:
            break

    # ---- after training: save the curves and the best model ---------------------
    write_csv(out / "trajectory.csv", trajectory)
    write_csv(out / "communication.csv", ledger)
    model.load_state_dict(stopper.best_state)     # back to the best round's weights
    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "best_round": stopper.best_step, "val_macro_f1": stopper.best_score,
                "initial_state_sha256": init_sha}, out / "checkpoint.pt")

    # ---- test the best model once ------------------------------------------------
    # all test records with their gateway: the whole set gives the threat-landscape
    # view, each gateway's own records give the gateway-matched view
    test_rows, test_client = portion_clients(root, dataset, seed, alpha, TEST, suffix)
    x_test, y_test = load_xy(root, dataset, seed, test_rows)
    y_true, y_pred = y_test.numpy(), predict(model, x_test)
    cm = confusion(y_true, y_pred, n_classes)
    per_client = confusion_by_client(y_true, y_pred, test_client, n_clients, n_classes)
    save_parquet(pd.DataFrame({"row_id": test_rows, "client": test_client.astype(np.int16),
                               "y_true": y_true.astype(np.int8),
                               "y_pred": y_pred.astype(np.int8)}),
                 out / "predictions.parquet")
    rows = [confusion_rows(method, "threat_landscape", cm, alpha)]
    rows += [confusion_rows(method, "gateway_matched", per_client[k], alpha, k)
             for k in range(n_clients)]
    save_parquet(pd.concat(rows, ignore_index=True), out / "confusion_counts.parquet")

    # ---- metrics.json: test results, stopping point and communication -----------
    best = stopper.best_step
    at_best = [l for l in ledger if l["round"] <= best]   # messages up to the best round
    result = {
        "global_test": metrics(cm), "confusion": cm.tolist(),
        "gateway_matched": [{"client": k, **{key: metrics(per_client[k])[key] for key in
                             ("n", "accuracy", "macro_f1_present", "fpr")}}
                            for k in range(n_clients)],
        "best_round": best, "stopped_round": stopped_at,
        "best_val_macro_f1": stopper.best_score,
        "parameter_bytes_at_best": sum(l["parameter_bytes"] for l in at_best),
        "serialized_bytes_at_best": sum(l["serialized_bytes"] for l in at_best),
        "parameter_bytes_at_stop": cum_param, "serialized_bytes_at_stop": cum_ser,
        "examples_processed": int(sum(t["examples_processed"] for t in trajectory)),
        "optimizer_steps": int(sum(t["optimizer_steps"] for t in trajectory)),
        "seconds": round(time.time() - started, 1),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    # final marker line: the run's log shows that it finished normally
    print(f"FL_RUN_OK {method} seed={seed} alpha={alpha:g} "
          f"test accuracy={result['global_test']['accuracy']:.4f}", flush=True)
