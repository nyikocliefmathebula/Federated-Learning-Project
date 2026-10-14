# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""ClientApp: one simulated gateway trains on its saved training partition."""
# In the Flower simulation each of the 105 gateways is a "node". When the server
# selects a gateway for a round, Flower calls train() below on that node with the
# current global model. The gateway trains it on its own records and sends the
# new weights back. The data never leave the gateway; only weights are sent.
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fliids.data import (TRAIN, dataset_info, load_xy, select_rows,
                         training_class_counts)
from fliids.models import make_model
from fliids.seeds import torch_seed
from fliids.training import class_weights, make_optimizer, train_epochs

app = ClientApp()     # Flower finds this object through flower-ids/pyproject.toml

# Training data of every gateway this worker has served, kept for the rest of
# the run, so each gateway's rows are read and transformed once instead of each
# time it is selected. At most the whole training portion is held (about
# 0.4 GB for CIC-IDS2017 and 0.3 GB for CIC-IoT2023). The data are identical
# either way, so results do not change.
_DATA = {}


def gateway_data(root, dataset, seed, alpha, suffix, pid):
    """(n_k, x, y) for one gateway's training partition, cached per run."""
    run_key = (root, dataset, seed, alpha, suffix)
    if _DATA and next(iter(_DATA))[0] != run_key:
        _DATA.clear()                              # a different run: start afresh
    key = (run_key, pid)
    if key not in _DATA:
        # first time this gateway is selected in this run: find its training
        # rows in partition.parquet and load + transform them
        rows = select_rows(root, dataset, seed, alpha, TRAIN, client=pid, suffix=suffix)
        x, y = load_xy(root, dataset, seed, rows)
        _DATA[key] = (len(rows), x, y)             # n_k = number of training records
    return _DATA[key]


@app.query()
def query(message: Message, context: Context):
    """Report this node's partition-id so the server can map gateways."""
    # Flower gives every node an internal id; "partition-id" (0..104) says which
    # gateway's data it holds. The server asks once at the start of a run.
    pid = int(context.node_config["partition-id"])
    reply = RecordDict({"metrics": MetricRecord({"partition-id": pid})})
    return Message(content=reply, reply_to=message)


@app.train()
def train(message: Message, context: Context):
    # settings of this run: flower-ids/pyproject.toml, changed by --run-config
    cfg = context.run_config
    torch.set_num_threads(int(cfg["torch-threads"]))     # 4 threads (decision D11)
    root, dataset = cfg["data-root"], cfg["dataset"]
    seed, alpha, suffix = int(cfg["seed"]), float(cfg["alpha"]), cfg["partition-suffix"]
    pid = int(context.node_config["partition-id"])       # which gateway this is
    # sent by the server with the model: the round number and, for FedProx, mu
    server_round = int(message.content["config"]["server-round"])
    mu = float(message.content["config"].get("proximal-mu", 0.0))   # 0 for FedAvg

    n_k, x, y = gateway_data(root, dataset, seed, alpha, suffix, pid)
    n_features, n_classes, _ = dataset_info(root, dataset, seed)
    model = make_model(n_features, n_classes, 0)                 # the architecture
    model.load_state_dict(message.content["arrays"].to_torch_state_dict())  # global w_t
    optimizer = make_optimizer(model, float(cfg["learning-rate"]),
                               float(cfg["weight-decay"]))   # fresh Adam state
    # random streams for shuffling and dropout, fixed by (seed, round, gateway),
    # so FedAvg and FedProx see exactly the same randomness
    generator = torch.Generator().manual_seed(
        torch_seed(seed, "minibatch", server_round, pid))
    torch.manual_seed(torch_seed(seed, "dropout", server_round, pid))
    # class weights of the loss (decision D18). The grid uses power 0: plain
    # cross-entropy (weight None). A power above 0 (sensitivity analysis only)
    # gives every gateway the same weights, from the class counts of the whole
    # training portion.
    power = float(cfg["class-weight-power"])
    weight = (class_weights(training_class_counts(root, dataset, seed), power)
              if power > 0 else None)
    # local training: E = 2 epochs in minibatches of 256; mu > 0 adds the FedProx term
    stats = train_epochs(model, x, y, int(cfg["local-epochs"]), int(cfg["batch-size"]),
                         optimizer, generator, mu=mu, class_weight=weight)

    # reply to the server: the trained weights and some numbers about the training
    reply = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({
            "num-examples": n_k,                 # n_k: assigned training records
            "train-loss": float(stats["loss"]),
            "examples-processed": stats["examples"],
            "optimizer-steps": stats["steps"],
            "partition-id": pid,
        }),
    })
    return Message(content=reply, reply_to=message)
