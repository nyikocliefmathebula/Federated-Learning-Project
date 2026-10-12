# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

# Server of the installation check: two rounds of FedAvg with both clients,
# with checks that each round really used both clients. Prints FLOWER_SMOKE_OK
# at the end if everything worked.
import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from smoke_app.task import make_model, validate

app = ServerApp()


class CheckedFedAvg(FedAvg):
    """Require successful replies from both installation-check clients."""

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        if len(replies) != 2 or any(reply.has_error() for reply in replies):
            raise RuntimeError("Both clients must return a successful update.")
        # the two clients must report 32 and 48 records (see client_app.py)
        counts = sorted(reply.content["metrics"]["num-examples"] for reply in replies)
        if counts != [32, 48]:
            raise RuntimeError("Unexpected client sample counts.")
        # then average the two models as normal FedAvg does (weighted by records)
        return super().aggregate_train(server_round, replies)


@app.main()
def main(grid: Grid, context: Context):
    torch.set_num_threads(1)
    rounds = int(context.run_config["num-server-rounds"])
    if rounds != 2:
        raise ValueError("Keep this installation check at two rounds.")
    # every round: train on all clients (fraction 1.0), no client-side evaluation
    strategy = CheckedFedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=2,
        min_available_nodes=2,
        weighted_by_key="num-examples",
    )
    # run the rounds; after each round the server scores the model with validate()
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(make_model().state_dict()),
        num_rounds=rounds,
        evaluate_fn=validate,
    )
    # both rounds must have training results, and the final weights must be finite
    if set(result.train_metrics_clientapp) != {1, 2}:
        raise RuntimeError("Both training rounds must finish.")
    state = result.arrays.to_torch_state_dict()
    if not state or not all(torch.isfinite(v).all().item() for v in state.values()):
        raise RuntimeError("Invalid final model.")
    print("FLOWER_SMOKE_OK: two synthetic training rounds completed.", flush=True)
