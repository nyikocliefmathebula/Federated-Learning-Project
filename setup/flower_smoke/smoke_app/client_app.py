# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

# One of the two simulated clients of the installation check. Each round it
# receives the model, makes one Adam step on its own random data and sends the
# new weights back. This only proves that Flower, Ray and PyTorch work together;
# it uses no IDS data.
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from smoke_app.task import make_data, make_model

app = ClientApp()


@app.train()
def train(message: Message, context: Context):
    torch.set_num_threads(1)
    # which client this is: 0 or 1
    partition_id = int(context.node_config["partition-id"])
    if partition_id not in (0, 1):
        raise ValueError("This smoke test requires exactly two clients.")
    # client 0 has 32 records, client 1 has 48 (so the server can check both replied)
    count = 32 + 16 * partition_id
    features, labels = make_data(partition_id, count)
    model = make_model()
    model.load_state_dict(message.content["arrays"].to_torch_state_dict())
    model.train()

    # one training step: loss -> gradients -> weight update
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(features), labels)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite synthetic training loss.")
    loss.backward()
    optimizer.step()

    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("Non-finite updated model parameters.")

    # send back the new weights, the record count and the loss
    reply = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({
            "num-examples": count,
            "loss": float(loss.item()),
        }),
    })
    return Message(content=reply, reply_to=message)
