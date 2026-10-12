# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

# Model and made-up data shared by the smoke-test client and server.
import torch


# a single linear layer: 6 inputs -> 2 classes; seed 42 so it always starts the same
def make_model():
    torch.manual_seed(42)
    return torch.nn.Linear(6, 2)


# random inputs; the label is 1 when x0 + 0.25*x1 > 0, else 0 (an easy rule a
# linear model can learn); each client gets its own random seed
def make_data(partition_id, count):
    generator = torch.Generator().manual_seed(1000 + partition_id)
    features = torch.randn(count, 6, generator=generator)
    labels = (features[:, 0] + 0.25 * features[:, 1] > 0).long()
    return features, labels


# server-side check after each round: loss and accuracy on 64 other random
# records (seed 99)
def validate(server_round, arrays):
    from flwr.app import MetricRecord

    model = make_model()
    model.load_state_dict(arrays.to_torch_state_dict())
    model.eval()
    features, labels = make_data(99, 64)
    with torch.no_grad():
        outputs = model(features)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        accuracy = (outputs.argmax(1) == labels).float().mean()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite synthetic validation loss.")
    return MetricRecord({
        "loss": float(loss.item()),
        "accuracy": float(accuracy.item()),
    })
