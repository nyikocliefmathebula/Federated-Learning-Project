# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""The shared MLP and its matched initial state (report Section 4.4)."""
# The same network architecture is used by all four approaches (centralized,
# local-only, FedAvg, FedProx). Only the input size (number of features) and
# output size (number of classes) depend on the dataset.
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn


def make_model(n_features: int, n_classes: int, seed: int) -> nn.Sequential:
    """d_in -> 128 -> 64 -> C. LayerNorm, ReLU and dropout 0.2 after each
    hidden layer; He (Kaiming) weights, zero biases, LayerNorm scale 1 bias 0."""
    # fix PyTorch's random generator so the random initial weights are repeatable
    torch.manual_seed(seed)
    # nn.Sequential applies the layers one after another (numbered 0 to 8):
    #   0 Linear(d_in, 128)  weighted sums of the inputs (W1 x + b1)
    #   1 LayerNorm(128)     rescale the 128 values of each record (stable training)
    #   2 ReLU               activation max(0, z): makes the network non-linear
    #   3 Dropout(0.2)       during training, zero a random 20% of the values
    #   4 Linear(128, 64)    second hidden layer
    #   5 LayerNorm(64), 6 ReLU, 7 Dropout(0.2)
    #   8 Linear(64, C)      one output score ("logit") per class
    model = nn.Sequential(
        nn.Linear(n_features, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, n_classes),
    )
    # initialise the three linear layers: He (Kaiming) normal weights suit ReLU,
    # biases start at 0. LayerNorm keeps PyTorch's default (scale 1, shift 0).
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    return model


def state_sha256(state: dict) -> str:
    """Checksum of a state dict in a fixed key order (float32 bytes)."""
    # a "state dict" is the dictionary of all weights of a model; the checksum
    # proves that two models start from exactly the same weights
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def load_initial_model(seed_dir: Path, n_features: int, n_classes: int):
    """Build the MLP and copy in the saved initial state for this seed."""
    # initial_state.pt was saved once per seed by split_data.py; loading it here
    # gives every approach exactly the same starting weights (a fair comparison)
    state = torch.load(Path(seed_dir) / "initial_state.pt", weights_only=True)
    model = make_model(n_features, n_classes, 0)   # architecture; its own random
                                                   # weights are replaced below
    model.load_state_dict(state)
    # check that every weight tensor was copied exactly
    for key, value in model.state_dict().items():
        if not torch.equal(value, state[key]):
            raise RuntimeError(f"Initial state mismatch in {key}")
    return model, state_sha256(state)              # checksum is stored with each run


def parameter_count(model: nn.Module) -> int:
    # total number of trainable numbers: 19,719 (CIC-IDS2017, 82 inputs)
    # or 14,280 (CIC-IoT2023, 39 inputs)
    return sum(p.numel() for p in model.parameters())
