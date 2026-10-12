# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Stable random-stream derivation (report Section 4.6).

Every stochastic operation draws from its own stream, derived from the master
seed, a fixed stream code and only the identifiers that are relevant to it.
Python's process-dependent hash() is never used, so streams are identical in
every process and on every run.

    split       master seed (+ attempt)      -> never depends on alpha
    subsample   master seed                  -> never depends on alpha
    init        master seed                  -> never depends on alpha
    schedule    master seed                  -> same for FedAvg, FedProx, all alpha
    partition   master seed, alpha (+ portion, attempt)
    minibatch   master seed, round, client   -> same for FedAvg and FedProx
    dropout     master seed, round, client   -> same for FedAvg and FedProx
"""
# Why separate streams: if, for example, the partition used the same random
# numbers as the split, changing alpha would also change the split. With one
# stream per purpose, each choice depends only on what it should depend on.
import numpy as np

# fixed code number for each purpose; part of every stream's seed
STREAMS = {
    "split": 101,
    "subsample": 102,
    "partition": 103,
    "init": 104,
    "schedule": 105,
    "minibatch": 106,
    "dropout": 107,
}


def alpha_id(alpha: float) -> int:
    """Integer identifier for a Dirichlet alpha (0.1 -> 100, 100 -> 100000)."""
    # random streams need whole numbers, so alpha is multiplied by 1000
    return int(round(float(alpha) * 1000))


def seed_sequence(master: int, stream: str, *ids: int) -> np.random.SeedSequence:
    if stream not in STREAMS:
        raise KeyError(f"Unknown random stream {stream!r}")
    # NumPy mixes the list [master seed, stream code, ids...] into a seed;
    # the same list always gives the same random numbers
    return np.random.SeedSequence([int(master), STREAMS[stream], *map(int, ids)])


def rng(master: int, stream: str, *ids: int) -> np.random.Generator:
    """NumPy generator for one stream."""
    return np.random.Generator(np.random.PCG64(seed_sequence(master, stream, *ids)))


def torch_seed(master: int, stream: str, *ids: int) -> int:
    """63-bit integer seed for torch.manual_seed / torch.Generator."""
    # PyTorch needs one integer seed; take it from the same stream construction
    state = seed_sequence(master, stream, *ids).generate_state(1, dtype=np.uint64)
    return int(state[0] % np.uint64(2**63 - 1))


def make_schedule(master: int, clients: int, per_round: int, rounds: int):
    """Clients (partition ids) trained in each round, without replacement."""
    if per_round > clients:
        raise ValueError("clients-per-round exceeds the number of clients")
    g = rng(master, "schedule", clients, per_round)
    # for each round: choose per_round different gateways (e.g. 21 of 105),
    # sorted; the list depends only on the seed, not on the dataset or alpha
    return [sorted(g.choice(clients, per_round, replace=False).tolist())
            for _ in range(rounds)]
