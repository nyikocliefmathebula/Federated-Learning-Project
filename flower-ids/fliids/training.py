# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""One training routine for every approach (report Section 4.5.1)."""
# Centralized, local-only, FedAvg and FedProx all update their models with
# train_epochs below and decide when to stop with EarlyStopping, so differences
# in results come from the approach, not from different training code.
import torch
from torch import nn


def make_optimizer(model, lr=1e-3, weight_decay=1e-5):
    """Adam with coupled weight decay, betas (0.9, 0.999), eps 1e-8."""
    # Adam adapts the step size of each weight. lr = learning rate (0.001);
    # weight_decay (0.00001) gently pulls weights towards zero.
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=weight_decay)


def proximal_term(model, reference):
    # FedProx: squared distance ||w - w_t||^2 between the current weights w and
    # the weights w_t received from the server at the start of the round
    return sum((p - r).square().sum() for p, r in zip(model.parameters(), reference))


def train_epochs(model, x, y, epochs, batch_size, optimizer, generator, mu=0.0):
    """Minibatch training; the FedProx term is added when mu > 0.

    The reference w_t is copied once, before the first update, and never
    changes during local training. The final incomplete minibatch is kept.
    Returns mean loss, examples processed and optimizer steps.
    """
    # x: inputs [n, d_in]; y: class codes [n]; epochs: passes over the data;
    # batch_size: 256; generator: seeded random stream for the shuffling
    reference = [p.detach().clone() for p in model.parameters()]   # w_t (FedProx)
    loss_fn = nn.CrossEntropyLoss()   # softmax + minus log-probability of the true class
    model.train()                     # training mode: dropout is switched on
    total, seen, steps = 0.0, 0, 0
    for _ in range(epochs):
        order = torch.randperm(len(y), generator=generator)   # new random order
        for i in range(0, len(y), batch_size):
            idx = order[i:i + batch_size]                     # one minibatch
            optimizer.zero_grad()                             # forget old gradients
            loss = loss_fn(model(x[idx]), y[idx])             # forward pass and loss
            if mu > 0:
                # FedProx only: penalise moving away from the global model
                loss = loss + 0.5 * mu * proximal_term(model, reference)
            if not torch.isfinite(loss):                      # stop on NaN or infinity
                raise FloatingPointError("Non-finite training loss")
            loss.backward()                                   # compute the gradients
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError("Non-finite gradient")
            optimizer.step()                                  # update the weights
            total += loss.item() * len(idx)                   # for the mean loss
            seen += len(idx)                                  # examples processed
            steps += 1                                        # optimizer steps
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite parameters after training")
    return {"loss": total / max(seen, 1), "examples": seen, "steps": steps}


class EarlyStopping:
    """Best checkpoint = strictly higher score (earliest kept on ties);
    patience resets only after an improvement of at least min_delta over the
    score at the last reset (report Section 4.5.1)."""

    def __init__(self, patience=15, min_delta=1e-4):
        # patience: how many checks without real improvement before stopping (15)
        # min_delta: the smallest improvement that counts as real (0.0001)
        self.patience, self.min_delta = patience, min_delta
        self.best_score, self.best_step, self.best_state = None, None, None
        self.reset_score, self.misses = None, 0

    def update(self, step, score, state):
        # step: epoch or round number; score: validation macro-F1;
        # state: the model's current weights
        if self.best_score is None or score > self.best_score:
            # a new best score: remember it and keep a copy of the weights
            self.best_score, self.best_step = score, step
            self.best_state = {k: v.detach().clone() for k, v in state.items()}
        if self.reset_score is None or score >= self.reset_score + self.min_delta:
            # a real improvement: restart the patience count
            self.reset_score, self.misses = score, 0
        else:
            self.misses += 1
        return self.misses >= self.patience      # True means: stop training now
