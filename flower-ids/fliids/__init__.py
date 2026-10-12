# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Federated IoT intrusion detection research app."""
# This folder is the Python package "fliids" (Federated Learning IoT Intrusion
# Detection System). It holds the code shared by ALL four training approaches:
#
#   data.py        read the prepared arrays, splits and partitions; transform inputs
#   models.py      the neural network (MLP) and the matched initial weights
#   training.py    one training routine (Adam, minibatches, FedProx term) and
#                  the early-stopping rule
#   evaluation.py  predictions, confusion matrices and metrics
#   accounting.py  size in bytes of the model messages (communication cost)
#   seeds.py       separate, repeatable random streams derived from a seed
#   records.py     run_config.json: settings, input checksums, software, Git version
#   client_app.py  Flower ClientApp: one simulated gateway training locally
#   server_app.py  Flower ServerApp: FedAvg / FedProx rounds, validation, stopping
#
# src/train_baseline.py (centralized and local-only) imports the same modules,
# so every approach uses identical model, training and metric code.
