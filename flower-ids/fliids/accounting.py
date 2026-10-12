# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Model-transfer accounting (report Sections 3.8 and 4.7)."""
# Federated learning sends models, not data. These functions measure the size
# of one model message; server_app.py calls them for every download (server to
# gateway) and upload (gateway to server) and writes communication.csv.
# "arrays" is a Flower ArrayRecord: a dictionary from layer name (for example
# "0.weight") to one array of numbers.
import numpy as np


def parameter_bytes(arrays) -> int:
    """Sum of tensor elements x byte width (the d*b of Eq. 3.14)."""
    # number of values in each array x bytes per value (4 for float32),
    # summed over all arrays: 19,719 x 4 = 78,876 bytes for the CIC-IDS2017 model
    return int(sum(int(np.prod(a.shape)) * np.dtype(a.dtype).itemsize
                   for a in arrays.values()))


def serialized_bytes(arrays) -> int:
    """Bytes of the serialized model arrays actually carried by a message."""
    # a.data is the array as stored in the message, including a small header
    # per array; in total 1,280 bytes more than parameter_bytes for this model
    return int(sum(len(a.data) for a in arrays.values()))


def tensor_types(arrays) -> str:
    # the number types used in the message, e.g. "float32"
    return ",".join(sorted({str(a.dtype) for a in arrays.values()}))
