# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

# OLDER VERSION, kept only for the record. setup/check_environment.py replaced
# it. Note: running this file would overwrite docs/hardware_environment.json
# with its shorter record.
# It records the hardware and package versions, then performs one tiny training
# step (10 inputs, 8 classes) on random numbers.
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow
import sklearn
import torch

torch.set_num_threads(4)

record = {
    "operating_system": platform.platform(),
    "architecture": platform.machine(),
    "processor": platform.processor(),
    "physical_cores": psutil.cpu_count(logical=False),
    "logical_cores": psutil.cpu_count(logical=True),
    "usable_ram_gib": round(
        psutil.virtual_memory().total / 1024**3, 2
    ),
    "free_drive_gib": round(
        psutil.disk_usage(".").free / 1024**3, 2
    ),
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "pyarrow": pyarrow.__version__,
    "pytorch": torch.__version__,
    "torch_threads": torch.get_num_threads(),
    "cuda_available": torch.cuda.is_available(),
}

Path("docs").mkdir(exist_ok=True)
Path("docs/hardware_environment.json").write_text(
    json.dumps(record, indent=2), encoding="utf-8"
)
print(json.dumps(record, indent=2))

torch.manual_seed(42)
model = torch.nn.Linear(10, 8)
features = torch.randn(32, 10)
labels = torch.randint(0, 8, (32,))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()
loss = torch.nn.functional.cross_entropy(model(features), labels)
if not torch.isfinite(loss).item():
    raise RuntimeError("Non-finite loss.")
loss.backward()
optimizer.step()
print("A training step completed successfully.")