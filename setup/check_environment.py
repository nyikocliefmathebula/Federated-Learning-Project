# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Record the actual environment and test one CPU training step."""
# Run once after installing the packages (from the project root, .venv active):
#     python setup/check_environment.py
# It writes docs/hardware_environment.json (used for the report's hardware and
# software table) and ends with ENVIRONMENT_CHECK_OK if everything works.
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import flwr
import psutil
import ray
import torch


def main():
    # same thread setting as the experiments (decision D11)
    torch.set_num_threads(4)
    # installed version of each package, read from the package records
    names = [
        "flwr", "ray", "torch", "numpy", "pandas", "scipy", "scikit-learn",
        "pyarrow", "matplotlib", "seaborn", "pyyaml", "pytest", "tqdm", "psutil",
    ]
    versions = {name: importlib.metadata.version(name) for name in names}
    here = Path(__file__).resolve().parent
    # Project root = the first folder at or above this script that holds .venv,
    # so the record goes to <project root>/docs wherever the script lives.
    root = next((p for p in (here, *here.parents) if (p / ".venv").is_dir()), here)
    # what this computer offers, as seen from inside WSL
    record = {
        "operating_system": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "architecture": platform.machine(),
        "logical_cpus_visible": psutil.cpu_count(),
        "ram_visible_gib": round(psutil.virtual_memory().total / 1024**3, 2),
        "free_project_drive_gib": round(psutil.disk_usage(root).free / 1024**3, 2),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "package_versions": versions,
    }
    # stop if the Python or Flower version differs from the one this project uses
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use the Python 3.12 WSL environment described in README.md.")
    if flwr.__version__ != "1.39.0":
        raise RuntimeError("The Flower commands of this project target version 1.39.0.")
    # one tiny training step on random numbers (6 inputs, 2 classes): checks that
    # PyTorch can compute a loss, gradients and an Adam update on this CPU
    torch.manual_seed(42)
    model = torch.nn.Linear(6, 2)
    features = torch.randn(32, 6)
    labels = torch.randint(0, 2, (32,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(features), labels)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite setup loss.")
    loss.backward()
    optimizer.step()
    record["synthetic_training_step"] = "passed"
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "hardware_environment.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    print("ENVIRONMENT_CHECK_OK")


if __name__ == "__main__":
    main()
