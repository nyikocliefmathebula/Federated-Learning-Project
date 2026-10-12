# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Run records: configuration hash, inputs, software and source revision."""
# write_run_config creates run_config.json at the start of every run. It records
# everything needed to show later exactly what produced a result.
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path


def config_hash(settings: dict) -> str:
    # short fingerprint of all settings: two runs with the same hash used the
    # same settings
    text = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def source_revision(start: Path) -> str:
    # the Git commit of the code; "+uncommitted" is added when tracked files were
    # changed or new non-ignored files exist since that commit
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=start, text=True,
                             capture_output=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=start, text=True,
                               capture_output=True, timeout=10).stdout.strip()
        if out.returncode == 0:
            return out.stdout.strip() + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "not a git repository"


def software():
    # versions of Python and the main packages, and the operating system
    import numpy, torch
    info = {"python": platform.python_version(), "numpy": numpy.__version__,
            "torch": torch.__version__, "platform": platform.platform()}
    try:
        import flwr
        info["flwr"] = flwr.__version__
    except ImportError:
        pass
    return info


def input_checksums(root, dataset, seed, alpha=None, suffix=""):
    """Checksums recorded by the data pipeline for the files a run reads."""
    # the checksums are copied from the manifests written by build_arrays.py,
    # split_data.py and partition_clients.py (they are not recomputed here)
    base = Path(root) / dataset
    arrays = json.loads((base / "arrays" / "arrays_manifest.json").read_text())
    split = json.loads((base / f"seed{seed}" / "split_summary.json").read_text())
    record = {"arrays": arrays["sha256"], "split": split["outputs"],
              "initial_state_sha256": split["initial_state_sha256"]}
    if alpha is not None:
        # runs that use gateways also record the partition's checksums
        part = json.loads((base / f"seed{seed}" / f"alpha{float(alpha):g}{suffix}"
                           / "partition_summary.json").read_text())
        record["partition"] = part["outputs"]
    return record


def write_run_config(out_dir: Path, settings: dict, root, dataset, seed,
                     alpha=None, suffix=""):
    # settings: all arguments of the run; written to <run folder>/run_config.json
    record = {"settings": settings, "config_hash": config_hash(settings),
              "inputs": input_checksums(root, dataset, seed, alpha, suffix),
              "software": software(),
              "source_revision": source_revision(Path(root).resolve()),
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "run_config.json").write_text(json.dumps(record, indent=2, default=str))
    return record
