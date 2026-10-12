# Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town 
# 2026

"""Running the planned experiment grid one configuration at a time (report Section 4.10).

Run from the project root with .venv active. Federated methods also need a
Flower SuperLink running in a separate terminal (see README.md):

    python src/run_grid.py --dataset cicids2017 --methods centralized local --dry-run
    python src/run_grid.py --dataset cicids2017 --methods centralized local --keep-going
    python src/run_grid.py --dataset cicids2017 --methods fedavg fedprox --keep-going

Defaults: seeds 13 42 271 1618 3141 and alpha 100 10 1 0.5 0.1, i.e. per
dataset 5 centralized + 25 local-only + 25 FedAvg + 25 FedProx = 80 runs.

Before running, the script checks that data_check.json passed and that every
split and partition exists; for FedAvg/FedProx it checks that the SuperLink is
reachable and sets the simulation to 105 nodes, one client at a time.

A run is complete when its metrics.json exists; completed runs are skipped, so
the command can simply be repeated after an interruption. The folder of an
unfinished run is renamed to <name>.incomplete-<time> before it is repeated.
Every attempt is logged in results/runs/logs/<tag>.log and results/runs/grid_log.csv.

Ctrl+C stops the grid. A federated run would otherwise keep running inside the
SuperLink after its log stream ends, so the runner also stops it (flwr stop).
"""
# This script does not train anything itself. For each planned run it starts
# another program (train_baseline.py or flwr run), waits for it to finish,
# writes everything that program prints to a log file, and records the result.
import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

# folders relative to the project root
RESULTS = Path("results/runs")
PROCESSED = Path("data/processed")
FEDERATED = ("fedavg", "fedprox")
# settings sent to the SuperLink before federated runs: 105 simulated gateways;
# Ray may use 4 CPUs in total (init-args) and each client reserves 4 CPUs
# (client-resources), so exactly one client trains at a time
SIMULATION = ["flwr", "federation", "simulation-config", "--num-supernodes", "105",
              "--client-resources-num-cpus", "4", "--client-resources-num-gpus", "0",
              "--init-args-num-cpus", "4", "--init-args-num-gpus", "0"]
CHILD_ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}   # logs are written as the run goes


def plan_runs(methods, seeds, alphas):
    """(method, seed, alpha) in execution order: seed by seed, method by method."""
    # example for one seed: centralized, then local x 5 alphas, then fedavg x 5,
    # then fedprox x 5 (only the methods given with --methods)
    plan = []
    for seed in seeds:
        for method in methods:
            for alpha in ([None] if method == "centralized" else alphas):
                plan.append((method, seed, alpha))
    return plan


# results folder of one run, e.g. results/runs/cicids2017/fedavg/seed42/alpha1
def run_dir(dataset, method, seed, alpha):
    out = RESULTS / dataset / method / f"seed{seed}"
    return out if alpha is None else out / f"alpha{alpha:g}"


# short name of one run, e.g. cicids2017_fedavg_seed42_alpha1 (used for the log)
def run_tag(dataset, method, seed, alpha):
    return f"{dataset}_{method}_seed{seed}" + ("" if alpha is None else f"_alpha{alpha:g}")


# the command that performs one run: train_baseline.py for the baselines, or
# flwr run for FedAvg/FedProx (settings not given here come from
# flower-ids/pyproject.toml)
def command(dataset, method, seed, alpha):
    if method in ("centralized", "local"):
        cmd = [sys.executable, "src/train_baseline.py", "--method", method,
               "--dataset", dataset, "--seed", str(seed)]
        return cmd + ([] if alpha is None else ["--alpha", f"{alpha:g}"])
    overrides = f"dataset='{dataset}' method='{method}' seed={seed} alpha={float(alpha)}"
    return ["flwr", "run", "flower-ids", "--stream", "--run-config", overrides]


def shown(cmd):
    """The command as typed in a terminal (python instead of the full path)."""
    words = ["python" if w == sys.executable else w for w in cmd]
    return " ".join(f'"{w}"' if " " in w else w for w in words)


def preflight(dataset, plan):
    """Problems that would make runs fail; empty list if ready."""
    problems = []
    # check_data.py must have run and passed
    check = PROCESSED / dataset / "data_check.json"
    if not check.exists():
        problems.append(f"{check} not found: run src/check_data.py --dataset {dataset}")
    elif json.loads(check.read_text()).get("failed"):
        problems.append(f"{check} lists failed checks: fix them first")
    # the split, initial weights and (except centralized) the partition of every
    # planned run must exist
    for method, seed, alpha in plan:
        sdir = PROCESSED / dataset / f"seed{seed}"
        need = [sdir / "split_indices.parquet", sdir / "initial_state.pt"]
        if alpha is not None:
            need.append(sdir / f"alpha{alpha:g}" / "partition.parquet")
        problems += [f"missing {p}" for p in need if not p.exists()]
    return sorted(set(problems))


# True if something accepts connections on the SuperLink port
def superlink_reachable(host="127.0.0.1", port=39091, timeout=3.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def active_flower_runs():
    """Ids of runs the SuperLink lists as pending, starting or running."""
    try:
        # ask the SuperLink for its list of runs; skip any text before the JSON
        out = subprocess.run(["flwr", "ls", "--format", "json"], capture_output=True,
                             text=True, timeout=120).stdout
        runs = json.loads(out[out.index("{"):]).get("runs", [])
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    return [str(r["run-id"]) for r in runs
            if r.get("status") in ("pending", "starting", "running")]


def started_run_id(log_path):
    """Run id printed by flwr run ('Successfully started run <id>'), or None."""
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return None
    # look for the line flwr run prints when the run has been accepted
    found = re.search(r"Successfully started run (\d+)", text)
    return found.group(1) if found else None


# add one line to results/runs/grid_log.csv (with a header line the first time)
def log_attempt(row):
    path = RESULTS / "grid_log.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def main():
    if not (PROCESSED.is_dir() and Path("src").is_dir()):
        raise SystemExit("Run this from the project root:\n  cd /mnt/c/Users/Nyiko/fl-iot-ids")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["cicids2017", "ciciot2023"])
    p.add_argument("--methods", nargs="+", required=True,
                   choices=["centralized", "local", "fedavg", "fedprox"])
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 271, 1618, 3141])
    p.add_argument("--alphas", type=float, nargs="+", default=[100, 10, 1, 0.5, 0.1])
    # --dry-run: only print the runs and their commands
    p.add_argument("--dry-run", action="store_true", help="list the plan, run nothing")
    # --keep-going: a failed run is recorded and the grid continues; without it
    # the grid stops at the first failure
    p.add_argument("--keep-going", action="store_true",
                   help="continue with the next run after a failure")
    args = p.parse_args()

    # ---- 1. plan: all runs, minus the ones that already have metrics.json
    plan = plan_runs(args.methods, args.seeds, args.alphas)
    todo = [r for r in plan if not (run_dir(args.dataset, *r) / "metrics.json").exists()]
    print(f"{args.dataset}: {len(plan)} planned runs, {len(plan) - len(todo)} already "
          f"complete, {len(todo)} to run", flush=True)
    # ---- 2. checks that need no SuperLink (data ready?)
    problems = preflight(args.dataset, plan)
    if problems:
        print("NOT READY:\n  " + "\n  ".join(problems[:20]))
        sys.exit(1)
    if args.dry_run:
        for r in todo:
            print("plan  " + run_tag(args.dataset, *r))
            print("      " + shown(command(args.dataset, *r)))
        print("DRY_RUN_OK")
        return
    # ---- 3. checks for federated runs: SuperLink up, no old run still active,
    # simulation set to 105 gateways
    if any(m in FEDERATED for m, _, _ in todo):
        if not superlink_reachable():
            print("NOT READY: no SuperLink on 127.0.0.1:39091. Start it in a separate "
                  "terminal (see README.md), then repeat this command.")
            sys.exit(1)
        active = active_flower_runs()
        if active:
            print("NOT READY: the SuperLink is still running Flower run(s) " + ", ".join(active)
                  + ". Stop each one with  flwr stop <run id>  and repeat this command.")
            sys.exit(1)
        if subprocess.call(SIMULATION) != 0:
            print("NOT READY: could not set the simulation to 105 nodes.")
            sys.exit(1)

    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    failures = []
    # ---- 4. the runs, one after another
    for i, (method, seed, alpha) in enumerate(todo, 1):
        tag = run_tag(args.dataset, method, seed, alpha)
        out = run_dir(args.dataset, method, seed, alpha)
        if out.exists():                                  # unfinished earlier attempt
            out.rename(out.with_name(out.name + time.strftime(".incomplete-%Y%m%d-%H%M%S")))
        print(f"[{i}/{len(todo)}] {time.strftime('%Y-%m-%d %H:%M')} run {tag}", flush=True)
        start = time.time()
        log_path = RESULTS / "logs" / f"{tag}.log"
        # one line of grid_log.csv, completed when the run ends
        attempt = {"tag": tag, "dataset": args.dataset, "method": method, "seed": seed,
                   "alpha": "" if alpha is None else f"{alpha:g}", "status": "",
                   "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)),
                   "seconds": 0.0}
        try:
            # start the run and wait; everything it prints goes to logs/<tag>.log
            with open(log_path, "w") as log:
                code = subprocess.call(command(args.dataset, method, seed, alpha),
                                       stdout=log, stderr=subprocess.STDOUT, env=CHILD_ENV)
        # Ctrl+C: stop the Flower run inside the SuperLink too (otherwise it would
        # keep training in the background), record 'interrupted' and exit
        except KeyboardInterrupt:
            stopped = []
            if method in FEDERATED:
                run_id = started_run_id(log_path)
                for rid in ([run_id] if run_id else active_flower_runs()):
                    subprocess.call(["flwr", "stop", rid],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    stopped.append(rid)
            attempt.update(status="interrupted", seconds=round(time.time() - start, 1))
            log_attempt(attempt)
            print(f"\nInterrupted: {tag} was stopped"
                  + (f" (Flower run {', '.join(stopped)})" if stopped else "")
                  + ". Repeat the same command to continue; this run starts again.")
            sys.exit(130)
        # a run counts as done only if it exited normally AND wrote metrics.json
        ok = code == 0 and (out / "metrics.json").exists()
        seconds = round(time.time() - start, 1)
        attempt.update(status="ok" if ok else f"failed (exit {code})", seconds=seconds)
        log_attempt(attempt)
        print(f"      {'done' if ok else 'FAILED'} in {seconds / 60:.1f} min", flush=True)
        if not ok:
            failures.append(tag)
            print(f"      see {RESULTS / 'logs' / (tag + '.log')}")
            if not args.keep_going:
                sys.exit(1)
    if failures:
        print(f"{len(failures)} run(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("GRID_STEP_OK")


if __name__ == "__main__":
    main()
