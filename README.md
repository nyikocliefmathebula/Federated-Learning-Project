# Federated Learning-Based Intrusion Detection for Distributed IoT Networks

- **Author:** Nyiko Mathebula
- **Course:** EEE4022S Final-Year Project, Department of Electrical Engineering, University of Cape Town
- **Year:** 2026

This repository contains the code, configuration and data records of my final-year project.
It is submitted together with the project report. Section, table and equation numbers in the
code comments (for example "report Section 4.3") refer to that report.

## 1. Aim

The project asks how well federated learning detects network intrusions when the data are
spread over many IoT gateways whose traffic differs. Four ways of training the same detector
are compared:

| Approach | What is trained | Role |
|---|---|---|
| Centralized | one model on all training data in one place | upper reference |
| Local-only | one model per gateway, on its own data only | no-sharing baseline |
| FedAvg | one global model, federated averaging (Flower 1.39) | federated method |
| FedProx | FedAvg plus a proximal term, mu = 0.01 | federated method for non-IID data |

- **Gateways:** K = 105 simulated IoT gateways.
- **Label skew:** controlled with a Dirichlet distribution, alpha in {100, 10, 1, 0.5, 0.1}
  (from almost identical class mixes to gateways that see only a few classes).
- **Datasets** (used separately): corrected CIC-IDS2017 (7 classes) and CIC-IoT2023 (8 classes).
- **Repeats:** five master seeds (13, 42, 271, 1618, 3141).
- **Grid:** per dataset 5 centralized + 25 local-only + 25 FedAvg + 25 FedProx = 80 runs,
  160 runs in total.
- **Metrics:** accuracy is the primary test metric; macro-F1, per-class recall and the
  false-positive rate support it; communication is measured in serialized model bytes.

The model is the same for every approach: an MLP d_in -> 128 -> 64 -> C with LayerNorm, ReLU
and dropout 0.2 after each hidden layer, and the same initial weights for a given dataset and
seed.

## 2. Repository layout

    README.md              this file
    start_session.sh       activates .venv in an Ubuntu terminal: source start_session.sh
    requirements-base.txt  packages that were installed
    requirements-lock.txt  exact versions of every installed package
    src/                   data pipeline, baselines, grid runner, results summary, Figure 3.2
    src/report/            tables and figures of Chapter 5 and Appendix G of the report
    flower-ids/            Flower app (FedAvg and FedProx) and the shared package fliids
    tests/                 25 known-answer tests (python -m pytest tests -q)
    setup/                 environment checks and a two-client Flower installation check
    docs/                  logs of the pipeline steps, test output, environment records
    docs/datasets/         dataset manifest, archive check, checksums and line counts

Not in the repository (listed in `.gitignore`): the datasets and everything derived from them
(`downloads/`, `data/`), experiment outputs (`results/`), the Python environment (`.venv/`),
caches, and `private/` (personal working notes that are not part of the submission).

### Source files

| File | Purpose |
|---|---|
| `src/schema.py` | expected columns of both datasets, excluded columns, raw-data locations |
| `src/label_maps.py` | mapping of every source label to a task class |
| `src/check_archives.py` | checks that the extracted files match the downloaded archives |
| `src/clean_data.py` | stage 1: audit and clean one dataset |
| `src/build_arrays.py` | stage 2: store all cleaned records as fixed arrays |
| `src/split_data.py` | stage 3: 70/10/20 split, training subsample, preprocessing, initial model, round schedule |
| `src/partition_clients.py` | stage 4: Dirichlet partitions over the 105 gateways |
| `src/check_data.py` | checks every data output before training |
| `src/train_baseline.py` | centralized and local-only training |
| `src/run_grid.py` | runs the planned grid, one run at a time, with logs |
| `src/collect_results.py` | one summary table of all completed runs |
| `src/cross_gateway.py` | cross-gateway test: every local-only model on every gateway's test data |
| `src/plot_label_skew.py` | Figure 3.2: measured label skew of the gateway partitions |
| `src/report/` | tables and figures of Chapter 5 and Appendix G of the report, from the saved run files (Section 9) |
| `flower-ids/fliids/` | shared code: data loading, model, training, metrics, seeds, run records, Flower client and server |
| `tests/test_pipeline.py` | known-answer tests of the pipeline, training and metric code |

## 3. Environment

The experiments run on a Windows 11 laptop in WSL2 (Ubuntu 24.04), CPU only:
Python 3.12.3, Flower 1.39.0 (Simulation Runtime on Ray 2.55.1), PyTorch 2.14.1 (CPU),
NumPy 2.5.3, pandas 3.0.6, PyArrow 25.0.1 and matplotlib 3.11.2. The full list is in
`requirements-lock.txt`; the measured hardware is in `docs/hardware_environment.json`.

Setting up the environment (once, in an Ubuntu terminal, in the project folder):

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-lock.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu   # torch 2.14.1+cpu
    python setup/check_environment.py     # ends with ENVIRONMENT_CHECK_OK
    python setup/check_ray.py             # ends with RAY_CHECK_OK

Every later command runs in an Ubuntu terminal, from the project root, with `.venv` active:

    source /mnt/c/Users/Nyiko/fl-iot-ids/start_session.sh

The scripts stop with a message if they are started from another folder. The data and
results folders of the Flower app are absolute paths in `flower-ids/pyproject.toml`
(`data-root`, `results-root`); change them if the project is placed elsewhere.

## 4. Data

The datasets are not included because of their size (about 1.8 GB of archives). Sources,
download dates, archive checksums and the label mapping are recorded in
`docs/datasets/MANIFEST.md`.

| Dataset | Archive in `downloads/` | Extracted to |
|---|---|---|
| Corrected CIC-IDS2017 (Engelen et al., 2021; Liu et al., 2022) | `CICIDS2017_improved.zip` | `data/raw/cicids2017_improved/` (`monday.csv` ... `friday.csv`) |
| CIC-IoT2023 (Neto et al., 2023), per-class CSV release | `CSV.zip` | `data/raw/ciciot2023/` (34 class folders) |

Data pipeline, run once per dataset (shown for CIC-IDS2017; repeat with
`--dataset ciciot2023`). Each script prints the marker on the right when it succeeds:

    python src/check_archives.py                           # ARCHIVES_OK (both datasets)
    python src/clean_data.py --dataset cicids2017          # CLEANING_OK
    python src/build_arrays.py --dataset cicids2017        # ARRAYS_OK
    python src/split_data.py --dataset cicids2017          # SPLIT_OK
    python src/partition_clients.py --dataset cicids2017   # PARTITION_OK
    python src/check_data.py --dataset cicids2017          # DATA_CHECK_OK

All outputs go to `data/processed/<dataset>/`. Every stage writes a JSON record with counts
and the SHA-256 checksums of its inputs and outputs, and `check_data.py` checks all outputs
before any training.

| Result of the pipeline | CIC-IDS2017 | CIC-IoT2023 |
|---|---|---|
| Records read | 2,099,976 | 46,776,697 (+3 incomplete last lines) |
| Records kept after cleaning | 1,714,194 | 20,164,196 |
| Task classes | 7 | 8 |
| Model inputs | 82 (81 for seed 1618) | 39 |
| Data checks passed (9 October 2026) | 148 of 148 | 143 of 143 |

The logs of these steps are in `docs/` (`archive_check.log`, `clean_*.log`, `partition_*.log`,
`data_check_*.txt`).

Figure 3.2 of the report (client label skew) is drawn from the saved partitions:

    python src/plot_label_skew.py --dataset ciciot2023     # FIGURE_OK
    python src/plot_label_skew.py --dataset cicids2017

The PNG, PDF and CSV files are written to `results/figures/`.

## 5. Experiments

FedAvg and FedProx runs are submitted to a local Flower SuperLink. Start it in a separate
Ubuntu terminal and leave it running (centralized and local-only runs do not need it):

    FLWR_DISABLE_UPDATE_CHECK=1 PYTHONUNBUFFERED=1 flower-superlink \
      --insecure --simulation --disable-runtime-dependency-installation \
      --host 127.0.0.1 --port 39091

The Flower CLI configuration (`~/.flwr`) points its default connection to this address.

**Pilot.** Before the grid, a short pilot checks the whole chain on CIC-IDS2017, seed 42: a
two-gateway partition (`--clients 2 --suffix _dev2`), both baselines for at most 5 epochs,
FedAvg and FedProx with mu = 0 for 3 rounds (the two must give identical trajectories), and one
5-round FedAvg run with all 105 gateways. Pilot outputs go to `results/dev/` and
`results/dev_mu0/`, never to `results/runs/`.

**Grid.** `run_grid.py` runs the planned runs one at a time, logs every attempt and skips runs
that already have a `metrics.json`, so an interrupted grid is resumed by repeating the same
command. Centralized and local-only runs can run in a second terminal at the same time as the
federated runs (decision D11).

    # federated runs (needs the SuperLink): 50 runs per dataset
    python src/run_grid.py --dataset cicids2017 --methods fedavg fedprox --dry-run
    python src/run_grid.py --dataset cicids2017 --methods fedavg fedprox --keep-going

    # baselines: 30 runs per dataset
    python src/run_grid.py --dataset cicids2017 --methods centralized local --dry-run
    python src/run_grid.py --dataset cicids2017 --methods centralized local --keep-going

    # progress and summary table, at any time
    python src/collect_results.py --dataset cicids2017     # COLLECT_OK

    # after the local-only runs: every local model on every gateway's test data
    python src/cross_gateway.py --dataset cicids2017       # CROSS_GATEWAY_OK

CIC-IDS2017 is run first, then CIC-IoT2023. Ctrl+C stops the grid cleanly; for a federated
run the runner also stops the run inside the SuperLink.

Single runs can also be started directly, for example:

    python src/train_baseline.py --method centralized --dataset cicids2017 --seed 42
    python src/train_baseline.py --method local --dataset cicids2017 --seed 42 --alpha 1

    # federated: first set the simulation to 105 gateways, one training at a time
    flwr federation simulation-config --num-supernodes 105 \
      --client-resources-num-cpus 4 --client-resources-num-gpus 0 \
      --init-args-num-cpus 4 --init-args-num-gpus 0
    flwr run flower-ids --stream --run-config "dataset='cicids2017' method='fedprox' seed=42 alpha=0.1"

### Outputs of a run

Each run writes to `results/runs/<dataset>/<method>/seed<S>/[alpha<A>/]`
(C = centralized, L = local-only, F = FedAvg and FedProx):

| File | Runs | Content |
|---|---|---|
| `run_config.json` | C, L, F | all settings and their hash, input checksums, software versions, Git commit |
| `trajectory.csv` | C, L, F | validation score and training statistics per epoch or round |
| `communication.csv` | F | one row per model message: direction, parameter bytes, serialized bytes, totals |
| `node_map.json`, `schedule.json` | F | Flower node of each gateway; the 21 gateways of each round |
| `checkpoint.pt` / `local_models.pt` | C, F / L | trained model(s) at the best epoch or round |
| `predictions.parquet` | C, L, F | true and predicted class of every test record |
| `confusion_counts.parquet` | C, L, F | confusion counts per evaluation view and gateway |
| `metrics.json` | C, L, F | test metrics, stopping point and (F) communication up to the best round |
| `cross_gateway.parquet` | L | written by `cross_gateway.py`: model k on the test records of gateway j (accuracy, macro-F1 over the classes present, false-positive rate) for all 105 x 105 pairs |

`results/runs/grid_log.csv` and `results/runs/logs/` record every attempt of the grid;
`collect_results.py` writes `results/summary/<dataset>_runs.csv` and `cross_gateway.py`
writes `results/summary/<dataset>_cross_gateway.csv`.

## 6. Tests

    python -m pytest tests -q          # 25 passed

The tests use small inputs with known answers (no dataset needed): random streams and the
round schedule, the stratified and group-aware splits, the training subsample, preprocessing,
the Dirichlet partition and top-up rule, the heterogeneity measures, cleaning rules, the
archive check, the FedProx term, the class weights of the loss, early stopping, the metrics,
the grid plan, the results summary and the cross-gateway scores. The saved output is in
`docs/tests_v8.txt` (the first 23 tests, 7 October), `docs/tests_d18.txt` (24) and
`docs/tests_d19.txt` (all 25).

## 7. Decisions fixed for every run

| ID | Decision |
|---|---|
| D1 | CIC-IDS2017 "- Attempted" flows (11,979; 9,091 after cleaning) are relabelled Benign, as the dataset authors recommend; the original labels stay in the record index. |
| D2 | `Infiltration - Portscan` is mapped to PortScan. |
| D3 | Heartbleed (11 records) and Infiltration (36) are excluded as too rare for 105 gateways, and counted. |
| D4 | The CIC-IDS2017 destination port is not a model input (a service identifier and a testbed shortcut); Protocol is kept. |
| D5 | Exact duplicates and conflicting feature vectors are removed before splitting; conflicts are judged on the task class; the first copy is kept. |
| D6 | Each feature is transformed with sign(x) log(1 + \|x\|) and standardised with training-portion statistics; features constant in a seed's training portion are dropped. |
| D7 | 70/10/20 split stratified by source label: Flow ID groups per day for CIC-IDS2017, record level for CIC-IoT2023 (no session or device identifier). |
| D8 | CIC-IoT2023 training portion subsampled per source label to min(n, max(ceil(0.05 n), 10,000)); validation and test are complete. |
| D9 | K = 105 gateways with at least 512 training, 74 validation and 147 test records each, enforced by the top-up rule (redrawing cannot meet the minimum at alpha = 0.1). |
| D10 | Adam, learning rate 0.001, weight decay 0.00001, batch 256; at most 200 epochs (baselines) or 200 rounds of 2 local epochs (FedAvg, FedProx); early stopping on validation macro-F1 with patience 25 and minimum improvement 0.0001 (both as amended by D17); 21 of 105 gateways per round on a fixed schedule per seed; FedProx mu = 0.01. |
| D11 | One Flower client trains at a time; 4 PyTorch threads per run; baselines may run in a second process at the same time. |
| D12 | Grid of 160 runs, CIC-IDS2017 first, after the pilot. |
| D13 | No mid-run resume: an interrupted run is repeated from its start; client updates are aggregated in a fixed order so results repeat on the same computer. |
| D14 | Local-only checkpoints use macro-F1 over the classes present in the gateway's validation data; the Jensen-Shannon divergence uses base-2 logarithms. |
| D15 | Sensitivity analyses (destination port kept; attempted flows excluded) are outside the main grid. |
| D16 | WSL memory limit of 13 GB when two CIC-IoT2023 runs run at the same time. |
| D17 | Amendment of 14-15 October 2026, made after the first CIC-IDS2017 grid and based on its validation scores only: patience 15 -> 25 for every approach, and the FedAvg/FedProx budget 100 -> 200 rounds (the baselines already had 200 epochs). In that grid 63 of 80 runs were ended by the patience rule, and 16 of the 17 federated runs that reached round 100 had their best validation score in rounds 90-100, so they were still improving. Both datasets use the amended settings; the first grid is kept in `results/archive/` as a sensitivity analysis of the stopping rule. |
| D18 | Class-weighted cross-entropy was tested on 15 October 2026 on the validation data of seed 42 only, and not adopted. Weights proportional to (N / (C n_c))^p with p = 0.5 or 1 lowered the validation accuracy (the primary metric) and raised the false-positive rate in most configurations, and changed the approaches unequally: local-only and, with p = 1, FedAvg's detection of WebAttack gained, while centralized training lost. Every run of the grid uses plain cross-entropy (`class-weight-power = 0.0`, `--class-weight-power 0`); the option remains for a sensitivity analysis. |
| D19 | Cross-gateway view added on 16 October 2026, after the CIC-IDS2017 grid and before CIC-IoT2023: every saved local-only model is also scored on the test data of the other 104 gateways (`src/cross_gateway.py`). It is an extra evaluation of the same models; no model or setting changed. Local-only models are compared with the shared models mainly on other gateways' data, the question a model that is reused elsewhere faces; the own-gateway (routed) view is still reported. |

## 8. Reproducibility

- Every random choice comes from its own stream derived from the master seed
  (`flower-ids/fliids/seeds.py`), so the split, partitions, initial weights and round schedule
  are the same on every run, and FedAvg and FedProx see the same schedule and minibatches.
- `run_config.json` records the Git commit of the code. "+uncommitted" is added when the
  working tree has changes, so runs are started from a clean, committed state.
- `data_manifest.json` records the SHA-256 of `clean_data.py` at cleaning time. The code
  comments were revised after the data were processed; the processing code itself is
  unchanged, so the current file hash differs from the recorded one.

## 9. Results of the report

All 160 runs of the grid completed: CIC-IDS2017 on commit 54df42e and CIC-IoT2023 on d982ef8
(`results/runs/grid_log.csv`; one interrupted attempt was repeated from its start). The
summaries and cross-gateway scores were then written for both datasets:

    python src/collect_results.py --dataset cicids2017     # COLLECT_OK
    python src/cross_gateway.py --dataset cicids2017       # CROSS_GATEWAY_OK
    python src/collect_results.py --dataset ciciot2023     # COLLECT_OK
    python src/cross_gateway.py --dataset ciciot2023       # CROSS_GATEWAY_OK

The tables and figures of Chapter 5 and Appendix G of the report are produced by the scripts in
`src/report/` from the saved run folders and, for the stopping-rule sensitivity analysis (D17),
from the first CIC-IDS2017 grid in `results/archive/runs_p15_r100/`. They are run in this order,
write only to `results/report/` (`csv/`, `tables/`, `figures/`) and change no run, model or
setting:

    python src/report/collect.py           # summary tables of all runs
    python src/report/paired.py            # paired differences between approaches (Eq. 3.15)
    python src/report/stability.py         # round-to-round change of validation macro-F1
    python src/report/tables.py            # Tables 5.1 to 5.6
    python src/report/appendix_tables.py   # Table G1
    python src/report/figs.py              # Figures 5.1 to 5.6
    python src/report/extra.py             # other numbers quoted in Chapter 5
