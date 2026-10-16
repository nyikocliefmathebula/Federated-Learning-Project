# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Validation checks with known answers (report Section 4.8).

Run from the project root:  python -m pytest tests -q
"""
# Each test_... function below builds a small made-up input whose correct
# answer is known in advance, runs one piece of the real project code on it,
# and checks the answer with assert. pytest runs every test_... function and
# reports how many passed. No real dataset is needed. 'python -m pytest tests -q'
# prints one dot per passed test and ends with e.g. '23 passed'.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# ROOT = the project folder; add src/ and flower-ids/ to Python's search path so
# the project files can be imported like libraries
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "flower-ids")]

import clean_data  # noqa: E402
from fliids.evaluation import confusion, confusion_by_client, metrics  # noqa: E402
from fliids.models import make_model, state_sha256  # noqa: E402
from fliids.seeds import make_schedule, rng, torch_seed  # noqa: E402
from fliids.training import (EarlyStopping, class_weights,  # noqa: E402
                             make_optimizer, proximal_term, train_epochs)
from partition_clients import assign, describe, draw, top_up  # noqa: E402
from schema import (CICIDS2017_EXCLUDED, CICIDS2017_HEADER,  # noqa: E402
                    CICIOT2023_KNOWN_TRUNCATED, ciciot2023_base)
import check_archives  # noqa: E402
import collect_results  # noqa: E402
import cross_gateway  # noqa: E402
import run_grid  # noqa: E402
from split_data import (fit_preprocessing, group_split, record_split,  # noqa: E402
                        subsample_training)


# ------------------------------------------------------------------ seeds
# Random streams: the same inputs always give the same numbers, different
# purposes give different numbers, and the schedule's first rounds do not
# depend on how many rounds are requested. Each round has 21 different gateways.
def test_streams_are_stable_and_separate():
    assert torch_seed(42, "init") == torch_seed(42, "init")
    assert torch_seed(42, "init") != torch_seed(42, "split")
    a = rng(13, "partition", 100, 0, 0).random(3)
    b = rng(13, "partition", 100, 0, 0).random(3)
    assert np.array_equal(a, b)
    assert make_schedule(42, 105, 21, 5)[:3] == make_schedule(42, 105, 21, 100)[:3]
    assert all(len(set(r)) == 21 for r in make_schedule(42, 105, 21, 10))


# ------------------------------------------------------------------ split
# Three labels with 1000, 200 and 50 records: each must get exactly 20 % test
# and 10 % validation (rounded).
def test_record_split_is_exact_per_stratum():
    strata = np.repeat(np.arange(3), [1000, 200, 50])
    portion = record_split(strata, rng(1, "split", 0))
    for s, n in ((0, 1000), (1, 200), (2, 50)):
        counts = np.bincount(portion[strata == s], minlength=3)
        assert counts[2] == round(0.2 * n) and counts[1] == round(0.1 * n)


# 3000 made-up groups of 1 to 5 records: no group may be split over two
# portions, and the overall shares must be within 1 % of 70/10/20.
def test_group_split_keeps_groups_together():
    g = np.random.default_rng(0)
    groups = np.repeat(np.arange(3000), g.integers(1, 6, 3000))
    strata = (groups % 3).astype(np.int64)
    portion = group_split(strata, groups, rng(2, "split", 0))
    per_group = pd.Series(portion).groupby(groups).nunique()
    assert per_group.max() == 1
    shares = np.bincount(portion, minlength=3) / len(portion)
    assert np.allclose(shares, [0.7, 0.1, 0.2], atol=0.01)


# Label 0 has 14,000 training records: 5 % = 700 is below the floor of 1000,
# so 1000 are kept. Label 1 has 210: fewer than the floor, so all are kept.
# Validation and test records must not change.
def test_subsampling_touches_training_only():
    strata = np.repeat([0, 1], [20000, 300])
    portion = record_split(strata, rng(3, "split", 0))
    new, kept = subsample_training(portion, strata, 0.05, 1000, rng(3, "subsample"))
    assert np.array_equal(new[portion != 0], portion[portion != 0])
    assert kept[0] == (14000, 1000) and kept[1] == (210, 210)
    assert np.sum((new == 0) & (strata == 0)) == 1000


# Fitted on rows 0-4 only: column 1 (always 7) and column 2 (all zeros in rows
# 0-4) are constant there and are dropped. Column 0 has values 0..4, so its mean
# is 2 and its scale is their standard deviation.
def test_preprocessing_uses_training_rows_and_drops_constants():
    X = np.column_stack([np.arange(10.0), np.full(10, 7.0), np.r_[np.zeros(5), np.ones(5)]])
    keep, mean, scale = fit_preprocessing(X.astype(np.float32), np.arange(5), "standard")
    assert keep.tolist() == [0]                    # column 2 is constant in rows 0-4
    assert np.isclose(mean[0], 2.0) and np.isclose(scale[0], np.std(np.arange(5.0)))


# ------------------------------------------------------------------ partition
# Strong skew (alpha 0.1) with the top-up rule: every record is given to exactly
# one of 20 gateways, every gateway meets its minimum, and the same draw
# repeated gives the same result.
def test_allocation_keeps_every_record_and_meets_minimums():
    y = {0: np.repeat(np.arange(4), [60000, 8000, 3000, 70]),
         1: np.repeat(np.arange(4), [8000, 1200, 400, 10]),
         2: np.repeat(np.arange(4), [16000, 2400, 800, 20])}
    floors = {0: 512, 1: 74, 2: 147, "clients": 20}
    result, ok = draw(y, 4, 0.1, 42, 0, floors, "topup")
    assert ok
    for j in (0, 1, 2):
        counts = result["counts"][j]
        assert counts.sum() == len(y[j]) and counts.sum(0).min() >= floors[j]
        assert np.array_equal(np.bincount(result["client"][j], minlength=20), counts.sum(0))
    again, _ = draw(y, 4, 0.1, 42, 0, floors, "topup")
    assert np.array_equal(again["client"][0], result["client"][0])


# With the reject rule, alpha 0.1 fails in all 20 attempts, while the nearly
# uniform alpha 100 succeeds within 3 attempts.
def test_reject_rule_reports_failure_at_strong_skew():
    y = {0: np.repeat(np.arange(4), [60000, 8000, 3000, 70]),
         1: np.repeat(np.arange(4), [8000, 1200, 400, 10]),
         2: np.repeat(np.arange(4), [16000, 2400, 800, 20])}
    floors = {0: 512, 1: 74, 2: 147, "clients": 20}
    outcomes = [draw(y, 4, 0.1, 42, a, floors, "reject")[1] for a in range(20)]
    assert not any(outcomes)                       # alpha = 0.1: every draw rejected
    assert any(draw(y, 4, 100, 42, a, floors, "reject")[1] for a in range(3))


# Every gateway already has 600 records (more than 512): nothing is moved.
def test_top_up_changes_nothing_when_minimum_met():
    counts = np.full((2, 5), 300)
    new, moved = top_up(counts, counts.astype(float), 512, np.random.default_rng(1))
    assert moved == 0 and np.array_equal(new, counts)


# Equal class mixes give entropy 1, JSD 0 and size CV 0. Two gateways that
# each hold only one class give entropy 0 and JSD 0.3112781 (worked out by hand
# with log base 2).
def test_heterogeneity_measures():
    iid = describe(np.full((4, 3), 10))
    assert np.isclose(iid["mean_normalized_entropy"], 1.0)
    assert np.isclose(iid["mean_jsd_to_global"], 0.0) and iid["quantity_cv"] == 0.0
    skew = describe(np.array([[10, 0], [0, 10]]))
    assert np.isclose(skew["mean_normalized_entropy"], 0.0)
    assert skew["mean_jsd_to_global"] == pytest.approx(0.3112781, abs=1e-6)
    assert skew["classes_present"] == [1, 1]


# ------------------------------------------------------------------ cleaning
# helper: a tiny CIC-IDS2017 table with all 91 columns; each row is given as
# (Label, Attempted Category, Fwd Header Length) and every other value is 0
def _cicids_chunk(rows):
    data = {c: [0] * len(rows) for c in CICIDS2017_HEADER}
    df = pd.DataFrame(data)
    df["id"] = np.arange(1, len(rows) + 1)
    for c in ("Flow ID", "Src IP", "Dst IP", "Timestamp"):
        df[c] = "x"
    df["Label"] = [r[0] for r in rows]
    df["Attempted Category"] = [r[1] for r in rows]
    df["Fwd Header Length"] = [r[2] for r in rows]
    return df


# helper: the settings clean_chunk needs, with the chosen attempted policy
def _ctx(policy):
    from collections import Counter
    return {"dataset": "cicids2017", "attempted": policy, "excluded": CICIDS2017_EXCLUDED,
            "nonnegative": ["Fwd Header Length", "Bwd Header Length"],
            "id_mismatch": Counter()}


# Five rows: BENIGN (kept), an attempted Botnet (relabelled Benign, or removed
# under --attempted exclude), Heartbleed (rare: excluded), DDoS with a negative
# header length (removed) and DoS Hulk (kept as DoS). Record keys must keep the
# original row positions 0, 1 and 4.
def test_cleaning_policies_and_reasons():
    chunk = _cicids_chunk([("BENIGN", -1, 20), ("Botnet - Attempted", 0, 20),
                           ("Heartbleed", -1, 20), ("DDoS", -1, -5), ("DoS Hulk", -1, 20)])
    path = Path("monday.csv")
    kept, _, label, cls, step = clean_data.clean_chunk(chunk.copy(), _ctx("benign"), path, None, 0)
    assert list(step) == ["", "", "excluded_rare_class", "negative_value", ""]
    assert kept["class"].tolist() == ["Benign", "Benign", "DoS"]
    assert kept["record_key"].tolist() == ["monday.csv:0", "monday.csv:1", "monday.csv:4"]
    kept, _, _, _, step = clean_data.clean_chunk(chunk.copy(), _ctx("exclude"), path, None, 0)
    assert step[1] == "attempted" and len(kept) == 2


# An unknown label ('Mystery') must stop cleaning; a row whose last values
# are missing must be marked malformed_row.
def test_cleaning_stops_on_unmapped_label_and_flags_short_rows():
    chunk = _cicids_chunk([("BENIGN", -1, 20), ("Mystery", -1, 20)])
    with pytest.raises(clean_data.CleaningError):
        clean_data.clean_chunk(chunk, _ctx("benign"), Path("monday.csv"), None, 0)
    short = _cicids_chunk([("BENIGN", -1, 20), ("BENIGN", -1, 20)])
    short.loc[1, ["Label", "Attempted Category"]] = np.nan
    _, _, _, _, step = clean_data.clean_chunk(short, _ctx("benign"), Path("monday.csv"), None, 0)
    assert step[1] == "malformed_row"


# tmp_path is an empty temporary folder that pytest creates for this test.
# The class folders may sit directly in ciciot2023/ or one level deeper; two
# copies or none must stop with an error.
def test_ciciot2023_folder_layouts(tmp_path):
    flat = tmp_path / "flat"
    (flat / "Benign_Final").mkdir(parents=True)
    (flat / "XSS").mkdir()
    assert ciciot2023_base(flat) == flat                      # current layout
    nested = tmp_path / "nested"
    (nested / "CIC-IoT2023" / "Benign_Final").mkdir(parents=True)
    assert ciciot2023_base(nested) == nested / "CIC-IoT2023"  # one extra level
    (nested / "CSV" / "Benign_Final").mkdir(parents=True)
    with pytest.raises(FileExistsError):                      # two copies
        ciciot2023_base(nested)
    with pytest.raises(FileNotFoundError):
        ciciot2023_base(tmp_path / "empty_missing")


# Only a listed file with the expected size and checksum counts as a known
# truncation; the same name with different content, or any other file, is
# unknown; a complete file is not marked at all.
def test_only_known_release_truncations_pass_without_switch():
    name, (size, digest) = next(iter(CICIOT2023_KNOWN_TRUNCATED.items()))
    files = [
        {"file": f"ciciot2023/{name}", "bytes": size, "sha256": digest,
         "ends_with_newline": False},                         # known release issue
        {"file": f"ciciot2023/{name}".replace(".pcap", "x.pcap"), "bytes": size,
         "sha256": digest, "ends_with_newline": False},       # other file: unknown
        {"file": "ciciot2023/XSS/XSS.pcap.csv", "bytes": 10,
         "ends_with_newline": True},                          # complete file
    ]
    known, unknown = clean_data.classify_truncation("ciciot2023", files)
    assert known == {files[0]["file"]} and unknown == {files[1]["file"]}
    changed = dict(files[0], sha256="0" * 64)                 # same name, other content
    assert clean_data.classify_truncation("ciciot2023", [changed]) == (set(), {changed["file"]})
    assert "truncation" not in files[2]


# Build a small zip and a matching folder: first everything matches; then one
# file is changed (same size, so only the CRC-32 can tell), one is deleted and
# one extra file is added, and all three must be reported.
def test_archive_comparison_finds_changed_missing_and_extra_files(tmp_path):
    import zipfile
    folder = tmp_path / "raw"
    (folder / "A").mkdir(parents=True)
    (folder / "A" / "a.csv").write_text("x,y\n1,2\n")
    (folder / "A" / "b.csv").write_text("x,y\n3,4\n")
    archive = tmp_path / "CSV.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("CSV/A/a.csv", "x,y\n1,2\n")
        zf.writestr("CSV/A/b.csv", "x,y\n3,4\n")
    r = check_archives.compare(archive, folder, progress=False)
    assert r["ok"] and r["matched"] == 2 and r["archive_top_folder"] == "CSV/"
    (folder / "A" / "b.csv").write_text("x,y\n3,5\n")      # same size, new content
    (folder / "A" / "a.csv").unlink()
    (folder / "A" / "notes.txt").write_text("mine")
    r = check_archives.compare(archive, folder, progress=False)
    assert not r["ok"]
    assert r["mismatched"][0]["reason"] == "CRC-32 differs"
    assert r["missing_on_disk"] == ["A/a.csv"] and r["extra_on_disk"] == ["A/notes.txt"]


# ------------------------------------------------------------------ training
# Two identical models trained with the same random streams: FedProx with a
# tiny mu must give the same weights as plain training (mu = 0). Then every
# weight is moved by 1, so the sum of squared differences from the reference
# (the proximal term) must equal the number of weights.
def test_proximal_term_and_mu_zero_equivalence():
    torch.manual_seed(0)
    x, y = torch.randn(64, 5), torch.randint(0, 3, (64,))
    a, b = make_model(5, 3, seed=1), make_model(5, 3, seed=1)
    assert state_sha256(a.state_dict()) == state_sha256(b.state_dict())
    torch.manual_seed(5)
    train_epochs(a, x, y, 1, 16, make_optimizer(a), torch.Generator().manual_seed(9), mu=0.0)
    torch.manual_seed(5)
    train_epochs(b, x, y, 1, 16, make_optimizer(b), torch.Generator().manual_seed(9), mu=1e-12)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)
    ref = [p.detach().clone() for p in a.parameters()]
    for p in a.parameters():
        p.data += 1.0
    assert proximal_term(a, ref).item() == sum(p.numel() for p in ref)


# 70 records in batches of 32 give 3 steps per epoch (32 + 32 + 6); two epochs
# give 6 steps and 140 examples.
def test_training_counts_examples_and_steps():
    model = make_model(4, 2, seed=3)
    stats = train_epochs(model, torch.randn(70, 4), torch.randint(0, 2, (70,)), 2, 32,
                         make_optimizer(model), torch.Generator().manual_seed(1))
    assert stats["examples"] == 140 and stats["steps"] == 6   # 32+32+6 per epoch


# Class weights (D18) for 900, 90 and 10 training records: power 0 means no
# weighting; power 1 gives every class the same total weight (w_c * n_c); power
# 0.5 makes the weights proportional to 1/sqrt(n_c). In every case the average
# weight per record is 1. Uniform weights must train exactly like no weights.
def test_class_weights_and_weighted_loss():
    n = torch.tensor([900.0, 90.0, 10.0])
    assert class_weights(n, 0) is None
    full, half = class_weights(n, 1.0), class_weights(n, 0.5)
    for w in (full, half):
        assert torch.isclose((w * n).sum(), n.sum())                 # mean weight 1
    assert torch.allclose(full * n, torch.full((3,), 1000 / 3))      # balanced totals
    assert torch.isclose(half[2] / half[0], torch.tensor(90.0).sqrt())
    with pytest.raises(ValueError):
        class_weights([5, 0, 3], 0.5)
    torch.manual_seed(0)
    x, y = torch.randn(64, 5), torch.randint(0, 3, (64,))
    a, b = make_model(5, 3, seed=1), make_model(5, 3, seed=1)
    torch.manual_seed(5)
    train_epochs(a, x, y, 1, 16, make_optimizer(a), torch.Generator().manual_seed(9))
    torch.manual_seed(5)
    train_epochs(b, x, y, 1, 16, make_optimizer(b), torch.Generator().manual_seed(9),
                 class_weight=torch.ones(3))
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)


# Patience 3, min-delta 0.0001: a tie is not an improvement; a gain smaller
# than 0.0001 makes that step the best but does not reset the miss count; the
# third miss stops training.
def test_early_stopping_rule():
    stop = EarlyStopping(patience=3, min_delta=1e-4)
    s = {"w": torch.zeros(1)}
    assert not stop.update(0, 0.50, s)
    assert not stop.update(1, 0.50, s)                 # tie keeps the earlier step
    assert stop.best_step == 0 and stop.misses == 1
    assert not stop.update(2, 0.50005, s)              # better, but < 1e-4: no reset
    assert stop.best_step == 2 and stop.misses == 2
    assert stop.update(3, 0.40, s)                     # third miss -> stop


# ------------------------------------------------------------------ metrics
# [[8, 2], [1, 9]]: 17 of 20 correct = 0.85 accuracy; 2 of 10 benign records
# flagged as attacks = 0.2 FPR. With no benign records the FPR is None. A class
# that never occurs counts as F1 = 0 in macro_f1 but is left out of
# macro_f1_present.
def test_metrics_from_known_confusion_counts():
    m = metrics([[8, 2], [1, 9]])
    assert m["accuracy"] == 0.85 and abs(m["fpr"] - 0.2) < 1e-12
    assert m["recall"] == [0.8, 0.9]
    assert metrics([[0, 0], [1, 9]])["fpr"] is None    # no benign records
    m = metrics([[5, 0, 0], [0, 5, 0], [0, 0, 0]])     # class 2 absent and unpredicted
    assert m["macro_f1"] == pytest.approx(2 / 3) and m["macro_f1_present"] == 1.0


# Adding up the per-gateway confusion matrices must give the global one;
# assign() must hand out records exactly as the count matrix says.
def test_confusion_by_client_sums_to_global():
    y = np.array([0, 1, 1, 0, 2, 2])
    p = np.array([0, 1, 0, 0, 2, 1])
    c = np.array([0, 0, 1, 1, 2, 2])
    per = confusion_by_client(y, p, c, 3, 3)
    assert np.array_equal(per.sum(0), confusion(y, p, 3))
    assert assign(np.array([0, 0, 1]), np.array([[1, 1], [0, 1]]),
                  np.random.default_rng(0)).tolist().count(1) == 2


# ------------------------------------------------------------------ grid and records
# The plan has 80 runs per dataset (5 centralized + 3 x 25), in the expected
# order, with the expected folder names and commands.
def test_grid_plan_covers_the_four_configurations():
    plan = run_grid.plan_runs(["centralized", "local", "fedavg", "fedprox"],
                              [13, 42, 271, 1618, 3141], [100, 10, 1, 0.5, 0.1])
    assert len(plan) == 80                                   # per dataset
    assert sum(m == "centralized" for m, _, _ in plan) == 5
    assert plan[0] == ("centralized", 13, None) and plan[1] == ("local", 13, 100)
    assert run_grid.run_dir("cicids2017", "fedprox", 42, 0.5).as_posix() == \
        "results/runs/cicids2017/fedprox/seed42/alpha0.5"
    assert run_grid.run_dir("cicids2017", "local", 42, 100.0).name == "alpha100"
    cmd = run_grid.command("ciciot2023", "fedavg", 13, 0.1)
    assert cmd[:3] == ["flwr", "run", "flower-ids"]
    assert cmd[-1] == "dataset='ciciot2023' method='fedavg' seed=13 alpha=0.1"
    assert run_grid.shown(cmd).endswith('--run-config "dataset=\'ciciot2023\' '
                                        'method=\'fedavg\' seed=13 alpha=0.1"')
    assert run_grid.command("cicids2017", "local", 13, 1.0)[-2:] == ["--alpha", "1"]


# monkeypatch temporarily replaces a function for this test only. Here the
# flwr command is faked, so no SuperLink is needed: the run id must be found
# in a log, only 'running' and 'starting' runs count as active, and a missing
# flwr program gives an empty list instead of a crash.
def test_grid_finds_flower_runs_to_stop(tmp_path, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text("Using SuperLink: local (127.0.0.1:39091)\n"
                   "Successfully started run 6764742101564904333\nround 1: ...\n")
    assert run_grid.started_run_id(log) == "6764742101564904333"
    assert run_grid.started_run_id(tmp_path / "missing.log") is None

    class Done:
        stdout = ('{"success": true, "runs": [{"run-id": "11", "status": "running"}, '
                  '{"run-id": "12", "status": "finished:completed"}, '
                  '{"run-id": "13", "status": "starting"}]}')
    monkeypatch.setattr(run_grid.subprocess, "run", lambda *a, **k: Done())
    assert run_grid.active_flower_runs() == ["11", "13"]

    def no_flwr(*a, **k):
        raise FileNotFoundError("flwr")
    monkeypatch.setattr(run_grid.subprocess, "run", no_flwr)
    assert run_grid.active_flower_runs() == []


# The gateway data cache in client_app.py: loading gateway 7 twice reads the
# files once; a different seed (another run) empties the cache first.
def test_client_data_is_loaded_once_per_gateway_and_run(monkeypatch):
    import warnings
    with warnings.catch_warnings():          # Flower's CLI library warns on import
        warnings.simplefilter("ignore", DeprecationWarning)
        from fliids import client_app
    calls = []
    monkeypatch.setattr(client_app, "select_rows",
                        lambda *a, **k: calls.append(k["client"]) or np.arange(3))
    monkeypatch.setattr(client_app, "load_xy",
                        lambda *a: (torch.zeros(3, 2), torch.zeros(3, dtype=torch.int64)))
    client_app._DATA.clear()
    first = client_app.gateway_data("r", "d", 42, 1.0, "", 7)
    again = client_app.gateway_data("r", "d", 42, 1.0, "", 7)
    client_app.gateway_data("r", "d", 42, 1.0, "", 8)
    assert first is again and calls == [7, 8] and first[0] == 3
    client_app.gateway_data("r", "d", 13, 1.0, "", 7)      # another run clears the cache
    assert calls == [7, 8, 7] and len(client_app._DATA) == 1
    client_app._DATA.clear()


# Fake metrics.json files for one centralized and one FedAvg run: the summary
# rows must be complete where a file exists, and the gateway averages must
# leave out gateways whose value is None.
def test_collect_results_reads_each_kind_of_run(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(collect_results, "RESULTS", tmp_path)
    test = {"accuracy": 0.9, "macro_f1": 0.8, "fpr": 0.01}
    gw = [{"client": 0, "n": 10, "accuracy": 1.0, "macro_f1_present": 1.0, "fpr": 0.0},
          {"client": 1, "n": 10, "accuracy": 0.5, "macro_f1_present": 0.5, "fpr": None}]
    c = tmp_path / "cicids2017" / "centralized" / "seed42"
    c.mkdir(parents=True)
    (c / "metrics.json").write_text(json.dumps({
        "global_test": test, "gateway_matched": {"alpha1": gw}, "best_epoch": 3,
        "stopped_epoch": 18, "best_val_macro_f1": 0.7, "seconds": 5}))
    f = tmp_path / "cicids2017" / "fedavg" / "seed42" / "alpha1"
    f.mkdir(parents=True)
    (f / "metrics.json").write_text(json.dumps({
        "global_test": test, "gateway_matched": gw, "best_round": 9, "stopped_round": 24,
        "best_val_macro_f1": 0.7, "serialized_bytes_at_best": 100,
        "serialized_bytes_at_stop": 200, "seconds": 5}))
    rows = collect_results.rows_for("cicids2017", "centralized", 42)
    assert len(rows) == 5 and all(r["complete"] for r in rows)
    at1 = next(r for r in rows if r["alpha"] == 1.0)
    assert at1["gateway_mean_accuracy"] == 0.75 and at1["gateway_mean_fpr"] == 0.0
    fed = collect_results.rows_for("cicids2017", "fedavg", 42)
    assert [r["complete"] for r in fed] == [False, False, True, False, False]
    assert next(r for r in fed if r["complete"])["best_step"] == 9


# Cross-gateway view: 6 test records, gateways 0 and 1 (three records each).
# Model 0 is right on gateway 0 and on 2 of 3 records of gateway 1; model 1
# gets one benign record of gateway 0 wrong (a false positive, FPR 1/2). The
# summary separates own-gateway pairs (k = j) from the others.
def test_cross_gateway_pairs_and_summary():
    y = np.array([0, 0, 1, 0, 1, 1])
    client = np.array([0, 0, 0, 1, 1, 1])
    rows = cross_gateway.pair_rows(0, y, np.array([0, 0, 1, 0, 1, 0]), client, 2, 2)
    rows += cross_gateway.pair_rows(1, y, np.array([1, 0, 1, 0, 1, 1]), client, 2, 2)
    t = pd.DataFrame(rows).set_index(["model", "gateway"])
    assert t.loc[(0, 0), "accuracy"] == 1.0 and t.loc[(0, 1), "accuracy"] == 2 / 3
    assert t.loc[(1, 0), "accuracy"] == 2 / 3 and t.loc[(1, 0), "fpr"] == 0.5
    s = cross_gateway.summarise(t.reset_index())
    assert s["own_accuracy"] == 1.0                     # (0,0) and (1,1) are both right
    assert s["other_accuracy"] == pytest.approx(2 / 3)  # (0,1) and (1,0)

