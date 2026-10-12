# Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town 
# 2026

"""Stage 4: Dirichlet gateway partitions (report Sections 3.5 and 4.3.2).

Run from the project root after split_data.py:
    python src/partition_clients.py --dataset cicids2017
    python src/partition_clients.py --dataset ciciot2023

For each seed and alpha, class c's allocation vector p_c ~ Dirichlet(alpha 1_K)
is drawn once and used for the training, validation and test portions, with
independent multinomial counts for each portion. Each gateway needs at least
512 training records and 74 validation / 147 test records (512 x 10/70 and
512 x 20/70, rounded up). Two rules are available when a draw misses that:

  --support-rule topup   (default) under-supported gateways receive extra
                         records in proportion to their own expected class mix,
                         taken from the gateways holding most of those classes;
                         no donor drops below the minimum. Works at every alpha.
  --support-rule reject  redraw until every gateway meets the minimum, up to
                         --max-attempts; rejected draws are recorded and the
                         run stops at the limit (fails at alpha = 0.1).

Outputs in data/processed/<dataset>/seed<S>/alpha<A>/:
    partition.parquet       row_id, portion (0/1/2), client (0..K-1)
    dirichlet.npy           class-by-client allocation probabilities
    partition_summary.json  sizes, classes present, entropy, JSD, CV, histograms,
                            attempts, top-up moves and checksums
"""
# The idea in simple terms: for each class, a Dirichlet draw gives 105 shares
# that add up to 1 (one share per gateway). With a large alpha (100) the shares
# are almost equal, so every gateway sees a similar class mix (close to IID).
# With a small alpha (0.1) a few gateways get most of a class, so each gateway
# sees only a few classes (strongly non-IID). The same shares are used for the
# training, validation and test portions, so a gateway's test records look like
# its own training records.
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from schema import PROCESSED, require_project_root

# use the random streams of fliids/seeds.py (same code as the Flower app)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flower-ids"))
from fliids.seeds import alpha_id, rng  # noqa: E402

# portion numbers (as in split_indices.parquet) and their names
NAMES = {0: "train", 1: "validation", 2: "test"}


# SHA-256 fingerprint of a file, read in 4 MB blocks
def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def top_up(counts, weights, floor, g):
    """Raise every client to `floor` records (see module docstring)."""
    # counts[c, k] = records of class c given to gateway k; weights[c, k] = how many
    # class-c records gateway k was expected to get (its own class mix)
    counts = counts.copy()
    n_clients = counts.shape[1]
    moved = 0
    # visit the gateways in random order; top up each one that is below the floor
    for k in g.permutation(n_clients):
        need = floor - counts[:, k].sum()
        while need > 0:
            totals = counts.sum(0)
            # donors: other gateways that have more than the minimum
            donors = totals > floor
            donors[k] = False
            # records of each class that donors hold; prefer the classes this gateway
            # was expected to have (if none are available, any available class)
            available = (counts * donors).sum(1)
            w = weights[:, k] * (available > 0)
            if w.sum() == 0:
                w = (available > 0).astype(float)
            if w.sum() == 0:
                raise RuntimeError("Minimum client support is infeasible")
            # how many of the needed records to take from each class
            wanted = g.multinomial(need, w / w.sum())
            progress = 0
            for c in np.flatnonzero(wanted):
                # take from the gateway holding the most class-c records first, but never
                # leave a donor below the minimum
                for j in np.argsort(-counts[c], kind="stable"):
                    if wanted[c] == 0:
                        break
                    if j == k or not donors[j] or counts[c, j] == 0:
                        continue
                    take = min(wanted[c], counts[c, j], counts[:, j].sum() - floor)
                    if take <= 0:
                        continue
                    counts[c, j] -= take
                    counts[c, k] += take
                    wanted[c] -= take
                    progress += take
            # stop with an error if nothing could be moved (avoids an endless loop)
            if progress == 0:
                raise RuntimeError("Top-up made no progress")
            need -= progress
            moved += progress
    return counts, moved


def assign(labels, counts, g):
    """Shuffle each class and hand out records according to the count matrix."""
    client = np.empty(len(labels), dtype=np.int16)
    for c in range(counts.shape[0]):
        # shuffle the records of class c, then give the first counts[c, 0] to gateway
        # 0, the next counts[c, 1] to gateway 1, and so on
        idx = g.permutation(np.flatnonzero(labels == c))
        client[idx] = np.repeat(np.arange(counts.shape[1]), counts[c])
    return client


def describe(counts):
    """Realised heterogeneity of a class-by-client count matrix (Eq. 3.5-3.6)."""
    # sizes = records per gateway; q = each gateway's class shares; glob = class
    # shares of the whole portion
    sizes = counts.sum(0)
    q = counts / np.maximum(sizes, 1)
    glob = counts.sum(1) / counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        # normalized entropy: 1 = all classes equally present, 0 = a single class
        logs = np.where(q > 0, np.log(q), 0.0)
        entropy = -(q * logs).sum(0) / np.log(len(glob))
        # Jensen-Shannon divergence between each gateway's class mix and the global
        # mix (0 = identical, 1 = completely different; log base 2)
        mid = 0.5 * (q + glob[:, None])
        def kl(a, b):
            return np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / b), 0.0).sum(0)
        jsd = 0.5 * kl(q, mid) + 0.5 * kl(np.repeat(glob[:, None], q.shape[1], 1), mid)
    return {
        "sizes": sizes.tolist(),
        "min_size": int(sizes.min()), "max_size": int(sizes.max()),
        # coefficient of variation of gateway sizes: spread of sizes / mean size
        "quantity_cv": float(sizes.std() / sizes.mean()),
        "classes_present": (counts > 0).sum(0).tolist(),
        "mean_classes_present": float((counts > 0).sum(0).mean()),
        "normalized_entropy": np.round(entropy, 6).tolist(),
        "mean_normalized_entropy": float(entropy.mean()),
        "jsd_to_global": np.round(jsd, 6).tolist(),
        "mean_jsd_to_global": float(jsd.mean()),
        "class_by_client": counts.tolist(),
    }


def draw(y_by_portion, n_classes, alpha, seed, attempt, floors, rule):
    """One Dirichlet draw and its three count matrices."""
    aid = alpha_id(alpha)
    clients = floors["clients"]
    # p[c] = shares of class c over the 105 gateways (one row per class)
    p = rng(seed, "partition", aid, attempt, 0).dirichlet(
        np.full(clients, alpha), size=n_classes)
    result = {"p": p, "counts": {}, "client": {}, "moved": {}, "short": {}}
    for j in (0, 1, 2):
        labels = y_by_portion[j]
        # for each portion (train, validation, test) with its own random stream:
        g = rng(seed, "partition", aid, attempt, 1 + j)
        n_c = np.bincount(labels, minlength=n_classes)
        # split the n_c records of each class over the gateways using the shares p[c]
        counts = np.stack([g.multinomial(n_c[c], p[c]) for c in range(n_classes)])
        floor = floors[j]
        # number of gateways below the minimum before any fix
        result["short"][j] = int((counts.sum(0) < floor).sum())
        if rule == "topup":
            counts, moved = top_up(counts, p * n_c[:, None], floor, g)
            result["moved"][j] = int(moved)
        # reject rule: give up on this draw if any gateway is short
        elif result["short"][j]:
            return result, False
        result["counts"][j] = counts
        # client[i] = gateway of the i-th record of this portion
        result["client"][j] = assign(labels, counts, g)
    return result, True


def main():
    require_project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cicids2017", "ciciot2023"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 271, 1618, 3141])
    # the five heterogeneity levels, from near-IID (100) to strongly non-IID (0.1)
    ap.add_argument("--alphas", type=float, nargs="+", default=[100, 10, 1, 0.5, 0.1])
    ap.add_argument("--clients", type=int, default=105)
    # minimum records per gateway in each portion
    ap.add_argument("--min-train", type=int, default=512)
    ap.add_argument("--min-val", type=int, default=74)
    ap.add_argument("--min-test", type=int, default=147)
    ap.add_argument("--support-rule", choices=["topup", "reject"], default="topup")
    ap.add_argument("--max-attempts", type=int, default=1000)
    # --suffix: write to alpha<A><suffix>/ instead, so a test partition never
    # replaces a main one
    ap.add_argument("--suffix", default="", help="e.g. _dev2 for a pilot partition")
    args = ap.parse_args()

    base = PROCESSED / args.dataset
    meta = json.loads((base / "arrays" / "arrays_manifest.json").read_text())
    classes = meta["classes"]
    y = np.load(base / "arrays" / "y.npy")
    # the minimum of each portion, plus the number of gateways
    floors = {0: args.min_train, 1: args.min_val, 2: args.min_test,
              "clients": args.clients}

    for seed in args.seeds:
        sdir = base / f"seed{seed}"
        split = pq.read_table(sdir / "split_indices.parquet",
                              columns=["row_id", "portion"]).to_pandas()
        # records of each portion (portion 3, unused training records, is left out)
        portion = split["portion"].to_numpy()
        rows = {j: np.flatnonzero(portion == j) for j in (0, 1, 2)}
        y_by_portion = {j: y[rows[j]].astype(np.int64) for j in (0, 1, 2)}
        for j in (0, 1, 2):                 # basic feasibility (n >= K * n_min)
            if len(rows[j]) < args.clients * floors[j]:
                raise SystemExit(f"seed {seed}: {NAMES[j]} portion has {len(rows[j])} "
                                 f"records < {args.clients} x {floors[j]}")
        for alpha in args.alphas:
            t0 = time.time()
            rejected = []
            # topup: one draw is enough; reject: redraw until no gateway is short
            for attempt in range(args.max_attempts if args.support_rule == "reject" else 1):
                result, ok = draw(y_by_portion, len(classes), alpha, seed, attempt,
                                  floors, args.support_rule)
                if ok:
                    break
                rejected.append({"attempt": attempt, "clients_below_minimum":
                                 {NAMES[j]: n for j, n in result["short"].items()}})
            else:
                raise SystemExit(
                    f"seed {seed}, alpha {alpha:g}: no accepted partition in "
                    f"{args.max_attempts} attempts under --support-rule reject")

            # gateway of every record (-1 = not given to any gateway); the asserts check
            # that the counts match and every gateway meets its minimum
            client = np.full(len(y), -1, dtype=np.int16)
            for j in (0, 1, 2):
                client[rows[j]] = result["client"][j]
                assert (np.bincount(result["client"][j], minlength=args.clients)
                        == result["counts"][j].sum(0)).all()
                assert result["counts"][j].sum(0).min() >= floors[j]
            # partition.parquet lists every train/validation/test record with its gateway
            used = portion < 3
            adir = sdir / f"alpha{alpha:g}{args.suffix}"
            adir.mkdir(exist_ok=True)
            pq.write_table(pa.table({"row_id": split["row_id"].to_numpy()[used],
                                     "portion": portion[used],
                                     "client": client[used]}),
                           adir / "partition.parquet", compression="zstd")
            np.save(adir / "dirichlet.npy", result["p"])
            # partition_summary.json: settings, top-up moves, heterogeneity measures of each
            # portion and checksums
            summary = {
                "dataset": args.dataset, "seed": seed, "alpha": alpha,
                "clients": args.clients, "classes": classes,
                "minimums": {NAMES[j]: floors[j] for j in (0, 1, 2)},
                "support_rule": args.support_rule,
                "accepted_attempt": attempt, "rejected_draws": rejected,
                "clients_below_minimum_before_top_up": {
                    NAMES[j]: result["short"][j] for j in (0, 1, 2)},
                "records_moved_by_top_up": {NAMES[j]: result["moved"].get(j, 0)
                                            for j in (0, 1, 2)},
                **{NAMES[j]: describe(result["counts"][j]) for j in (0, 1, 2)},
                "inputs": {"split_indices_sha256": sha256(sdir / "split_indices.parquet")},
                "outputs": {"partition_sha256": sha256(adir / "partition.parquet"),
                            "dirichlet_sha256": sha256(adir / "dirichlet.npy")},
                "seconds": round(time.time() - t0, 1),
            }
            (adir / "partition_summary.json").write_text(json.dumps(summary))
            t = summary["train"]
            print(f"seed {seed} alpha {alpha:g}: min train {t['min_size']}, "
                  f"mean classes {t['mean_classes_present']:.2f}, "
                  f"mean JSD {t['mean_jsd_to_global']:.3f}, "
                  f"mean entropy {t['mean_normalized_entropy']:.3f}, "
                  f"CV {t['quantity_cv']:.2f}, "
                  f"moved {summary['records_moved_by_top_up']['train']}", flush=True)
    print("PARTITION_OK")


if __name__ == "__main__":
    main()
