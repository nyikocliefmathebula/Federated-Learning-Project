# Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town 
# 2026

"""Figure: client label skew in the measured gateway partitions.

Run from the project root (/mnt/c/Users/Nyiko/fl-iot-ids) with .venv active,
after partition_clients.py has finished for the dataset:
    python src/plot_label_skew.py --dataset cicids2017
    python src/plot_label_skew.py --dataset ciciot2023

For one dataset, seed and portion (default: seed 42, training portion) it draws
one panel per alpha (default 100, 10, 1, 0.5, 0.1). In each panel every bar is
one of the 105 gateways and shows the share of each class in that gateway's
records; gateways keep the same order (0 to 104) in every panel. The bar marked
"All" shows the class mix of the whole portion. With --counts the bars show
record counts instead of shares (quantity skew).

Only partition_summary.json (one per alpha) and split_summary.json are read.
Nothing in data/ is changed. Outputs in results/figures/:
    fig3_2_label_skew_<dataset>_seed<S>_<portion>.png   300 dpi, for Word
    fig3_2_label_skew_<dataset>_seed<S>_<portion>.pdf   vector, for LaTeX
    fig3_2_label_skew_<dataset>_seed<S>_<portion>.csv   the plotted numbers
A partition suffix, a different list of alphas (e.g. _alpha100-1-0.1) and
_counts are added to the names, so variants never replace the main figure.
"""
# How it works, in short:
#   1. read the class-by-gateway count table of each alpha (written by
#      partition_clients.py) and check it against the split's class totals
#   2. turn counts into shares (each gateway's bar adds up to 100 %)
#   3. draw one stacked-bar panel per alpha, save PNG, PDF and a CSV table
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")              # draw into files only; no window is opened
import matplotlib.pyplot as plt    # noqa: E402
plt.rcParams["pdf.fonttype"] = 42  # keep the text in the PDF editable
import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402

from schema import PROCESSED, require_project_root  # noqa: E402

# Names used in the figure for the dataset folders
DISPLAY = {"cicids2017": "CIC-IDS2017", "ciciot2023": "CIC-IoT2023"}
DEFAULT_ALPHAS = [100, 10, 1, 0.5, 0.1]
# One colour per class, in class order (Benign first). These are eight fixed,
# colour-blind-checked hues; class i always gets colour i in every panel.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"   # text, labels, light lines
PORTION_KEY = {"train": "train_by_class", "validation": "validation_by_class",
               "test": "test_by_class"}


def load_counts(base, seed, alpha, suffix, portion):
    """Class names and the [classes x gateways] count table of one partition."""
    folder = base / f"seed{seed}" / f"alpha{alpha:g}{suffix}"
    path = folder / "partition_summary.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run partition_clients.py for this dataset "
                         "first.")
    summary = json.loads(path.read_text())
    counts = np.array(summary[portion]["class_by_client"], dtype=np.int64)
    return summary["classes"], counts, summary


def main():
    require_project_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["cicids2017", "ciciot2023"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--portion", choices=["train", "validation", "test"], default="train")
    ap.add_argument("--suffix", default="", help="partition suffix, e.g. _dev2")
    ap.add_argument("--counts", action="store_true",
                    help="show record counts instead of class shares")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    base = PROCESSED / args.dataset
    split = json.loads((base / f"seed{args.seed}" / "split_summary.json").read_text())

    # ---- 1. read every alpha's counts and check them
    panels, classes = [], None
    for alpha in args.alphas:
        names, counts, summary = load_counts(base, args.seed, alpha, args.suffix,
                                             args.portion)
        if classes is None:
            classes = names
        elif names != classes:
            raise SystemExit(f"alpha {alpha:g}: class order differs from the first panel")
        # every record of the portion must be in exactly one gateway: the column
        # sums per class must equal the class totals recorded by split_data.py
        # (the pilot partitions made with --suffix are skipped by this check)
        totals = split[PORTION_KEY[args.portion]]
        if not args.suffix and counts.sum(1).tolist() != [totals[c] for c in classes]:
            raise SystemExit(f"alpha {alpha:g}: counts do not add up to the "
                             f"{args.portion} class totals in split_summary.json")
        panels.append((alpha, counts, summary[args.portion]))
    n_classes, n_clients = panels[0][1].shape
    if n_classes > len(COLOURS):
        raise SystemExit(f"{n_classes} classes but only {len(COLOURS)} colours")

    # ---- 2. the numbers to plot, also saved as a table (CSV)
    rows = []
    for alpha, counts, _ in panels:
        sizes = counts.sum(0)                       # records per gateway
        shares = counts / np.maximum(sizes, 1)      # each gateway's class shares
        for k in range(n_clients):
            for c, name in enumerate(classes):
                rows.append({"alpha": alpha, "client": k, "class": name,
                             "records": int(counts[c, k]),
                             "share": round(float(shares[c, k]), 6)})
    table = pd.DataFrame(rows)

    # ---- 3. draw: one row of axes (panel) per alpha, the same gateway order in all
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(6.3, 0.35 + 1.2 * n), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    x = np.arange(n_clients)                        # gateway 0 ... 104
    x_all = -3.0                                    # position of the "All" bar
    for ax, (alpha, counts, stats) in zip(axes, panels):
        sizes = counts.sum(0)
        values = counts / np.maximum(sizes, 1) * 100 if not args.counts else counts / 1000
        bottom = np.zeros(n_clients)
        for c in range(n_classes):                  # stack the classes, Benign at the bottom
            # a hairline of white between stacked classes keeps neighbours apart
            ax.bar(x, values[c], bottom=bottom, width=0.8, color=COLOURS[c],
                   edgecolor="white", linewidth=0.15, label=classes[c])
            bottom += values[c]
        if not args.counts:
            # reference bar: class mix of the whole portion
            mix = counts.sum(1) / counts.sum() * 100
            b = 0.0
            for c in range(n_classes):
                ax.bar(x_all, mix[c], bottom=b, width=1.6, color=COLOURS[c],
                       edgecolor="white", linewidth=0.15)
                b += mix[c]
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 50, 100])
            ax.set_yticklabels(["0", "50", "100%"])
        else:
            ax.set_ylim(0, None)
        # one line of measured heterogeneity above each panel (from the summary)
        ax.set_title(f"α = {alpha:g}    mean classes present "
                     f"{stats['mean_classes_present']:.2f} of {n_classes}    "
                     f"mean JSD {stats['mean_jsd_to_global']:.3f}    "
                     f"size CV {stats['quantity_cv']:.2f}",
                     loc="left", fontsize=7.5, color=INK, pad=3)
        ax.tick_params(axis="both", labelsize=7, colors=MUTED, length=2)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
            ax.spines[side].set_linewidth(0.6)
    ticks = [t for t in (0, 20, 40, 60, 80, 100) if t < n_clients]
    labels = [str(t) for t in ticks]
    if not args.counts:                             # the "All" bar left of gateway 0
        ticks, labels = [x_all] + ticks, ["All"] + labels
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlim((x_all - 1.5) if not args.counts else -1, n_clients - 0.3)
    axes[-1].set_xlabel("Gateway (client) k", fontsize=7.5, color=MUTED)
    ylabel = ("Share of records" if not args.counts else "Records, thousands")
    fig.text(0.012, 0.5, f"{ylabel} ({args.portion} portion)", rotation=90,
             va="center", fontsize=7.5, color=MUTED)
    # legend above the first panel: one swatch and name per class
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_classes, fontsize=7,
               frameon=False, handlelength=1.0, handleheight=0.8, columnspacing=1.1,
               bbox_to_anchor=(0.53, 1.0))
    fig.tight_layout(rect=(0.03, 0, 1, 1 - 0.2 / fig.get_figheight()), h_pad=0.6)

    # ---- 4. save
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # other alpha lists get their own file names, so the default figure is kept
    alphas = ("" if args.alphas == DEFAULT_ALPHAS
              else "_alpha" + "-".join(f"{a:g}" for a in args.alphas))
    stem = (f"fig3_2_label_skew_{args.dataset}_seed{args.seed}_{args.portion}"
            + args.suffix + alphas + ("_counts" if args.counts else ""))
    fig.savefig(out / f"{stem}.png", dpi=300)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    table.to_csv(out / f"{stem}.csv", index=False)
    print(f"{DISPLAY[args.dataset]}, seed {args.seed}, {args.portion} portion, "
          f"{n_clients} gateways, classes: {', '.join(classes)}")
    for alpha, counts, stats in panels:
        print(f"  alpha {alpha:g}: mean classes present {stats['mean_classes_present']:.2f}, "
              f"mean JSD {stats['mean_jsd_to_global']:.3f}, "
              f"size CV {stats['quantity_cv']:.2f}, "
              f"gateway sizes {int(counts.sum(0).min())}-{int(counts.sum(0).max())}")
    for ext in ("png", "pdf", "csv"):
        print(f"Written {out / (stem + '.' + ext)}")
    print("FIGURE_OK")


if __name__ == "__main__":
    main()
