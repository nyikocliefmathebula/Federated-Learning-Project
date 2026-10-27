# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Figure 5.1: global test accuracy and macro-F1 against alpha, both datasets.

Run from the project root:  python src/report/fig5_1_global_test_performance.py
Output: results/report/figures/fig5_1_global_test_performance.pdf and .png
"""
from figure_common import *

fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6), sharex=True)
for c, (ds, title) in enumerate(DS):
    for r, (met, ylab, scale, lim) in enumerate([('acc', 'Accuracy (%)', 100, (40, 101)),
                                                 ('f1', 'Macro-F1', 1, (0, 1.02))]):
        ax = axes[r, c]
        cen = runs[(runs.ds == ds) & (runs.method == 'centralized')][met] * scale
        ax.axhspan(cen.mean() - cen.std(), cen.mean() + cen.std(), color=COL['centralized'], alpha=0.12, lw=0)
        ax.axhline(cen.mean(), color=COL['centralized'], ls='--', lw=1.2, label=LAB['centralized'])
        for i, k in enumerate(['fedavg', 'fedprox', 'local_tl', 'local_routed']):
            g = series(ds, k, met) * scale
            off = (i - 1.5) * 0.06
            ax.errorbar(X + off, g['mean'], yerr=g['std'], color=COL[k], marker=MARK[k], ms=4.2, lw=1.4,
                        elinewidth=0.9, capsize=2, mec='white', mew=0.6, label=LAB[k])
        ax.set_ylim(*lim)
        style(ax)
        if r == 0:
            ax.set_title(title, loc='left', color=INK, pad=4)
        if c == 0:
            ax.set_ylabel(ylab, color=MUTED)
        if r == 1:
            ax.set_xticks(X)
            ax.set_xticklabels(XL)
            ax.set_xlabel(r'Dirichlet $\alpha$ (label skew increases to the right)', color=MUTED)
h, l = axes[0, 0].get_legend_handles_labels()
order = [0, 3, 1, 4, 2]   # row-wise: centralized, FedAvg, FedProx / local views
fig.legend([h[i] for i in order], [l[i] for i in order], loc='upper center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.8, w_pad=1.5)
save(fig, 'fig5_1_global_test_performance')
