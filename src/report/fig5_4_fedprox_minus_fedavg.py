# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Figure 5.4: paired FedProx - FedAvg differences in macro-F1 and accuracy for every seed.

Run from the project root:  python src/report/fig5_4_fedprox_minus_fedavg.py
Output: results/report/figures/fig5_4_fedprox_minus_fedavg.pdf and .png
"""
from figure_common import *

P = runs[runs.method.isin(['fedavg', 'fedprox'])].pivot_table(index=['ds', 'seed', 'alpha'], columns='method',
                                                              values=['acc', 'f1']).reset_index()
fig, axes = plt.subplots(2, 2, figsize=(6.3, 3.9), sharex=True, sharey='row')
rng = np.random.default_rng(0)
jit = {s: v for s, v in zip(SEED, np.linspace(-0.16, 0.16, 5))}
for c, (ds, title) in enumerate(DS):
    d = P[P.ds == ds]
    for r, (met, ylab, scale) in enumerate([('f1', 'FedProx $-$ FedAvg\nmacro-F1', 1),
                                            ('acc', 'FedProx $-$ FedAvg\naccuracy (pp)', 100)]):
        ax = axes[r, c]
        ax.axhline(0, color=MUTED, lw=0.8)
        means = []
        for i, a in enumerate(A):
            v = d[d.alpha == a]
            diff = (v[(met, 'fedprox')] - v[(met, 'fedavg')]) * scale
            xs = [i + jit[s] for s in v.seed]
            ax.scatter(xs, diff, s=16, color=COL['fedprox'], edgecolor='white', linewidth=0.5, zorder=3,
                       label='One master seed' if i == 0 else None)
            means.append(diff.mean())
            ax.plot([i - 0.25, i + 0.25], [diff.mean()] * 2, color=INK, lw=1.6, zorder=4,
                    label='Mean of five seeds' if i == 0 else None)
        style(ax)
        if r == 0:
            ax.set_title(title, loc='left', color=INK, pad=4)
        if c == 0:
            ax.set_ylabel(ylab, color=MUTED)
        if r == 1:
            ax.set_xticks(X)
            ax.set_xticklabels(XL)
            ax.set_xlabel(r'Dirichlet $\alpha$', color=MUTED)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.8, w_pad=1.5)
save(fig, 'fig5_4_fedprox_minus_fedavg')
