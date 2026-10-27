# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Figure 5.5: share of gateways where the shared model is at least as accurate as the own or borrowed local models.

Run from the project root:  python src/report/fig5_5_gateway_level_comparison.py
Output: results/report/figures/fig5_5_gateway_level_comparison.pdf and .png
"""
from figure_common import *

G = pd.read_csv(f'{AN}/gateway_matched.csv')
PG = G.pivot_table(index=['ds', 'seed', 'alpha', 'gw'], columns='method', values='acc').reset_index()
C = pd.read_csv(f'{AN}/cross_pergw.csv')
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
for ax, (ds, title) in zip(axes, DS):
    d = PG[PG.ds == ds]
    for k in ['centralized', 'fedavg', 'fedprox']:
        w = d.assign(b=d[k] >= d.local).groupby(['alpha', 'seed']).b.mean().groupby('alpha').mean().loc[A] * 100
        ax.plot(X, w.values, color=COL[k], marker=MARK.get(k, 'v'), ms=4.2, lw=1.4, mec='white', mew=0.6,
                label=f'{LAB[k]} vs own local model')
    x2 = d.merge(C, on=['ds', 'seed', 'alpha', 'gw'])
    if len(x2):
        for k in ['centralized', 'fedavg', 'fedprox']:
            w = x2.assign(b=x2[k] > x2.borrowed_acc).groupby(['alpha', 'seed']).b.mean().groupby('alpha').mean().loc[A] * 100
            ax.plot(X, w.values, color=COL[k], marker=MARK.get(k, 'v'), ms=3.6, lw=1.1, ls=':', mfc='white',
                    label=f'{LAB[k]} vs borrowed local models')
    ax.set_ylim(0, 103)
    ax.set_xticks(X)
    ax.set_xticklabels(XL)
    ax.set_xlabel(r'Dirichlet $\alpha$', color=MUTED)
    style(ax)
    ax.set_title(title, loc='left', color=INK, pad=4)
axes[0].set_ylabel('Gateways where the shared\nmodel is at least as accurate (%)', color=MUTED)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=6.8)
fig.tight_layout(rect=(0, 0, 1, 0.76), w_pad=1.5)
save(fig, 'fig5_5_gateway_level_comparison')
