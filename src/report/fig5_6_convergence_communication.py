# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Figure 5.6: validation macro-F1 against cumulative model transfers, seed 42.

Run from the project root:  python src/report/fig5_6_convergence_communication.py
Output: results/report/figures/fig5_6_convergence_communication.pdf and .png
"""
from figure_common import *

fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.9), sharex='row', sharey='row')
for r, (ds, title) in enumerate(DS):
    cen = json.load(open(f'{R}/{ds}/centralized/seed42/metrics.json'))['best_val_macro_f1']
    for c, a in enumerate([100.0, 1.0, 0.1]):
        ax = axes[r, c]
        ax.axhline(cen, color=COL['centralized'], ls='--', lw=1.1, label='Centralized best')
        ax.axhline(0.9 * cen, color=COL['centralized'], ls=':', lw=1.0, label='90% of centralized best')
        for k in ['fedavg', 'fedprox']:
            t = pd.read_csv(f'{R}/{ds}/{k}/seed42/alpha{a:g}/trajectory.csv')
            led = pd.read_csv(f'{R}/{ds}/{k}/seed42/alpha{a:g}/communication.csv')
            cum = led.groupby('round').serialized_bytes.sum().cumsum() / 1e6
            mb = np.r_[0.0, cum.reindex(t['round'].iloc[1:]).values]
            ax.plot(mb, t.val_macro_f1, color=COL[k], lw=1.1, label=LAB[k])
            b = int(t.best_round.iloc[-1])
            ax.plot(mb[b], t.val_macro_f1.iloc[b], marker=MARK[k], color=COL[k], ms=4.5, mec='white', mew=0.6)
        style(ax)
        ax.set_ylim(0, 1.02 if ds == 'cicids2017' else 0.75)
        ax.set_title(f'{title}, ' + r'$\alpha$' + f' = {a:g}', loc='left', color=INK, pad=3, fontsize=7.5)
        if c == 0:
            ax.set_ylabel('Validation macro-F1', color=MUTED)
        if r == 1:
            ax.set_xlabel('Cumulative model transfers (MB)', color=MUTED)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.9, w_pad=0.8)
save(fig, 'fig5_6_convergence_communication')
