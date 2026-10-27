# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Figure 5.2: per-class recall heat maps (global test set).

Run from the project root:  python src/report/fig5_2_per_class_recall.py
Output: results/report/figures/fig5_2_per_class_recall.pdf and .png
"""
from figure_common import *

ROWS = [('centralized', None, 'Centralized'), ('fedavg', 100.0, r'FedAvg, $\alpha$ = 100'),
        ('fedprox', 100.0, r'FedProx, $\alpha$ = 100'), ('fedavg', 0.1, r'FedAvg, $\alpha$ = 0.1'),
        ('fedprox', 0.1, r'FedProx, $\alpha$ = 0.1'), ('local_routed', 0.1, r'Local-only, routed, $\alpha$ = 0.1'),
        ('local_tl', 0.1, r'Local-only, $T_{\mathrm{glob}}$, $\alpha$ = 0.1')]
fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.9))
for ax, (ds, title) in zip(axes, DS):
    cl = CL[ds]
    M = []
    for key, a, _ in ROWS:
        if key == 'local_routed':
            d = runs[(runs.ds == ds) & (runs.method == 'local') & (runs.view == 'routed') & (runs.alpha == a)]
        elif key == 'local_tl':
            d = runs[(runs.ds == ds) & (runs.method == 'local') & (runs.view == 'threat_landscape') & (runs.alpha == a)]
        elif key == 'centralized':
            d = runs[(runs.ds == ds) & (runs.method == 'centralized')]
        else:
            d = runs[(runs.ds == ds) & (runs.method == key) & (runs.alpha == a)]
        M.append([d['rec_' + c].mean() for c in cl])
    M = np.array(M)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.8,
                    color='white' if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(cl)))
    ax.set_xticklabels(cl)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[2] for r in ROWS])
    ax.tick_params(length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cl)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS)), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', length=0)
    ax.set_title(title, loc='left', color=INK, pad=4)
fig.subplots_adjust(left=0.235, right=0.865, top=0.95, bottom=0.07, hspace=0.42)
cax = fig.add_axes([0.885, 0.3, 0.017, 0.4])
cb = fig.colorbar(im, cax=cax)
cb.set_label('Recall (mean of five seeds)', color=MUTED)
cb.ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
cb.outline.set_visible(False)
save(fig, 'fig5_2_per_class_recall')
