# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Figure 5.3: confusion matrix of the centralized CIC-IoT2023 model.

Run from the project root:  python src/report/fig5_3_ciciot2023_confusion_centralized.py
Output: results/report/figures/fig5_3_ciciot2023_confusion_centralized.pdf and .png
"""
from figure_common import *

cms = np.load(f'{AN}/confusions.npy', allow_pickle=True).item()
cm = cms['ciciot2023|centralized|None'].astype(float)
cmn = cm / cm.sum(1, keepdims=True) * 100
cl = CL['ciciot2023']
fig, ax = plt.subplots(figsize=(4.3, 3.5))
im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=100)
for i in range(cmn.shape[0]):
    for j in range(cmn.shape[1]):
        v = cmn[i, j]
        if v >= 0.5:
            ax.text(j, i, f'{v:.0f}' if v >= 9.5 else f'{v:.1f}', ha='center', va='center', fontsize=6.6,
                    color='white' if v > 55 else INK)
ax.set_xticks(range(len(cl)))
ax.set_xticklabels(cl, rotation=35, ha='right')
ax.set_yticks(range(len(cl)))
ax.set_yticklabels(cl)
ax.set_xlabel('Predicted class', color=MUTED)
ax.set_ylabel('True class', color=MUTED)
ax.tick_params(length=0, colors=MUTED)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-0.5, len(cl)), minor=True)
ax.set_yticks(np.arange(-0.5, len(cl)), minor=True)
ax.grid(which='minor', color='white', linewidth=1.2)
ax.tick_params(which='minor', length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label('Share of the true class (%)', color=MUTED)
cb.ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
cb.outline.set_visible(False)
fig.tight_layout()
save(fig, 'fig5_3_ciciot2023_confusion_centralized')
