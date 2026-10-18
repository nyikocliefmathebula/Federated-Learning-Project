# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Chapter 5 figures, drawn only from the saved run files (via results/report/csv/)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
    'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 7.2, 'axes.linewidth': 0.6,
})
from report_paths import FIGURES as OUT, CSV as AN, RUNS as R
INK, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
COL = {'centralized': '#52514e', 'fedavg': '#2a78d6', 'fedprox': '#eb6834',
       'local_tl': '#1baf7a', 'local_routed': '#eda100'}
MARK = {'fedavg': 'o', 'fedprox': 's', 'local_tl': '^', 'local_routed': 'D'}
LAB = {'centralized': 'Centralized', 'fedavg': 'FedAvg', 'fedprox': 'FedProx',
       'local_tl': r'Local-only, gateway models on $T_{\mathrm{glob}}$',
       'local_routed': 'Local-only, routed deployment'}
A = [100.0, 10.0, 1.0, 0.5, 0.1]
X = np.arange(len(A))
XL = ['100', '10', '1', '0.5', '0.1']
DS = [('cicids2017', 'CIC-IDS2017'), ('ciciot2023', 'CIC-IoT2023')]
SEED = [13, 42, 271, 1618, 3141]

runs = pd.read_csv(f'{AN}/runs.csv')


def style(ax, grid_y=True):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=2.5, width=0.6)
    if grid_y:
        ax.grid(axis='y', color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def series(ds, key, met):
    if key == 'local_tl':
        d = runs[(runs.ds == ds) & (runs.method == 'local') & (runs.view == 'threat_landscape')]
    elif key == 'local_routed':
        d = runs[(runs.ds == ds) & (runs.method == 'local') & (runs.view == 'routed')]
    else:
        d = runs[(runs.ds == ds) & (runs.method == key)]
    g = d.groupby('alpha')[met].agg(['mean', 'std'])
    return g.loc[A]


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf')
    fig.savefig(f'{OUT}/{name}.png', dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5.1
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

# ---------------------------------------------------------------- Figure 5.4 (FedProx - FedAvg per seed)
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

# ---------------------------------------------------------------- Figure 5.2 (per-class recall)
CL = {'cicids2017': ['Benign', 'DoS', 'DDoS', 'PortScan', 'BruteForce', 'WebAttack', 'Botnet'],
      'ciciot2023': ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web', 'BruteForce']}
ROWS = [('centralized', None, 'Centralized'), ('fedavg', 100.0, r'FedAvg, $\alpha$ = 100'),
        ('fedprox', 100.0, r'FedProx, $\alpha$ = 100'), ('fedavg', 0.1, r'FedAvg, $\alpha$ = 0.1'),
        ('fedprox', 0.1, r'FedProx, $\alpha$ = 0.1'), ('local_routed', 0.1, r'Local-only, routed, $\alpha$ = 0.1'),
        ('local_tl', 0.1, r'Local-only, $T_{\mathrm{glob}}$, $\alpha$ = 0.1')]
cmap = LinearSegmentedColormap.from_list('blue', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
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

# ---------------------------------------------------------------- Figure 5.5 (gateway views)
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

# ---------------------------------------------------------------- Figure 5.6 (convergence vs bytes)
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

# ---------------------------------------------------------------- Figure 5.3 (CIC-IoT2023 confusion matrix)
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
print('figures written')
