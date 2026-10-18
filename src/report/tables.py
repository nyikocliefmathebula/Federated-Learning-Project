# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""LaTeX table bodies for Chapter 5, generated from results/report/csv/ (no hand-typed numbers)."""
import numpy as np
import pandas as pd

from report_paths import CSV as AN, TABLES as T
import os
os.makedirs(T, exist_ok=True)
A = [100.0, 10.0, 1.0, 0.5, 0.1]
runs = pd.read_csv(f'{AN}/runs.csv')


def pm(m, s, d):
    return f'{m:.{d}f} $\\pm$ {s:.{d}f}'


def sel(ds, key):
    if key == 'local_tl':
        return runs[(runs.ds == ds) & (runs.method == 'local') & (runs.view == 'threat_landscape')]
    if key == 'local_routed':
        return runs[(runs.ds == ds) & (runs.method == 'local') & (runs.view == 'routed')]
    return runs[(runs.ds == ds) & (runs.method == key)]


NAMES = {'fedavg': '\\fedavg{}', 'fedprox': '\\fedprox{}', 'local_tl': 'Local-only on $T_{\\mathrm{glob}}$$^{a}$',
         'local_routed': 'Local-only, routed'}

# ---------------- Tables 5.1 / 5.2: global test results
for ds, dec_acc, dec_fpr in [('cicids2017', 3, 3), ('ciciot2023', 2, 2)]:
    lines = []
    for met, title, scale, d in [('acc', 'Accuracy (\\%)', 100, dec_acc), ('f1', 'Macro-F1', 1, 3),
                                 ('fpr', 'False-positive rate (\\%)', 100, dec_fpr)]:
        lines.append(f'\\ThesisGroup{{6}}{{{title}}} \\\\ \\hline')
        c = sel(ds, 'centralized')[met] * scale
        lines.append(f'Centralized & \\multicolumn{{5}}{{c|}}{{{pm(c.mean(), c.std(), d)} (one model per seed; independent of $\\alpha$)}} \\\\ \\hline')
        for key in ['fedavg', 'fedprox', 'local_tl', 'local_routed']:
            g = sel(ds, key).groupby('alpha')[met].agg(['mean', 'std']).loc[A] * scale
            cells = ' & '.join(pm(g.loc[a, 'mean'], g.loc[a, 'std'], d) for a in A)
            lines.append(f'{NAMES[key]} & {cells} \\\\ \\hline')
    open(f'{T}/t_global_{ds}.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------- Table 5.3: paired differences in macro-F1
P = pd.read_csv(f'{AN}/paired.csv')
lines = []
for a in A:
    cells = []
    for ds in ['cicids2017', 'ciciot2023']:
        for pair in ['FedProx-FedAvg', 'FedAvg-Cen', 'FedProx-Cen']:
            q = P[(P.ds == ds) & (P.alpha == a) & (P.pair == pair) & (P.met == 'f1')].iloc[0]
            cells.append(f'{q["mean"]:+.3f} ({q.wins}/5)')
    lines.append(f'{a:g} & ' + ' & '.join(cells) + ' \\\\ \\hline')
open(f'{T}/t_paired.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------- Table 5.4: gateway-matched mean and worst-gateway accuracy
G = pd.read_csv(f'{AN}/gateway_matched.csv')
PG = G.pivot_table(index=['ds', 'seed', 'alpha', 'gw'], columns='method', values='acc').reset_index()
lines = []
for ds, name in [('cicids2017', '\\cicids{}'), ('ciciot2023', '\\ciciot{}')]:
    lines.append(f'\\ThesisGroup{{5}}{{{name}}} \\\\ \\hline')
    for a in [100.0, 1.0, 0.1]:
        d = PG[(PG.ds == ds) & (PG.alpha == a)]
        cells = []
        for k in ['local', 'centralized', 'fedavg', 'fedprox']:
            mean = d.groupby('seed')[k].mean().mean() * 100
            worst = d.groupby('seed')[k].min().mean() * 100
            cells.append(f'{mean:.1f} ({worst:.1f})')
        lines.append(f'{a:g} & ' + ' & '.join(cells) + ' \\\\ \\hline')
open(f'{T}/t_gateway.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------- Table 5.5: convergence and communication
C = pd.read_csv(f'{AN}/convergence.csv')
lines = []
for ds, name in [('cicids2017', '\\cicids{}'), ('ciciot2023', '\\ciciot{}')]:
    lines.append(f'\\ThesisGroup{{8}}{{{name}}} \\\\ \\hline')
    for k in ['fedavg', 'fedprox']:
        for a in [100.0, 1.0, 0.1]:
            d = C[(C.ds == ds) & (C.method == k) & (C.alpha == a)]
            n95 = d.r95.notna().sum()
            med95 = f'{d.mb95.median():.0f}' if n95 else '--'
            n90 = d.r90.notna().sum()
            med90 = f'{d.mb90.median():.0f}' if n90 else '--'
            lines.append(f'{NAMES[k]} & {a:g} & {d.best.mean():.0f} $\\pm$ {d.best.std():.0f} & {d.stop.mean():.0f} & '
                         f'{int(d.hit_budget.sum())}/5 & {d.mb_best.mean():.0f} & {n90}/5 ({med90}) & {n95}/5 ({med95}) \\\\ \\hline')
open(f'{T}/t_convergence.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------- Table 5.6: stopping-rule sensitivity (CIC-IDS2017)
s = pd.read_csv(f'{AN}/stopping_sensitivity.csv')
s['df1'] = s.f1_new - s.f1_old
lines = []
for k, name in [('centralized', 'Centralized'), ('local', 'Local-only (routed)'), ('fedavg', '\\fedavg{}'),
                ('fedprox', '\\fedprox{}')]:
    g = s[s.method == k]
    up, down = int((g.df1 > 1e-12).sum()), int((g.df1 < -1e-12).sum())
    same = len(g) - up - down
    unit = 'epoch' if k in ('centralized', 'local') else 'round'
    lines.append(f'{name} & {len(g)} & {g.f1_old.mean():.3f} & {g.f1_new.mean():.3f} & {up} / {same} / {down} & '
                 f'{g.best_old.mean():.0f} $\\rightarrow$ {g.best_new.mean():.0f} \\\\ \\hline')
open(f'{T}/t_stopping.tex', 'w').write('\n'.join(lines) + '\n')
for f in sorted(os.listdir(T)):
    print('=====', f)
    print(open(f'{T}/{f}').read())
