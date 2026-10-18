# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Supporting numbers for the Chapter 5 text (all computed from the exported run files)."""
import json, glob, os
import numpy as np
import pandas as pd

from report_paths import RUNS as R, CSV as O
A = [100.0, 10.0, 1.0, 0.5, 0.1]
SEEDS = [13, 42, 271, 1618, 3141]


def m(ds, meth, seed, a=None):
    p = f'{R}/{ds}/{meth}/seed{seed}' + ('' if a is None else f'/alpha{a:g}')
    return json.load(open(f'{p}/metrics.json'))


print('=== A. CIC-IDS2017 macro-F1 without WebAttack (class 5), FedProx-FedAvg paired')
for a in A:
    d6, wf = [], []
    for s in SEEDS:
        f = {k: np.array(m('cicids2017', k, s, a)['global_test']['f1']) for k in ['fedavg', 'fedprox']}
        d6.append(np.delete(f['fedprox'], 5).mean() - np.delete(f['fedavg'], 5).mean())
        wf.append(f['fedprox'][5])
    print(f'a={a:g}: dF1 excl WebAttack mean {np.mean(d6):+.4f} (min {min(d6):+.4f} max {max(d6):+.4f}); FedProx WebAttack F1 {np.round(wf,3)}')

print('=== B. CIC-IoT2023 centralized confusion summed over seeds')
C = np.load(f'{O}/confusions.npy', allow_pickle=True).item()
cls = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web', 'BruteForce']
for key in ['ciciot2023|centralized|None', 'ciciot2023|fedavg|100.0', 'ciciot2023|fedprox|0.1', 'ciciot2023|fedavg|0.1']:
    M = C[key].astype(float)
    err = M.sum() - np.trace(M)
    dd = M[1, 2] + M[2, 1]
    merged = (np.trace(M) + dd) / M.sum()
    att = M[1:, 1:].sum() / M[1:, :].sum()          # attack predicted as any attack
    fp = M[0, 1:].sum() / M[0].sum()
    prec_alarm = M[1:, 1:].sum() / M[:, 1:].sum()
    print(f'{key}: acc {np.trace(M)/M.sum():.4f}; DoS<->DDoS share of errors {dd/err:.3f}; merged acc {merged:.4f}; '
          f'attack detection {att:.4f}; FPR {fp:.4f}; benign->Recon {M[0,4]/M[0].sum():.4f}; alarm precision {prec_alarm:.4f}; '
          f'attacks predicted benign {M[1:,0].sum()/M[1:].sum():.4f}')

print('=== C. CIC-IDS2017 binary view')
for key in ['cicids2017|centralized|None', 'cicids2017|fedavg|100.0', 'cicids2017|fedprox|100.0', 'cicids2017|fedavg|0.1',
            'cicids2017|fedprox|0.1', 'cicids2017|local_routed|0.1']:
    M = C[key].astype(float)
    print(f'{key}: attack detection {M[1:,1:].sum()/M[1:].sum():.4f}; FPR {M[0,1:].sum()/M[0].sum():.5f}; '
          f'alarm precision {M[1:,1:].sum()/M[:,1:].sum():.4f}; false alarms per seed {M[0,1:].sum()/5:.0f}')

print('=== D. benign test counts, seed 42')
for ds in ['cicids2017', 'ciciot2023']:
    g = m(ds, 'centralized', 42)['confusion']
    g = np.array(g)
    print(ds, 'test n', g.sum(), 'benign', g[0].sum(), f'share {g[0].sum()/g.sum():.4f}')

print('=== E. gateway-level shares (pooled over seeds, gateway counted once per seed)')
G = pd.read_csv(f'{O}/gateway_matched.csv')
P = G.pivot_table(index=['ds', 'seed', 'alpha', 'gw'], columns='method', values='acc').reset_index()
X = pd.read_csv(f'{O}/cross_pergw.csv')
P = P.merge(X[['ds', 'seed', 'alpha', 'gw', 'borrowed_acc']], how='left')
for ds in ['cicids2017', 'ciciot2023']:
    for a in A:
        d = P[(P.ds == ds) & (P.alpha == a)]
        s = ' '.join(f'{k}>=own {100*(d[k] >= d.local - 1e-12).mean():.1f}' for k in ['centralized', 'fedavg', 'fedprox'])
        if d.borrowed_acc.notna().any():
            s += ' | ' + ' '.join(f'{k}>borrowed {100*(d[k] > d.borrowed_acc).mean():.1f}' for k in ['centralized', 'fedavg', 'fedprox'])
            s += f' | own-borrowed mean {100*(d.local-d.borrowed_acc).mean():.1f}pp'
        print(ds, a, s)

print('=== F. worst federated gateways, CIC-IDS2017 alpha 0.1')
for meth in ['fedavg', 'fedprox']:
    for s in SEEDS:
        cc = pd.read_parquet(f'{R}/cicids2017/{meth}/seed{s}/alpha0.1/confusion_counts.parquet')
        cc = cc[cc.view == 'gateway_matched']
        acc = cc.assign(ok=(cc.true == cc.pred) * cc['count']).groupby('client')[['ok', 'count']].sum()
        acc['acc'] = acc.ok / acc['count']
        w = acc.acc.idxmin()
        comp = cc[cc.client == w].groupby('true')['count'].sum()
        comp = (comp / comp.sum()).round(3).to_dict()
        print(meth, s, 'worst gw', w, f'acc {acc.acc.min():.3f} n {acc.loc[w,"count"]}', 'true-class shares', comp)

print('=== G. run time (s) mean by ds, method')
r = pd.read_csv(f'{O}/runs.csv')
print(r[r.view != 'routed'].groupby(['ds', 'method']).seconds.agg(['mean', 'sum']).round(0))

print('=== H. hit budget by alpha (federated)')
cv = pd.read_csv(f'{O}/convergence.csv')
print(cv[cv.method.isin(['fedavg', 'fedprox'])].groupby(['ds', 'method', 'alpha'])[['hit_budget', 'best', 'stop', 'mb_stop']].agg(['sum', 'mean']).round(1))
print(cv[cv.method.isin(['fedavg', 'fedprox'])].groupby(['ds', 'method', 'alpha']).apply(lambda d: f'r90 {d.r90.notna().sum()}/5 med {d.r90.median()} mb {d.mb90.median():.0f}; r95 {d.r95.notna().sum()}/5 med {d.r95.median()} mb {d.mb95.median():.0f}'))
print(cv[cv.method == 'local'].groupby(['ds', 'alpha'])[['best_mean', 'stop_mean', 'stop_max', 'hit_budget']].mean().round(1))
cen = r[(r.method == 'centralized')]
print(cen.groupby('ds')[['best', 'stop', 'best_val']].agg(['mean', 'min', 'max']))

print('=== J. paired, all pairs')
pp = pd.read_csv(f'{O}/paired.csv')
pp['s'] = pp.apply(lambda q: f'{q["mean"]:+.4f}({int(q.wins)}/5,sd {q.sd:.4f}, {q.mn:+.4f}..{q.mx:+.4f})', axis=1)
print(pp.pivot_table(index=['ds', 'pair', 'met'], columns='alpha', values='s', aggfunc='first').to_string())
