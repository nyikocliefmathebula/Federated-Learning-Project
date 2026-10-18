# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Unified analysis of the final grid (both datasets) for Chapter 5.

Reads only saved run files exported from the laptop (results/runs, results/archive).
Writes CSV tables to results/report/csv/.
"""
import json, glob, os
import numpy as np
import pandas as pd

from report_paths import RUNS as R, ARCHIVE as ARCH, CSV as OUT
os.makedirs(OUT, exist_ok=True)
S = [13, 42, 271, 1618, 3141]
A = [100.0, 10.0, 1.0, 0.5, 0.1]
CLASSES = {
    'cicids2017': ['Benign', 'DoS', 'DDoS', 'PortScan', 'BruteForce', 'WebAttack', 'Botnet'],
    'ciciot2023': ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web', 'BruteForce'],
}
BYTES = {'cicids2017': None, 'ciciot2023': None}   # serialized bytes per message, read from ledgers


def jl(p):
    return json.load(open(p))


def run_dir(base, ds, m, s, a=None):
    d = f'{base}/{ds}/{m}/seed{s}'
    return d if a is None else f'{d}/alpha{a:g}'


def metrics_cm(cm):
    cm = np.asarray(cm, float)
    tp = np.diag(cm); sup = cm.sum(1); pred = cm.sum(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(pred > 0, tp / pred, 0.0)
        rec = np.where(sup > 0, tp / sup, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    ben = cm[0].sum()
    return dict(acc=tp.sum() / cm.sum(), f1=f1.mean(), fpr=(ben - cm[0, 0]) / ben if ben else np.nan,
                recall=rec)


rows = []          # one row per (dataset, method, seed, alpha) and view
gm_rows = []       # gateway-matched per gateway
conv_rows = []     # convergence / communication per federated run
for ds, CL in CLASSES.items():
    cen_best_val = {}
    for s in S:
        m = jl(run_dir(R, ds, 'centralized', s) + '/metrics.json')
        g = m['global_test']
        cen_best_val[s] = m['best_val_macro_f1']
        r = dict(ds=ds, method='centralized', view='global', seed=s, alpha=np.nan, acc=g['accuracy'],
                 f1=g['macro_f1'], fpr=g['fpr'], best=m['best_epoch'], stop=m['stopped_epoch'],
                 best_val=m['best_val_macro_f1'], seconds=m['seconds'])
        for c, v in zip(CL, g['recall']):
            r['rec_' + c] = v
        rows.append(r)
        for a in A:
            for x in m['gateway_matched'][f'alpha{a:g}']:
                gm_rows.append(dict(ds=ds, seed=s, alpha=a, gw=x['client'], method='centralized',
                                    acc=x['accuracy'], f1p=x['macro_f1_present'], fpr=x['fpr'], n=x['n']))
    for s in S:
        for a in A:
            # ---------------- local-only
            d = run_dir(R, ds, 'local', s, a)
            m = jl(d + '/metrics.json')
            g = m['routed']
            r = dict(ds=ds, method='local', view='routed', seed=s, alpha=a, acc=g['accuracy'], f1=g['macro_f1'],
                     fpr=g['fpr'], seconds=m['seconds'])
            for c, v in zip(CL, g['recall']):
                r['rec_' + c] = v
            rows.append(r)
            cl = m['clients']
            tl_acc = np.array([c['threat_landscape']['accuracy'] for c in cl])
            tl_f1 = np.array([c['threat_landscape']['macro_f1'] for c in cl])
            tl_fpr = np.array([c['threat_landscape']['fpr'] for c in cl], float)
            tl_rec = np.array([c['threat_landscape']['recall'] for c in cl])
            r = dict(ds=ds, method='local', view='threat_landscape', seed=s, alpha=a, acc=tl_acc.mean(),
                     f1=tl_f1.mean(), fpr=np.nanmean(tl_fpr), acc_median=np.median(tl_acc), acc_min=tl_acc.min(),
                     acc_max=tl_acc.max(), acc_p10=np.percentile(tl_acc, 10), f1_median=np.median(tl_f1),
                     best=np.mean([c['best_epoch'] for c in cl]))
            for c, v in zip(CL, tl_rec.mean(0)):
                r['rec_' + c] = v
            rows.append(r)
            tr = pd.read_csv(d + '/trajectory.csv')
            last = tr.groupby('client').epoch.max()
            conv_rows.append(dict(ds=ds, method='local', seed=s, alpha=a, best_mean=np.mean([c['best_epoch'] for c in cl]),
                                  stop_mean=last.mean(), stop_max=last.max(), hit_budget=int((last >= 200).sum())))
            for c in cl:
                x = c['gateway_matched']
                gm_rows.append(dict(ds=ds, seed=s, alpha=a, gw=c['client'], method='local',
                                    acc=x['accuracy'], f1p=x['macro_f1_present'], fpr=x['fpr'], n=x['n'],
                                    n_train=c['n_train']))
            # ---------------- federated
            for meth in ['fedavg', 'fedprox']:
                d = run_dir(R, ds, meth, s, a)
                m = jl(d + '/metrics.json')
                g = m['global_test']
                r = dict(ds=ds, method=meth, view='global', seed=s, alpha=a, acc=g['accuracy'], f1=g['macro_f1'],
                         fpr=g['fpr'], best=m['best_round'], stop=m['stopped_round'], best_val=m['best_val_macro_f1'],
                         mb_best=m['serialized_bytes_at_best'] / 1e6, mb_stop=m['serialized_bytes_at_stop'] / 1e6,
                         seconds=m['seconds'])
                for c, v in zip(CL, g['recall']):
                    r['rec_' + c] = v
                rows.append(r)
                for x in m['gateway_matched']:
                    gm_rows.append(dict(ds=ds, seed=s, alpha=a, gw=x['client'], method=meth,
                                        acc=x['accuracy'], f1p=x['macro_f1_present'], fpr=x['fpr'], n=x['n']))
                # convergence: rounds and MB to 90 % / 95 % of the same seed's centralized best val macro-F1
                tr = pd.read_csv(d + '/trajectory.csv')
                led = pd.read_csv(d + '/communication.csv')
                per_round = led.groupby('round').serialized_bytes.sum()
                cum = per_round.cumsum()
                assert (per_round == per_round.iloc[0]).all()
                BYTES[ds] = int(led.serialized_bytes.iloc[0])
                assert abs(cum.loc[m['best_round']] - m['serialized_bytes_at_best']) < 1 if m['best_round'] > 0 else True
                rr = dict(ds=ds, method=meth, seed=s, alpha=a, best=m['best_round'], stop=m['stopped_round'],
                          mb_best=m['serialized_bytes_at_best'] / 1e6, mb_stop=m['serialized_bytes_at_stop'] / 1e6,
                          hit_budget=int(m['stopped_round'] >= 200), seconds=m['seconds'],
                          mb_round=per_round.iloc[0] / 1e6)
                for frac in (0.90, 0.95):
                    thr = frac * cen_best_val[s]
                    hit = tr[(tr['round'] > 0) & (tr.val_macro_f1 >= thr)]
                    if len(hit):
                        r0 = int(hit['round'].iloc[0])
                        rr[f'r{int(frac*100)}'] = r0
                        rr[f'mb{int(frac*100)}'] = cum.loc[r0] / 1e6
                    else:
                        rr[f'r{int(frac*100)}'] = np.nan
                        rr[f'mb{int(frac*100)}'] = np.nan
                conv_rows.append(rr)

runs = pd.DataFrame(rows)
runs.to_csv(f'{OUT}/runs.csv', index=False)
GM = pd.DataFrame(gm_rows)
GM.to_csv(f'{OUT}/gateway_matched.csv', index=False)
CONV = pd.DataFrame(conv_rows)
CONV.to_csv(f'{OUT}/convergence.csv', index=False)
print('serialized bytes per message', BYTES)
print(runs.groupby(['ds', 'method', 'view']).size())

# ---------------- cross-gateway (every local-only run with a cross_gateway.parquet)
cross = []
pergw = []
for ds in CLASSES:
    for s in S:
        for a in A:
            p = run_dir(R, ds, 'local', s, a) + '/cross_gateway.parquet'
            if not os.path.exists(p):
                continue
            t = pd.read_parquet(p)
            own = t[t.model == t.gateway]
            oth = t[t.model != t.gateway]
            # per model: pooled accuracy on all other gateways' test records (T_glob without T_k)
            t2 = oth.assign(c=oth.accuracy * oth.n)
            pm = t2.groupby('model').agg(c=('c', 'sum'), n=('n', 'sum'))
            pooled = (pm.c / pm.n)
            cross.append(dict(ds=ds, seed=s, alpha=a, own_acc=own.accuracy.mean(), other_pair_acc=oth.accuracy.mean(),
                              other_pooled_acc=pooled.mean(), other_pooled_min=pooled.min(),
                              share_pairs_below90=(oth.accuracy < 0.9).mean(),
                              own_f1p=own.macro_f1_present.mean(), other_f1p=oth.macro_f1_present.mean(),
                              own_fpr=own.fpr.astype(float).mean(), other_fpr=oth.fpr.astype(float).mean()))
            # per gateway j: mean accuracy of the 104 borrowed models on T_j
            b = oth.groupby('gateway').agg(borrowed_acc=('accuracy', 'mean'), borrowed_f1p=('macro_f1_present', 'mean'))
            for j, v in b.iterrows():
                pergw.append(dict(ds=ds, seed=s, alpha=a, gw=j, borrowed_acc=v.borrowed_acc, borrowed_f1p=v.borrowed_f1p))
CROSS = pd.DataFrame(cross)
CROSS.to_csv(f'{OUT}/cross_runs.csv', index=False)
PG = pd.DataFrame(pergw)
PG.to_csv(f'{OUT}/cross_pergw.csv', index=False)
print('cross-gateway runs:', CROSS.groupby('ds').size().to_dict())

# ---------------- stopping-rule sensitivity (CIC-IDS2017 first grid vs re-run)
sens = []
for meth in ['centralized', 'local', 'fedavg', 'fedprox']:
    for s in S:
        for a in ([None] if meth == 'centralized' else A):
            new = jl(run_dir(R, 'cicids2017', meth, s, a) + '/metrics.json')
            old = jl(run_dir(ARCH, 'cicids2017', meth, s, a) + '/metrics.json')
            if meth == 'local':
                gn, go = new['routed'], old['routed']
                bn = np.mean([c['best_epoch'] for c in new['clients']]); bo = np.mean([c['best_epoch'] for c in old['clients']])
            else:
                gn, go = new['global_test'], old['global_test']
                k = 'best_epoch' if meth == 'centralized' else 'best_round'
                bn, bo = new[k], old[k]
            sens.append(dict(method=meth, seed=s, alpha=a, acc_old=go['accuracy'], acc_new=gn['accuracy'],
                             f1_old=go['macro_f1'], f1_new=gn['macro_f1'], best_old=bo, best_new=bn,
                             val_old=old.get('best_val_macro_f1', np.nan), val_new=new.get('best_val_macro_f1', np.nan)))
SENS = pd.DataFrame(sens)
SENS.to_csv(f'{OUT}/stopping_sensitivity.csv', index=False)

# ---------------- confusion matrices (global test, summed over seeds)
cms = {}
for ds in CLASSES:
    for meth in ['centralized', 'fedavg', 'fedprox']:
        for a in ([None] if meth == 'centralized' else [100.0, 0.1]):
            tot = None
            for s in S:
                m = jl(run_dir(R, ds, meth, s, a) + '/metrics.json')
                c = np.array(m['confusion'])
                tot = c if tot is None else tot + c
            cms[(ds, meth, a)] = tot
    for a in [100.0, 0.1]:
        tot = None
        for s in S:
            c = np.array(jl(run_dir(R, ds, 'local', s, a) + '/metrics.json')['routed_confusion'])
            tot = c if tot is None else tot + c
        cms[(ds, 'local_routed', a)] = tot
np.save(f'{OUT}/confusions.npy', {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in cms.items()}, allow_pickle=True)
print('done')
