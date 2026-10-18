# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Paired differences (Eq. 3.15) between approaches, per dataset, alpha and metric, from results/report/csv/runs.csv."""
import pandas as pd
from report_paths import CSV
r = pd.read_csv(f'{CSV}/runs.csv')
A = [100.0, 10.0, 1.0, 0.5, 0.1]
def get(ds, key, a):
    if key == 'Cen':
        d = r[(r.ds == ds) & (r.method == 'centralized')]
    elif key == 'LocalTL':
        d = r[(r.ds == ds) & (r.method == 'local') & (r.view == 'threat_landscape') & (r.alpha == a)]
    elif key == 'LocalRouted':
        d = r[(r.ds == ds) & (r.method == 'local') & (r.view == 'routed') & (r.alpha == a)]
    else:
        d = r[(r.ds == ds) & (r.method == key.lower()) & (r.alpha == a)]
    return d.set_index('seed')
PAIRS = [('FedProx', 'FedAvg'), ('FedAvg', 'Cen'), ('FedProx', 'Cen'), ('FedAvg', 'LocalTL'), ('FedProx', 'LocalTL'),
         ('FedAvg', 'LocalRouted'), ('FedProx', 'LocalRouted')]
rows = []
for ds in ['cicids2017', 'ciciot2023']:
    for a in A:
        for x, y in PAIRS:
            for met in ['acc', 'f1', 'fpr']:
                d = (get(ds, x, a)[met] - get(ds, y, a)[met]).dropna()
                rows.append(dict(ds=ds, alpha=a, pair=f'{x}-{y}', met=met, mean=d.mean(), sd=d.std(), mn=d.min(), mx=d.max(),
                                 wins=int((d > 0).sum()), ties=int((d == 0).sum()), n=len(d)))
out = pd.DataFrame(rows)
out.to_csv(f'{CSV}/paired.csv', index=False)
print(len(out), 'paired differences written')
