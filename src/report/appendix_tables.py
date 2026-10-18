# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Appendix G: full paired-difference statistics (Eq. 3.15) from results/report/csv/paired.csv, as LaTeX longtable rows."""
import pandas as pd
from report_paths import CSV, TABLES

P = pd.read_csv(f'{CSV}/paired.csv')
A = [100.0, 10.0, 1.0, 0.5, 0.1]
PAIRS = [('FedProx-FedAvg', '\\fedprox{} $-$ \\fedavg{}'),
         ('FedAvg-Cen', '\\fedavg{} $-$ centralized'),
         ('FedProx-Cen', '\\fedprox{} $-$ centralized'),
         ('FedAvg-LocalTL', '\\fedavg{} $-$ local-only on $T_{\\mathrm{glob}}$'),
         ('FedProx-LocalTL', '\\fedprox{} $-$ local-only on $T_{\\mathrm{glob}}$'),
         ('FedAvg-LocalRouted', '\\fedavg{} $-$ local-only, routed'),
         ('FedProx-LocalRouted', '\\fedprox{} $-$ local-only, routed')]
print('ties present:', P[P.ties > 0][['ds', 'alpha', 'pair', 'met', 'ties']].to_string())


def f(x, d=3):
    s = f'{x:+.{d}f}'
    return s.replace('-', '$-$') if s.startswith('-') else s


lines = []
for ds, name in [('cicids2017', 'Corrected \\cicids{}'), ('ciciot2023', '\\ciciot{}')]:
    lines.append(f'\\ThesisGroup{{11}}{{{name}}} \\\\ \\hline')
    for key, label in PAIRS:
        lines.append(f'\\multicolumn{{11}}{{|l|}}{{\\textit{{{label}}}}} \\\\ \\hline')
        for a in A:
            cells = []
            for met, scale in [('acc', 100), ('f1', 1)]:
                q = P[(P.ds == ds) & (P.alpha == a) & (P.pair == key) & (P.met == met)].iloc[0]
                cells += [f(q['mean'] * scale), f'{q.sd * scale:.3f}', f(q.mn * scale), f(q.mx * scale), f'{int(q.wins)}']
            lines.append(f'{a:g} & ' + ' & '.join(cells) + ' \\\\ \\hline')
open(f'{TABLES}/t_appG_paired.tex', 'w').write('\n'.join(lines) + '\n')
print(len(lines), 'rows written')
