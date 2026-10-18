# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Round-to-round variability of validation macro-F1 in federated runs (from trajectory.csv)."""
import pandas as pd, numpy as np
from report_paths import RUNS as R, CSV
rows=[]
for ds in ['cicids2017','ciciot2023']:
    for k in ['fedavg','fedprox']:
        for s in [13,42,271,1618,3141]:
            for a in [100,10,1,0.5,0.1]:
                t=pd.read_csv(f'{R}/{ds}/{k}/seed{s}/alpha{a:g}/trajectory.csv')
                v=t.sort_values('round').val_macro_f1.values
                v=v[1:]  # exclude round 0 (untrained model)
                d=np.diff(v)
                rows.append(dict(ds=ds,method=k,seed=s,alpha=a,mad=np.abs(d).mean(),drops=(d<-0.05).sum(),n=len(d)))
D=pd.DataFrame(rows)
D.to_csv(f'{CSV}/stability.csv',index=False)
g=D.groupby(['ds','method','alpha'])[['mad','drops']].mean().unstack('method')
print(g.round(4))
P=D.pivot_table(index=['ds','alpha','seed'],columns='method',values='mad').reset_index()
P['d']=P.fedprox-P.fedavg
print(P.groupby(['ds','alpha']).d.agg(['mean',lambda x:(x<0).sum()]).round(4))
