# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Shared settings of the Chapter 5 figure scripts (fig5_*.py in this folder).

Fonts, colours, line markers, legend names, the alpha axis and the save function are
defined once here, so a change (for example a legend name in LAB) applies to every figure.
The figures are drawn only from the tables in results/report/csv/ and the saved run files.
"""
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
# class names in the order of the model outputs (used as axis labels)
CL = {'cicids2017': ['Benign', 'DoS', 'DDoS', 'PortScan', 'BruteForce', 'WebAttack', 'Botnet'],
      'ciciot2023': ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web', 'BruteForce']}
# sequential blue colour map of the heat maps (Figures 5.2 and 5.3)
cmap = LinearSegmentedColormap.from_list('blue', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])

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
    """Write the figure as PDF (used in the report) and 300-dpi PNG to results/report/figures/."""
    fig.savefig(f'{OUT}/{name}.pdf')
    fig.savefig(f'{OUT}/{name}.png', dpi=300)
    plt.close(fig)
    print(f'written: {OUT}/{name}.pdf and .png')
