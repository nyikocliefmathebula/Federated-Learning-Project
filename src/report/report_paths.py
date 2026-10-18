# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Folders used by the report scripts in src/report/ (Chapter 5 and Appendix G of the report).

The scripts read only the saved run folders in results/runs/ and, for the stopping-rule
sensitivity analysis, the first CIC-IDS2017 grid in results/archive/runs_p15_r100/.
Everything they write goes to results/report/ (csv/, tables/, figures/).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = str(ROOT / 'results' / 'runs')
ARCHIVE = str(ROOT / 'results' / 'archive' / 'runs_p15_r100')
REPORT = ROOT / 'results' / 'report'
CSV = str(REPORT / 'csv')
TABLES = str(REPORT / 'tables')
FIGURES = str(REPORT / 'figures')
for _d in (CSV, TABLES, FIGURES):
    os.makedirs(_d, exist_ok=True)
