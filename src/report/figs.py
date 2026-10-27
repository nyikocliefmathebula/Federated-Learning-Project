# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Draw all six Chapter 5 figures by running the fig5_*.py scripts one after another.

Each figure can also be drawn on its own, for example:  python src/report/fig5_1_global_test_performance.py
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for script in sorted(HERE.glob('fig5_*.py')):
    runpy.run_path(str(script), run_name='__main__')
print('figures written')
