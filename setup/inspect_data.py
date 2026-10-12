# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

# First look at the raw data (run from the project root:
# python setup/inspect_data.py). For each dataset folder it counts the CSV
# files, then reads the first 1000 rows of the first two files and prints their
# size, column names and, if there is a 'Label' column, how often each label
# occurs. It only reads files. CIC-IoT2023 files have no label column (the
# folder name is the label), so for them it prints 'Inspect the schema ...'.
from pathlib import Path

import pandas as pd

folders = [
    Path("data/raw/cicids2017_improved"),
    Path("data/raw/ciciot2023"),
]
for folder in folders:
    # all .csv files in the folder and its subfolders, in name order
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() == ".csv"
    )
    print(f"\nDataset folder: {folder}")
    print(f"CSV files found: {len(files)}")
    for path in files[:2]:
        sample = pd.read_csv(
            path, nrows=1000, encoding="utf-8-sig"
        )
        # remove spaces around column names (some CIC-IDS2017 versions have them)
        sample.columns = sample.columns.str.strip()
        print(f"\nFile: {path.name}")
        print("Sample shape:", sample.shape)
        print("Columns:", sample.columns.tolist())
        # columns called 'label' in any upper/lower case
        label_columns = [
            c for c in sample.columns
            if c.casefold() == "label"
        ]
        if len(label_columns) == 1:
            print(sample[label_columns[0]].value_counts(
                dropna=False))
        else:
            print("Inspect the schema to identify the label.")
