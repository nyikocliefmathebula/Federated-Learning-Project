# Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town 
# 2026

"""Stage 1: audit and clean one dataset (report Sections 3.3, 3.4 and 4.3.1).

Running from the project root (/mnt/c/Users/Nyiko/fl-iot-ids) with .venv active:
    python src/clean_data.py --dataset cicids2017
    python src/clean_data.py --dataset ciciot2023

Raw files are only read. Outputs in data/processed/<dataset>/:
    clean/part-*.parquet   kept records: identifiers, labels, float32 features
    data_manifest.json     files, checksums, schema, label mapping, counts, rules
    data_audit.csv         row counts by source label after every cleaning step
    cleaning_error.json    written instead when a check stops the run
"""
# How this script works, in short:
#   checks   - every raw file must have exactly the expected columns, end with
#              a complete line and (optionally) match its recorded checksum
#   pass 1   - read each file in chunks of 200,000 rows, give every row a class,
#              and remove rows that are broken, unwanted or have invalid values
#   pass 2   - over the whole dataset, find rows with identical feature values
#              (label conflicts and exact duplicates)
#   pass 3   - write the kept rows to clean/part-*.parquet
#   records  - data_audit.csv and data_manifest.json, with checks that every row
#              read is either kept or counted under exactly one removal step
import argparse
import csv
import hashlib
import io
import json
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from label_maps import (ATTEMPTED_SUFFIX, CICIDS2017_MAP, CICIOT2023_MAP,
                        DATASETS, ciciot_folder_to_fine, mapping_table)
from schema import (CICIOT2023_FOLDERS, DAYS, PROCESSED, RAW, ciciot2023_base,
                    known_truncation, require_project_root, schema)

# rows read at a time (one input chunk -> one output part file)
CHUNK_ROWS = 200_000
# the removal steps in the order they are applied; a removed row is counted
# only under the first step that removes it
STEPS = ["truncated_final_line", "malformed_row", "attempted",
         "excluded_rare_class", "invalid_value", "negative_value",
         "label_conflict", "exact_duplicate"]


# raised by any check that must stop cleaning; main() catches it and writes
# cleaning_error.json
class CleaningError(Exception):
    """A check failed; the message is written to cleaning_error.json."""


# ----------------------------------------------------------------- discovery
def list_sources(dataset):
    """Return [(csv_path, folder_label_or_None)] in a fixed order."""
    if dataset == "cicids2017":
        # CIC-IDS2017: the five day files, Monday to Friday (no folder label)
        paths = [RAW / "cicids2017_improved" / f"{d}.csv" for d in DAYS]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise CleaningError(f"Missing CIC-IDS2017 files: {missing}")
        return [(p, None) for p in paths]
    try:
        # CIC-IoT2023: every CSV in each of the 34 class folders; the folder name
        # is the label of all rows in its files
        base = ciciot2023_base()
    except (FileNotFoundError, FileExistsError) as exc:
        raise CleaningError(str(exc)) from None
    # sorted, so the files are always read in the same order
    folders = sorted(p for p in base.iterdir() if p.is_dir())
    if len(folders) != CICIOT2023_FOLDERS:
        raise CleaningError(f"Expected {CICIOT2023_FOLDERS} class folders in "
                            f"{base}, found {len(folders)}")
    sources = []
    for folder in folders:
        files = sorted(folder.glob("*.csv"))
        if not files:
            raise CleaningError(f"No CSV files in {folder}")
        sources += [(p, folder.name) for p in files]
    return sources


# SHA-256 fingerprint of a file, read in 1 MB blocks
def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def read_reference_checksums(dataset):
    """Checksums recorded earlier with sha256sum (docs/datasets), if any."""
    # each line of the reference file is '<checksum>  <file name>'
    ref = Path("docs/datasets") / f"{dataset}_sha256.txt"
    if not ref.exists():
        return None, str(ref)
    table = {}
    for line in ref.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            table[parts[1].lstrip("*").strip()] = parts[0]
    return table, str(ref)


def integrity_scan(dataset, sources, expected, with_checksums):
    """Header, size, final-newline and checksum checks before any parsing."""
    records, problems = [], []
    for path, _ in sources:
        rel = str(path.relative_to(RAW))
        size = path.stat().st_size
        # read only the first line (the column names)
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            header = [c.strip() for c in next(csv.reader(f), [])]
        # compare the column names with the expected list in schema.py
        dupes = sorted({c for c in header if header.count(c) > 1})
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        with open(path, "rb") as f:
            # look at the very last byte: a complete file ends with a newline
            f.seek(max(size - 1, 0))
            ends_newline = f.read(1) == b"\n"
        rec = {"file": rel, "bytes": size, "columns": len(header),
               "ends_with_newline": ends_newline}
        if with_checksums:
            rec["sha256"] = sha256(path)
        records.append(rec)
        if dupes:
            problems.append(f"{rel}: duplicated column names {dupes}")
        if missing:
            problems.append(f"{rel}: missing columns {missing}")
        if extra:
            problems.append(f"{rel}: unexpected columns {extra}")
        if not dupes and not missing and not extra and header != expected:
            problems.append(f"{rel}: columns are in a different order")
        # a file no larger than its header line has no data
        if size <= len(",".join(header)) + 1:
            problems.append(f"{rel}: file has no data rows")
    return records, problems


def classify_truncation(dataset, files):
    """Split files without a final newline into (known, unknown) sets.

    Known: listed in schema.CICIOT2023_KNOWN_TRUNCATED with matching size and,
    when computed, SHA-256 (an issue of the published archive). Unknown: any
    other truncated file. Adds a "truncation" reason to each affected record.
    """
    known, unknown = set(), set()
    for f in files:
        if f["ends_with_newline"]:
            continue
        # known = one of the three listed DoS-UDP_Flood files with the expected size
        # and checksum; any other file without a final newline is unknown
        spec = known_truncation(f["file"]) if dataset == "ciciot2023" else None
        if spec and f["bytes"] == spec[0] and f.get("sha256", spec[1]) == spec[1]:
            known.add(f["file"])
            f["truncation"] = "incomplete last line in the published archive (known)"
        else:
            unknown.add(f["file"])
            f["truncation"] = "incomplete last line (allowed by --allow-truncated)"
    return known, unknown


# ------------------------------------------------------------------ parsing
def open_source(path, truncated):
    """File handle for pandas; a truncated final line is cut off first."""
    if not truncated:
        return path, 0
    # read the file and keep everything up to the last newline; the incomplete
    # line after it is dropped and counted (1) as truncated_final_line
    data = path.read_bytes()
    cut = data.rfind(b"\n") + 1
    return io.BytesIO(data[:cut]), 1


def malformed_rows(chunk):
    """Rows whose missing values form a contiguous block at the end (short or
    blank lines). pandas pads such lines with NaN instead of failing."""
    na = chunk.isna().to_numpy()
    if not na.any():
        return np.zeros(len(chunk), dtype=bool)
    # count missing values from the end of each row; if all of a row's missing
    # values are at its end (and the last column is missing), the line was short
    suffix = np.cumprod(na[:, ::-1], axis=1).sum(axis=1)
    return na[:, -1] & (suffix == na.sum(axis=1))


def clean_chunk(chunk, ctx, path, folder, first_row):
    """Clean one chunk. Returns (kept DataFrame, per-row step labels)."""
    dataset, policy = ctx["dataset"], ctx["attempted"]
    # remove spaces around the column names
    chunk.columns = [c.strip() for c in chunk.columns]
    n = len(chunk)
    # position of each row in its source file (0 = first data row)
    rows = np.arange(first_row, first_row + n, dtype=np.int64)
    # step[i] = name of the step that removes row i; "" = still kept
    step = np.full(n, "", dtype=object)

    bad = malformed_rows(chunk)
    if dataset == "cicids2017":
        # --- CIC-IDS2017: the label is in the 'Label' column
        label = chunk["Label"].astype("string").str.strip()
        bad |= label.isna().to_numpy()
        step[bad] = "malformed_row"
        ok = ~bad
        label = label.fillna("<malformed>").astype(str).to_numpy()
        # a row is an attempted attack if 'Attempted Category' is not -1; the label
        # must then end with ' - Attempted' (and the other way round)
        attempted_cat = pd.to_numeric(chunk["Attempted Category"], errors="coerce")
        attempted = (attempted_cat.fillna(-1).to_numpy() != -1)
        suffix = np.char.endswith(label.astype(str), ATTEMPTED_SUFFIX)
        clash = ok & (attempted != suffix)
        if clash.any():
            raise CleaningError(f"{path.name}: Label and Attempted Category "
                                f"disagree in {int(clash.sum())} rows")
        # the 'id' column should equal the row position + 1; differences are only
        # counted (recorded in the manifest), not removed
        ids = pd.to_numeric(chunk["id"], errors="coerce").to_numpy()
        ctx["id_mismatch"][path.name] += int((ok & (ids != rows + 1)).sum())
        # label without ' - Attempted'; every label must be in label_maps.py
        base = np.where(attempted, [l[:-len(ATTEMPTED_SUFFIX)] for l in label],
                        label)
        unknown = ok & ~attempted & ~np.isin(label, list(CICIDS2017_MAP))
        unknown |= ok & attempted & ~np.isin(base, list(CICIDS2017_MAP))
        if unknown.any():
            raise CleaningError(f"{path.name}: unmapped labels "
                                f"{sorted(set(label[unknown]))}")
        # task class of each row (None = rare class that is excluded)
        cls = np.array([CICIDS2017_MAP.get(l) for l in label], dtype=object)
        if policy == "benign":
            cls[attempted] = "Benign"
        else:
            step[ok & attempted & (step == "")] = "attempted"
        step[ok & ~attempted & (step == "") &
             np.array([c is None for c in cls])] = "excluded_rare_class"
        # drop identifier, timestamp and label columns; the rest are features
        features = chunk.drop(columns=list(ctx["excluded"]))
        # kept next to the features but never used as inputs: group_key is
        # '<day>|<Flow ID>' (used by split_data.py), timestamp is for tracing
        meta = {
            "group_key": (path.stem + "|" +
                          chunk["Flow ID"].astype("string").fillna("")).to_numpy(),
            "timestamp": chunk["Timestamp"].astype("string").fillna("").to_numpy(),
        }
    else:
        # --- CIC-IoT2023: every row in this file has the folder's label
        fine = ciciot_folder_to_fine(folder)
        label = np.full(n, fine, dtype=object)
        cls = np.full(n, CICIOT2023_MAP[fine], dtype=object)
        step[bad] = "malformed_row"
        features = chunk
        meta = {}

    # --- both datasets: text that is not a number becomes NaN, and so does
    # +/- infinity; a row with any NaN is removed as invalid_value
    x = features.apply(pd.to_numeric, errors="coerce").astype("float64")
    x = x.replace([np.inf, -np.inf], np.nan)
    step[(step == "") & x.isna().any(axis=1).to_numpy()] = "invalid_value"
    # a negative value in a column that cannot be negative -> negative_value
    if ctx["nonnegative"]:
        neg = (x[ctx["nonnegative"]] < 0).any(axis=1).to_numpy()
        step[(step == "") & neg] = "negative_value"

    # build the table of kept rows: identifiers first, then float32 features
    keep = step == ""
    x32 = x[keep].astype("float32").reset_index(drop=True)
    out = pd.DataFrame({
        # unique name of each record: '<file name>:<row position>'
        "record_key": [f"{path.name}:{r}" for r in rows[keep]],
        "source_file": path.name,
        "source_row": rows[keep],
        "source_label": label[keep].astype(str),
        "class": cls[keep].astype(str),
    })
    for name, values in meta.items():
        out[name] = values[keep]
    # 64-bit fingerprint of each row's float32 features: rows with identical
    # features get the same number (used to find duplicates in pass 2)
    out["feat_hash"] = pd.util.hash_pandas_object(x32, index=False).to_numpy()
    return pd.concat([out, x32], axis=1), list(features.columns), label, cls, step


def find_conflicts_and_duplicates(h, f):
    """h: 64-bit feature hashes, f: task-class codes, in file order.
    Returns (conflict, duplicate) masks: every row of a hash group with more
    than one class is a conflict; otherwise all but the first are duplicates."""
    # Idea: sort the rows by fingerprint so identical rows sit next to each other.
    # Each run of equal fingerprints is one group. If the group's smallest and
    # largest class codes differ, the group has more than one class.
    order = np.argsort(h, kind="stable")          # keeps first occurrence first
    hs, fs = h[order], f[order]
    if len(hs):
        # starts = position where each new group begins
        starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])
        mixed = np.minimum.reduceat(fs, starts) != np.maximum.reduceat(fs, starts)
    else:
        starts, mixed = np.zeros(0, np.int64), np.zeros(0, bool)
    del hs
    # spread each group's result to all of its rows
    conflict_sorted = np.repeat(mixed, np.diff(np.r_[starts, len(fs)]))
    # the first row of each group (kept if not a conflict)
    first = np.zeros(len(fs), dtype=bool)
    first[starts] = True
    del fs, starts, mixed
    # put the answers back into the original file order
    conflict = np.empty(len(h), dtype=bool)
    duplicate = np.empty(len(h), dtype=bool)
    conflict[order] = conflict_sorted
    duplicate[order] = ~first & ~conflict_sorted
    return conflict, duplicate


# ---------------------------------------------------------------------- main
def main():
    require_project_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--attempted", choices=["benign", "exclude"], default="benign",
                    help="CIC-IDS2017 '- Attempted' flows: relabel as Benign "
                         "(dataset authors' recommendation) or exclude")
    ap.add_argument("--allow-truncated", action="store_true",
                    help="drop the incomplete final line of any file instead of "
                         "stopping (the three known DoS-UDP_Flood files of the "
                         "published CIC-IoT2023 archive need no switch)")
    # --skip-checksums: do not compute file SHA-256 values (faster, but the manifest
    # then records no checksums)
    ap.add_argument("--skip-checksums", action="store_true")
    # --chunk-rows: rows per chunk (also the size of each output part)
    ap.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    args = ap.parse_args()
    started = time.time()
    out_dir = PROCESSED / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    error_file = out_dir / "cleaning_error.json"
    # remove an error file left by an earlier failed attempt
    if error_file.exists():
        error_file.unlink()
    try:
        run(args, out_dir, started)
    except CleaningError as exc:
        error_file.write_text(json.dumps({
            "dataset": args.dataset, "error": str(exc),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
        print(f"CLEANING STOPPED: {exc}\nRecorded in {error_file}", file=sys.stderr)
        sys.exit(1)


def run(args, out_dir, started):
    sch = schema(args.dataset)
    sources = list_sources(args.dataset)
    print(f"{len(sources)} source files; checking headers, endings"
          f"{'' if args.skip_checksums else ' and checksums'} ...", flush=True)
    # ---- checks before reading any data
    files, problems = integrity_scan(args.dataset, sources, sch["header"],
                                     not args.skip_checksums)
    known, unknown = classify_truncation(args.dataset, files)
    truncated = known | unknown
    if unknown and not args.allow_truncated:
        problems.append(
            "files end with an incomplete line and are not in the known list "
            f"(interrupted download or extraction?): {sorted(unknown)}. Extract or "
            "download them again, or rerun with --allow-truncated to drop the "
            "incomplete final line.")
    if known:
        print(f"Known release issue: the last line of {len(known)} file(s) is "
              f"incomplete and will be dropped: {sorted(known)}", flush=True)
    if problems:
        raise CleaningError(" | ".join(problems))
    # compare with docs/datasets/<dataset>_sha256.txt if it exists; a difference
    # only prints a WARNING (it does not stop cleaning)
    reference, ref_path = read_reference_checksums(args.dataset)
    ref_status = "no reference file"
    if reference is not None and not args.skip_checksums:
        mismatched = [f["file"] for f in files
                      if f["file"] in reference and reference[f["file"]] != f["sha256"]]
        absent = [f["file"] for f in files if f["file"] not in reference]
        ref_status = ("all match" if not mismatched and not absent else
                      f"mismatch: {mismatched}; not in reference: {absent}")
        if mismatched or absent:
            print(f"WARNING: checksums differ from {ref_path}: {ref_status}")

    # stage1/ holds the pass-1 output temporarily; clean/ the final parts. Both are
    # emptied first, so an earlier run's files are never mixed in.
    stage, final = out_dir / "stage1", out_dir / "clean"
    for folder in (stage, final):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    ctx = {"dataset": args.dataset, "attempted": args.attempted,
           "excluded": sch["excluded"], "nonnegative": sch["nonnegative"],
           "id_mismatch": Counter()}
    read = Counter()                       # (source_label, class) -> rows read
    removed = defaultdict(Counter)         # step -> (source_label, class) -> rows
    feature_cols, part = None, 0
    dtypes = ({c: "string" for c in ["Flow ID", "Src IP", "Dst IP",
                                     "Timestamp", "Label"]}
              if args.dataset == "cicids2017" else None)
    by_file = {f["file"]: f for f in files}
    # ---- pass 1: read and clean each source file chunk by chunk
    for path, folder in sources:
        rel = str(path.relative_to(RAW))
        handle, cut_lines = open_source(path, rel in truncated)
        first_row = 0
        reader = pd.read_csv(handle, chunksize=args.chunk_rows, dtype=dtypes,
                             encoding="utf-8-sig", low_memory=False,
                             skip_blank_lines=False)
        for chunk in reader:
            kept, feature_cols, label, cls, step = clean_chunk(
                chunk, ctx, path, folder, first_row)
            first_row += len(chunk)
            # count every row read and every row removed, by (source label, class)
            dest = np.array(["excluded" if c is None else c for c in cls], dtype=object)
            if args.dataset == "cicids2017" and args.attempted == "exclude":
                dest[step == "attempted"] = "excluded"
            for key, count in Counter(zip(label, dest)).items():
                read[key] += count
            for s, l, d in zip(step[step != ""], label[step != ""], dest[step != ""]):
                removed[s][(l, d)] += 1
            # save this chunk's kept rows as stage1/part-NNNNN.parquet
            pq.write_table(pa.Table.from_pandas(kept, preserve_index=False),
                           stage / f"part-{part:05d}.parquet")
            part += 1
        # a dropped incomplete last line is counted as read and then removed
        if cut_lines:
            fine = (ciciot_folder_to_fine(folder) if folder else "<unknown>")
            dest = CICIOT2023_MAP.get(fine, "excluded") if folder else "excluded"
            read[(fine, dest)] += 1
            removed["truncated_final_line"][(fine, dest)] += 1
        by_file[rel]["data_rows"] = first_row + cut_lines
        print(f"read {first_row + cut_lines:>10,} rows  {rel}", flush=True)

    # Pass 2: duplicates and label conflicts across the whole dataset.
    # Integer codes keep memory low (CIC-IoT2023 has about 46 million rows).
    parts = sorted(stage.glob("part-*.parquet"))
    classes = DATASETS[args.dataset]["classes"]
    label_list = sorted({l for l, _ in read})
    # load only the fingerprint, class and label of every kept row
    hashes, fams, labels = [], [], []
    for p in parts:
        t = pq.read_table(p, columns=["feat_hash", "class", "source_label"]).to_pandas()
        hashes.append(t["feat_hash"].to_numpy(np.uint64))
        fams.append(pd.Categorical(t["class"], classes).codes.astype(np.int16))
        labels.append(pd.Categorical(t["source_label"], label_list).codes.astype(np.int16))
    h = np.concatenate(hashes) if hashes else np.zeros(0, np.uint64)
    f = np.concatenate(fams) if fams else np.zeros(0, np.int16)
    lab = np.concatenate(labels) if labels else np.zeros(0, np.int16)
    del hashes, fams, labels
    if (f < 0).any() or (lab < 0).any():
        raise CleaningError("A kept record has an unknown class or source label")
    conflict, duplicate = find_conflicts_and_duplicates(h, f)
    # count removed rows by (source label, class); label*64 + class packs both
    # into one number so np.unique can count them quickly
    pair = lab.astype(np.int32) * 64 + f.astype(np.int32)
    for name, mask in (("label_conflict", conflict), ("exact_duplicate", duplicate)):
        codes, counts = np.unique(pair[mask], return_counts=True)
        for code, count in zip(codes, counts):
            key = (label_list[code // 64], classes[code % 64])
            removed[name][key] += int(count)
    # True for each row that survives pass 2
    keep_all = ~(conflict | duplicate)
    del h, f, lab, pair

    # Pass 3: write the final parts.
    start, kept_class, kept_label = 0, Counter(), Counter()
    for p in parts:
        table = pq.read_table(p).to_pandas()
        # the part of keep_all that belongs to this part file
        mask = keep_all[start:start + len(table)]
        start += len(table)
        table = table[mask].drop(columns=["feat_hash"]).reset_index(drop=True)
        kept_class.update(table["class"].value_counts().to_dict())
        kept_label.update(Counter(zip(table["source_label"], table["class"])))
        pq.write_table(pa.Table.from_pandas(table, preserve_index=False),
                       final / p.name, compression="zstd")
    # delete the temporary stage1/ folder
    shutil.rmtree(stage)

    # Audit: rows remaining by source label after each step.
    # one row per (step, source label, class): rows removed by that step and rows
    # still remaining after it
    audit, remaining = [], Counter(read)
    for i, s in enumerate(["read"] + STEPS):
        if s != "read":
            for key, count in removed[s].items():
                remaining[key] -= count
        for (l, d) in sorted(read):
            audit.append({"step_order": i, "step": s, "source_label": l,
                          "class": d, "removed": 0 if s == "read" else removed[s][(l, d)],
                          "remaining": remaining[(l, d)]})
    pd.DataFrame(audit).to_csv(out_dir / "data_audit.csv", index=False)
    # accounting check: rows read = rows removed + rows kept, for every label
    total_read = sum(read.values())
    total_removed = sum(sum(c.values()) for c in removed.values())
    total_kept = sum(kept_class.values())
    if total_read != total_removed + total_kept:
        raise CleaningError("Row accounting does not balance "
                            f"({total_read} != {total_removed} + {total_kept})")
    for (l, d), count in kept_label.items():
        if remaining[(l, d)] != count:
            raise CleaningError(f"Audit mismatch for {l} -> {d}")

    # ---- data_manifest.json: everything needed to describe and repeat this run.
    # Note: it records the SHA-256 of this script file as it was at cleaning time.
    script = Path(__file__).resolve()
    manifest = {
        "dataset": args.dataset,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_folder": str(RAW.resolve()),
        "release": ("Corrected CIC-IDS2017 (Engelen et al. 2021; Liu et al., CNS 2022), "
                    "CICIDS2017_improved" if args.dataset == "cicids2017" else
                    "CIC-IoT2023 (Neto et al. 2023), per-class CSV folders, 39 features"),
        "files": files,
        "truncated_files": [{"file": f["file"], "reason": f["truncation"]}
                            for f in files if "truncation" in f],
        "checksum_reference": {"file": ref_path, "status": ref_status},
        "id_column_mismatches": dict(ctx["id_mismatch"]),
        "source_columns": sch["header"],
        "excluded_columns": sch["excluded"],
        "feature_columns": feature_cols,
        "n_features_before_constant_removal": len(feature_cols),
        "nonnegative_columns": sch["nonnegative"],
        "attempted_policy": args.attempted if args.dataset == "cicids2017" else None,
        "label_mapping": [{"source_label": a, "class": b, "note": c}
                          for a, b, c in mapping_table(args.dataset, args.attempted)],
        "class_order": classes,
        "class_codes": {c: i for i, c in enumerate(classes)},
        "rules": [
            "header must match the expected schema exactly",
            "record key = source file name + 0-based data-row position",
            "short or blank lines are removed (malformed_row)",
            "every source label must be mapped; an unmapped label stops the run",
            "identifier, timestamp and label columns are excluded from features",
            "non-numeric, missing or infinite feature values are removed",
            "negative values in fields that cannot be negative are removed",
            "identical feature vectors with different task classes are all removed",
            "identical feature vectors with the same class keep the first copy",
            "features stored as float32; duplicate check uses a 64-bit row hash "
            "of the float32 values",
        ],
        "rows_read": int(total_read),
        "rows_removed": int(total_removed),
        "rows_kept": int(total_kept),
        "removed_by_step": {s: int(sum(removed[s].values())) for s in STEPS},
        "kept_by_class": {c: int(kept_class.get(c, 0)) for c in classes},
        "n_classes": len(classes),
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "pyarrow": pa.__version__},
        "script": {"file": str(script.name), "sha256": sha256(script)},
        "chunk_rows": args.chunk_rows,
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ("rows_read", "rows_removed", "rows_kept",
                       "removed_by_step", "kept_by_class")}, indent=2))
    if truncated:
        print(f"NOTE: incomplete final lines dropped in {sorted(truncated)} "
              "(recorded in data_manifest.json)")
    print("CLEANING_OK")


if __name__ == "__main__":
    main()
