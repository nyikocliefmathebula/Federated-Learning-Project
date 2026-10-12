# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town 
# 2026

"""Check that the extracted raw files are exact copies of the downloaded archives.

Run from the project root (/mnt/c/Users/Nyiko/fl-iot-ids) with .venv active:
    python src/check_archives.py

For each dataset it compares every file in the zip archive in downloads/ with
the file of the same name under data/raw (size and CRC-32, the checksum stored
in the archive), lists files that are missing or extra, and records the
archive's size and SHA-256. Raw files and archives are only read.

Writes docs/datasets/archive_check.json and prints ARCHIVES_OK when every file
matches. A dataset whose archive is not in downloads/ is reported as skipped.
"""
import argparse
import hashlib
import json
import sys
import time
import zipfile
import zlib
from pathlib import Path

from schema import RAW, ciciot2023_base, require_project_root

# where the downloaded zip files are expected (they are only read)
ARCHIVES = {
    "cicids2017": Path("downloads/CICIDS2017_improved.zip"),
    "ciciot2023": Path("downloads/CSV.zip"),
}


# folder under data/raw that should hold the extracted files of one dataset
def raw_folder(dataset):
    if dataset == "cicids2017":
        return RAW / "cicids2017_improved"
    return ciciot2023_base()


# CRC-32 of a file, read in blocks of 16 MB so large files do not fill the
# memory; a zip archive stores this value for every file it contains
def file_crc(path, block=16 << 20):
    crc = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            crc = zlib.crc32(chunk, crc)
    return crc


# SHA-256 of a file (a longer, safer fingerprint), also read in 16 MB blocks
def file_sha256(path, block=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def common_prefix(names):
    """Top-level folder shared by every member (for example 'CSV/'), or ''."""
    tops = {n.split("/", 1)[0] for n in names}
    if len(tops) == 1 and all("/" in n for n in names):
        return tops.pop() + "/"
    return ""


def compare(archive, folder, progress=True):
    """Compare one archive with one folder. Returns a result dictionary."""
    # 1. list the files in the archive (only the table of contents is read)
    with zipfile.ZipFile(archive) as zf:
        members = [i for i in zf.infolist() if not i.is_dir()]
    # 2. names inside the archive may start with a top folder such as 'CSV/';
    #    remove it so the names match the paths under data/raw
    prefix = common_prefix([m.filename for m in members])
    expected = {m.filename[len(prefix):]: (m.file_size, m.CRC) for m in members}
    # 3. every file that is really on disk under the raw folder
    on_disk = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    # 4. names in only one of the two lists
    missing = sorted(set(expected) - on_disk)
    extra = sorted(on_disk - set(expected))
    mismatched, matched, last_dir = [], 0, None
    # 5. files in both: compare the size first (fast), then the CRC-32 (reads
    #    the whole file)
    for rel in sorted(set(expected) & on_disk):
        size, crc = expected[rel]
        path = folder / rel
        if progress and rel.split("/")[0] != last_dir and "/" in rel:
            last_dir = rel.split("/")[0]
            print(f"  checking {last_dir}/", flush=True)
        if path.stat().st_size != size:
            mismatched.append({"file": rel, "reason": "size differs",
                               "archive_bytes": size, "disk_bytes": path.stat().st_size})
        elif file_crc(path) != crc:
            mismatched.append({"file": rel, "reason": "CRC-32 differs"})
        else:
            matched += 1
    # ok is True only if nothing differs and nothing is missing (extra files
    # are listed but do not make the check fail)
    return {"archive": str(archive), "archive_bytes": archive.stat().st_size,
            "archive_sha256": file_sha256(archive), "archive_top_folder": prefix,
            "raw_folder": str(folder), "members": len(expected), "matched": matched,
            "mismatched": mismatched, "missing_on_disk": missing,
            "extra_on_disk": extra,
            "ok": not mismatched and not missing}


def main():
    require_project_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=["cicids2017", "ciciot2023", "all"],
                    default="all")
    ap.add_argument("--out", default="docs/datasets/archive_check.json")
    args = ap.parse_args()
    # --dataset all (the default) checks both datasets
    datasets = list(ARCHIVES) if args.dataset == "all" else [args.dataset]
    report, failed, checked = {"created": time.strftime("%Y-%m-%d %H:%M:%S")}, False, 0
    for d in datasets:
        archive = ARCHIVES[d]
        # no zip in downloads/: skip this dataset (it is not counted as a failure)
        if not archive.exists():
            print(f"{d}: SKIPPED, {archive} not found")
            report[d] = {"archive": str(archive), "skipped": "archive not found"}
            continue
        try:
            folder = raw_folder(d)
        except (FileNotFoundError, FileExistsError) as exc:
            print(f"{d}: FAILED, {exc}")
            report[d] = {"archive": str(archive), "error": str(exc)}
            failed = True
            continue
        print(f"{d}: comparing {archive} with {folder} ...", flush=True)
        t0 = time.time()
        r = compare(archive, folder)
        r["seconds"] = round(time.time() - t0, 1)
        report[d] = r
        checked += 1
        print(f"{d}: {r['matched']} of {r['members']} files identical to the archive; "
              f"{len(r['mismatched'])} different, {len(r['missing_on_disk'])} missing, "
              f"{len(r['extra_on_disk'])} extra; archive SHA-256 {r['archive_sha256']}")
        for m in r["mismatched"]:
            print(f"  DIFFERENT: {m['file']} ({m['reason']})")
        for m in r["missing_on_disk"]:
            print(f"  MISSING:   {m}")
        for m in r["extra_on_disk"]:
            print(f"  EXTRA (not in the archive): {m}")
        failed |= not r["ok"]
    # write the full report, then stop with exit code 1 if any check failed or
    # no archive was found at all
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Recorded in {out}")
    if failed or checked == 0:
        sys.exit(1)
    print("ARCHIVES_OK")


if __name__ == "__main__":
    main()
