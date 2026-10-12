# Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town 
# 2026

"""Expected source schemas, identifier columns and file locations.

The headers below were read from the downloaded files on 5 October 2026. A
missing, extra or duplicated column stops cleaning with a recorded error
(report Section 4.3.1).

Raw data layout:
    data/raw/cicids2017_improved/monday.csv ... friday.csv
    data/raw/ciciot2023/<34 class folders>/*.pcap.csv   (+ README_CSV.pdf)
"""
# Like label_maps.py, this file only holds fixed facts that the other scripts
# check against: the exact column names of each dataset, which columns are not
# model inputs, which columns cannot be negative, and where the raw files are.
from pathlib import Path

# Paths relative to the project folder; this is why every script must be run
# from /mnt/c/Users/Nyiko/fl-iot-ids (see require_project_root at the end)
RAW = Path("data/raw")
PROCESSED = Path("data/processed")

# The 91 column names of every CIC-IDS2017 day file, in order. clean_data.py
# compares each file's first line with this list and stops if they differ.
CICIDS2017_HEADER = [
    "id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
    "Timestamp", "Flow Duration", "Total Fwd Packet", "Total Bwd packets",
    "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max",
    "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
    "Bwd URG Flags", "Fwd RST Flags", "Bwd RST Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Packet Length Min",
    "Packet Length Max", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWR Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Fwd Segment Size Avg", "Bwd Segment Size Avg", "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "FWD Init Win Bytes", "Bwd Init Win Bytes", "Fwd Act Data Pkts",
    "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "ICMP Code", "ICMP Type",
    "Total TCP Flow Time", "Label", "Attempted Category",
]
# Identifiers, timestamps and labels: kept for tracing and grouping, never inputs.
CICIDS2017_EXCLUDED = {
    "id": "row number in the source file (kept as a check)",
    "Flow ID": "flow identifier (used only as the grouping key)",
    "Src IP": "IP address (device identifier)",
    "Src Port": "ephemeral source port (connection identifier)",
    "Dst Port": ("destination port number: a service identifier stored as a number; "
                 "excluded to avoid testbed shortcuts (decision D4)"),
    "Dst IP": "IP address (device identifier)",
    "Timestamp": "capture time (testbed-specific)",
    "Label": "target label",
    "Attempted Category": "label-derived field",
}
# Columns that cannot be negative. -1 in ICMP Code/Type and in the initial
# window fields means "not applicable" and is kept.
CICIDS2017_NONNEGATIVE = ["Fwd Header Length", "Bwd Header Length"]

# The 39 feature columns of every CIC-IoT2023 per-class file (no label column:
# the class comes from the folder name, see label_maps.py)
CICIOT2023_HEADER = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate", "fin_flag_number",
    "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count",
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP",
    "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max",
    "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
]
CICIOT2023_EXCLUDED = {}          # all 39 columns are traffic features
CICIOT2023_NONNEGATIVE = list(CICIOT2023_HEADER)   # no negative value is valid

# the five CIC-IDS2017 files: data/raw/cicids2017_improved/<day>.csv
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
CICIOT2023_FOLDERS = 34                  # one folder per fine label (benign + 33 attacks)
CICIOT2023_DIR = RAW / "ciciot2023"
CICIOT2023_MARKER = "Benign_Final"        # a class folder present in every copy

# Files that end part-way through a line in the published CIC-IoT2023 archive
# (CSV.zip from the official page). Downloads on 4 and 5 October 2026 gave the
# same sizes and the same incomplete last line, and the extracted files match
# the archive's CRC-32 values, so this is a property of the release, not of
# the download. Cleaning drops the incomplete last line of these files only if
# size and SHA-256 match, and records it. Any other file that ends part-way
# through a line still stops cleaning (or needs --allow-truncated).
CICIOT2023_KNOWN_TRUNCATED = {
    # folder/file: (bytes, sha256)
    "DoS-UDP_Flood/DoS-UDP_Flood7.pcap.csv": (
        35696059, "a7a073e44053713b18938cfad4a8bf762be774a1f5331dad1c0e7bba7e8628bd"),
    "DoS-UDP_Flood/DoS-UDP_Flood8.pcap.csv": (
        32624109, "8e24fee423564aa9622c12f6380570cb2c14331d3fcd703d9dfca102b3e7f7e1"),
    "DoS-UDP_Flood/DoS-UDP_Flood9.pcap.csv": (
        25251429, "eefba41ed734d8829e07a32e7189d77a11427f47b1b9653f94b1263cfbd42638"),
}


def ciciot2023_base(top=None):
    """Folder that directly contains the CIC-IoT2023 class folders.

    Accepts data/raw/ciciot2023/<class folders> (current layout) or one extra
    level, e.g. data/raw/ciciot2023/CIC-IoT2023/<class folders>. Stops if no
    copy or more than one copy is found, so records can never be read twice.
    """
    top = Path(top) if top is not None else CICIOT2023_DIR
    # look in data/raw/ciciot2023 itself and in each folder directly inside it
    # for a folder called Benign_Final; where it is found, the class folders are
    if not top.is_dir():
        raise FileNotFoundError(f"Folder not found: {top}")
    candidates = [top] + sorted(p for p in top.iterdir() if p.is_dir())
    found = [p for p in candidates if (p / CICIOT2023_MARKER).is_dir()]
    if not found:
        raise FileNotFoundError(
            f"No CIC-IoT2023 class folders found in {top} (expected for example "
            f"{top / CICIOT2023_MARKER})")
    if len(found) > 1:
        raise FileExistsError(
            "More than one copy of the CIC-IoT2023 class folders: "
            f"{[str(p) for p in found]}. Keep exactly one copy under {top}.")
    return found[0]


def known_truncation(path):
    """(bytes, sha256) if path is a file known to be truncated in the release."""
    path = Path(path)
    # look up "<folder>/<file name>" in the table above; None if not listed
    return CICIOT2023_KNOWN_TRUNCATED.get(f"{path.parent.name}/{path.name}")


def schema(dataset):
    # the three schema facts of one dataset, as one dictionary
    if dataset == "cicids2017":
        return {"header": CICIDS2017_HEADER, "excluded": CICIDS2017_EXCLUDED,
                "nonnegative": CICIDS2017_NONNEGATIVE}
    if dataset == "ciciot2023":
        return {"header": CICIOT2023_HEADER, "excluded": CICIOT2023_EXCLUDED,
                "nonnegative": CICIOT2023_NONNEGATIVE}
    raise ValueError(dataset)


def require_project_root():
    """Stop with a clear message unless run from the project root."""
    # the project root is the folder that contains both data/raw and src
    if not (RAW.is_dir() and Path("src").is_dir()):
        raise SystemExit(
            "Run this from the project root, for example:\n"
            "  cd /mnt/c/Users/Nyiko/fl-iot-ids\n"
            "  source .venv/bin/activate\n"
            f"(current folder: {Path.cwd()})")
