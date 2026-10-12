# Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula
# EEE4022S final-year project
# University of Cape Town 
# 2026

"""Explicit label mappings for both datasets (report Section 3.3).

Every source label must appear here. An unknown label stops the pipeline and
is written to cleaning_error.json instead of being silently dropped or merged.
A mapping is changed only after agreement with the supervisor, and every
change is recorded in Chapter 3 of the report.
"""
# This file holds no code that reads data. It only lists, for each dataset, the
# task classes and which source label goes into which class. clean_data.py uses
# it to turn the original labels into class names and writes the class order to
# data_manifest.json; build_arrays.py then numbers the classes in that order
# (Benign = 0, then 1, 2, ...).

# ---------------------------------------------------------------- CIC-IDS2017
# Seven task classes. "None" means excluded from the main task (counted in the
# data audit). "- Attempted" labels are handled by the --attempted policy in
# clean_data.py: "benign" relabels them as Benign (the dataset authors'
# recommendation), "exclude" removes them.
# The order of this list sets the class numbers: Benign = 0, DoS = 1, ... Botnet = 6
CICIDS2017_CLASSES = [
    "Benign", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Botnet",
]
# source label (exactly as written in the CSV "Label" column) -> task class
CICIDS2017_MAP = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS Slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "Portscan": "PortScan",
    "Infiltration - Portscan": "PortScan",   # decision D2 (see README.md)
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - SQL Injection": "WebAttack",
    "Botnet": "Botnet",
    "Heartbleed": None,      # 11 confirmed records: too rare for 105 gateways
    "Infiltration": None,    # 36 confirmed records: too rare for 105 gateways
}
# end of the label of a failed attack attempt, e.g. "DoS Hulk - Attempted"
ATTEMPTED_SUFFIX = " - Attempted"

# ---------------------------------------------------------------- CIC-IoT2023
# Benign traffic plus the seven attack categories of Neto et al. (2023).
# Class numbers: Benign = 0, DDoS = 1, ... BruteForce = 7
CICIOT2023_CLASSES = [
    "Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce",
]
# Fine label (upper case, as written in the merged files) -> category.
CICIOT2023_MAP = {
    "BENIGN": "Benign",
    "DDOS-ACK_FRAGMENTATION": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS",
    "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-SLOWLORIS": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS",
    "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DOS-HTTP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS",
    "DOS-TCP_FLOOD": "DoS",
    "DOS-UDP_FLOOD": "DoS",
    "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
    "RECON-HOSTDISCOVERY": "Recon",
    "RECON-OSSCAN": "Recon",
    "RECON-PINGSWEEP": "Recon",
    "RECON-PORTSCAN": "Recon",
    "VULNERABILITYSCAN": "Recon",
    "DNS_SPOOFING": "Spoofing",
    "MITM-ARPSPOOFING": "Spoofing",
    "BACKDOOR_MALWARE": "Web",
    "BROWSERHIJACKING": "Web",
    "COMMANDINJECTION": "Web",
    "SQLINJECTION": "Web",
    "UPLOADING_ATTACK": "Web",
    "XSS": "Web",
    "DICTIONARYBRUTEFORCE": "BruteForce",
}
# Per-class folder names differ in case (DDoS-PSHACK_FLOOD) and the benign
# folder is called Benign_Final.
FOLDER_ALIASES = {"BENIGN_FINAL": "BENIGN"}


def ciciot_folder_to_fine(folder_name: str) -> str:
    # CIC-IoT2023 has no label column in the per-class files: the folder name is
    # the label. Example: folder "DDoS-PSHACK_Flood" -> "DDOS-PSHACK_FLOOD".
    # An unknown folder name stops the pipeline (ValueError).
    key = folder_name.strip().upper()
    key = FOLDER_ALIASES.get(key, key)
    if key not in CICIOT2023_MAP:
        raise ValueError(f"Unmapped CIC-IoT2023 folder: {folder_name!r}")
    return key


# one entry per dataset, so other files can write DATASETS[dataset]["classes"]
DATASETS = {
    "cicids2017": {"classes": CICIDS2017_CLASSES, "map": CICIDS2017_MAP},
    "ciciot2023": {"classes": CICIOT2023_CLASSES, "map": CICIOT2023_MAP},
}


def mapping_table(dataset: str, attempted: str = "benign"):
    """Rows of (source label, task class or 'excluded', note) for the manifest."""
    # clean_data.py stores these rows as "label_mapping" in data_manifest.json
    rows = []
    if dataset == "cicids2017":
        for label, cls in CICIDS2017_MAP.items():
            rows.append((label, cls or "excluded", "" if cls else "too rare"))
        # one extra row that says how "- Attempted" labels were treated
        rows.append((f"<any>{ATTEMPTED_SUFFIX}",
                     "Benign" if attempted == "benign" else "excluded",
                     f"attempted policy = {attempted}"))
    else:
        for label, cls in CICIOT2023_MAP.items():
            rows.append((label, cls, ""))
    return rows
