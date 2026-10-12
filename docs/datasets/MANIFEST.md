# Dataset manifest

Project: Federated Learning-Based Intrusion Detection for Distributed IoT Networks (Nyiko Mathebula, UCT).
Written 6 October 2026 from the files in this folder. The machine-readable
records are `archive_check.json`, the `*_sha256.txt` files and `line_counts.txt` here, and
`data/processed/<dataset>/data_manifest.json` once `clean_data.py` has run.

## 1. Corrected CIC-IDS2017

| Field | Value |
|---|---|
| Source | Corrected ("improved") CIC-IDS2017 release, Distrinet group, KU Leuven |
| Citation | G. Engelen, V. Rimmer and W. Joosen, "Troubleshooting an intrusion detection dataset: the CICIDS2017 case study," IEEE Security and Privacy Workshops (SPW), 2021; L. Liu, G. Engelen, T. Lynar, D. Essam and W. Joosen, "Error prevalence in NIDS datasets: a case study on CIC-IDS-2017 and CSE-CIC-IDS-2018," IEEE Conference on Communications and Network Security (CNS), 2022 |
| Download page | https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html |
| Downloaded | 2 October 2026 |
| Archive | `downloads/CICIDS2017_improved.zip`, 343,549,013 bytes, SHA-256 `97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883` |
| Copy used | `data/raw/cicids2017_improved/`: `monday.csv` ... `friday.csv` (5 files) |
| Archive check | 6 October 2026: 5 of 5 files identical to the archive (size and CRC-32), `ARCHIVES_OK` |
| Records | 2,099,976 flows (2,099,981 lines including 5 headers) |

File checksums (from `cicids2017_sha256.txt`):

| File | SHA-256 |
|---|---|
| monday.csv | `51fe5dc962626efb4ae70dce0303072fb780da0932822b651202ee9c2fbc1aff` |
| tuesday.csv | `e2a0a5b631dfc6b455cc9f9a88b944110637d70a7f74171473925f76f38b6b0c` |
| wednesday.csv | `bf46c5f3c792e8817381f724511229569606918eaf07ac986d7a2592b6341bc2` |
| thursday.csv | `78a4d11eaf473d099e30e71ddb01e0f38218e844c0a9cdd36602145d674af482` |
| friday.csv | `ebd499e6f23bd59f9cb81bec28178491b02b925fa5640a24215c9437d79482d0` |

Label mapping (`src/label_maps.py`):

| Source labels | Task class | Decision |
|---|---|---|
| BENIGN | Benign | |
| any label ending in " - Attempted" (11,979 flows) | Benign; original label kept in the record index | D1 |
| DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest | DoS | |
| DDoS | DDoS | |
| Portscan, Infiltration - Portscan | PortScan | D2 |
| FTP-Patator, SSH-Patator | BruteForce | |
| Web Attack - Brute Force, Web Attack - XSS, Web Attack - SQL Injection | WebAttack | |
| Botnet | Botnet | |
| Heartbleed (11), Infiltration (36) | excluded and counted | D3 |

Inputs: all flow statistics except the identifiers, timestamps and labels, and except the
destination port (decision D4); `Protocol` is kept. 82 inputs after cleaning (81 for seed 1618,
where `Bwd URG Flags` is constant in the training portion).

## 2. CIC-IoT2023

| Field | Value |
|---|---|
| Source | Canadian Institute for Cybersecurity, University of New Brunswick: CIC-IoT2023, per-class CSV release |
| Citation | E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu and A. A. Ghorbani, "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," Sensors, vol. 23, no. 13, 5941, 2023 |
| Download page | https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| Downloaded | 5 October 2026 (first download 4 October; same files) |
| Archive | `downloads/CSV.zip`, 1,430,268,604 bytes, SHA-256 `e211e878d2f39226ea3a854683f91ae6175b628807ee96eeecb7a04a36a28dbd` |
| Copy used | `data/raw/ciciot2023/`: 34 class folders (benign and 33 attacks), 309 CSV files, and `README_CSV.pdf`; the merged copy was deleted on 5 October |
| Archive check | 6 October 2026: 310 of 310 files identical to the archive (size and CRC-32), `ARCHIVES_OK` |
| Records | 46,776,697 complete records and 39 packet-window features; the class is given by the folder |
| File checksums | `ciciot2023_sha256.txt` (310 lines); line counts in `line_counts.txt` |

Known issue in the published archive: three files end part-way through a record. The
incomplete last line is removed by `clean_data.py` (listed with size and SHA-256 in
`src/schema.py`); the files may also lack records after that point, so DoS-UDP_Flood is
probably somewhat under-represented.

| File in `DoS-UDP_Flood/` | Size (bytes) | Fields in last line | SHA-256 |
|---|---|---|---|
| DoS-UDP_Flood7.pcap.csv | 35,696,059 | 33 of 39 | `a7a073e44053713b18938cfad4a8bf762be774a1f5331dad1c0e7bba7e8628bd` |
| DoS-UDP_Flood8.pcap.csv | 32,624,109 | 35 of 39 | `8e24fee423564aa9622c12f6380570cb2c14331d3fcd703d9dfca102b3e7f7e1` |
| DoS-UDP_Flood9.pcap.csv | 25,251,429 | 4 of 39 | `eefba41ed734d8829e07a32e7189d77a11427f47b1b9653f94b1263cfbd42638` |

Label mapping (`src/label_maps.py`):

| Task class | Folders |
|---|---|
| Benign | Benign_Final |
| DDoS (12) | DDoS-ACK_Fragmentation, DDoS-HTTP_Flood, DDoS-ICMP_Flood, DDoS-ICMP_Fragmentation, DDoS-PSHACK_FLOOD, DDoS-RSTFINFLOOD, DDoS-SlowLoris, DDoS-SynonymousIP_Flood, DDoS-SYN_Flood, DDoS-TCP_Flood, DDoS-UDP_Flood, DDoS-UDP_Fragmentation |
| DoS (4) | DoS-HTTP_Flood, DoS-SYN_Flood, DoS-TCP_Flood, DoS-UDP_Flood |
| Mirai (3) | Mirai-greeth_flood, Mirai-greip_flood, Mirai-udpplain |
| Recon (5) | Recon-HostDiscovery, Recon-OSScan, Recon-PingSweep, Recon-PortScan, VulnerabilityScan |
| Spoofing (2) | DNS_Spoofing, MITM-ArpSpoofing |
| Web (6) | Backdoor_Malware, BrowserHijacking, CommandInjection, SqlInjection, Uploading_Attack, XSS |
| BruteForce (1) | DictionaryBruteForce |

## 3. Processing decisions (full list D1-D16 in `README.md`)

- D1-D3: label policy above.
- D4: CIC-IDS2017 destination port excluded from the inputs.
- D5: exact duplicates and conflicting feature vectors removed before splitting (first copy kept; conflicts judged on the task class).
- D6: signed logarithm, then standardisation with training-portion statistics; constant features dropped per seed.
- D7: 70/10/20 split stratified by source label; Flow ID groups per day for CIC-IDS2017, record level for CIC-IoT2023; seeds 13, 42, 271, 1618, 3141.
- D8: CIC-IoT2023 training portion subsampled per source label to min(n, max(ceil(0.05 n), 10,000)); validation and test complete.
- D9: 105 gateways, Dirichlet alpha in {100, 10, 1, 0.5, 0.1}, minimums 512/74/147 with the top-up rule.

Counts after cleaning are written to `data/processed/<dataset>/data_manifest.json` and
`data_audit.csv` (logs: `docs/clean_cicids2017.log`, `docs/clean_ciciot2023.log`).
Cleaning on 7 October 2026 gave:

| | CIC-IDS2017 | CIC-IoT2023 |
|---|---|---|
| Rows read | 2,099,976 | 46,776,700 |
| Exact duplicates removed | 382,467 | 12,554,483 |
| Conflicting records removed | 2,678 | 14,056,979 |
| Other removals (rare class, invalid or negative value, incomplete line) | 637 | 1,042 |
| Records kept | 1,714,194 | 20,164,196 |
