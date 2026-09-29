# WARDEN-T Dataset Reconnaissance Report

**Date:** September 25, 2026  
**Project:** WARDEN-T — Temporal Transformer for Network Attack Forecasting  
**Task:** Dataset Reconnaissance Only (Task 1)

---

## EXECUTIVE SUMMARY

After systematic investigation of all publicly available cybersecurity datasets specified in Problem Statement ID 26153, **CIC-IDS2017** is recommended as the primary MVP dataset, with **UNSW-NB15** as the optional secondary dataset for cross-validation.

The recommendation is based on:
- Rich temporal information (timestamps on every flow record)
- Complete network flow features (source/dest IP, ports, protocol, TCP flags, bytes, packets, duration, IAT statistics)
- 15+ distinct attack categories with temporal ordering
- PCAP files available for packet-level feature extraction
- Moderate size (~51 GB with PCAPs, ~2.8M rows CSV-only) that is feasible on a laptop/Colab
- Well-documented attack schedules enabling temporal attack progression analysis

---

## TASK 1: DATASET RECONNAISSANCE

### 1. CIC-IDS2017

| Criterion | Finding |
|-----------|---------|
| **A. Name** | CIC-IDS2017 (Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017) |
| **B. Official Source** | https://www.unb.ca/cic/datasets/ids-2017.html |
| **C. Download Format** | Downloaded via `http://cicresearch.ca/CICDataset/CIC-IDS-2017/` — contains PCAP files + CSV flow files (MachineLearningCSV.zip, GeneratedLabelledFlows.zip) |
| **D. Approximate Size** | ~51 GB total (5 days × ~10 GB/day). CSV-only subset is ~2.8M rows, ~500 MB compressed |
| **E. Timestamps** | **YES.** Each CSV flow record contains a `Timestamp` column. Flows are ordered chronologically. The dataset spans Monday July 3 to Friday July 7, 2017, 9 AM to 5 PM each day. |
| **F. Source/Dest IPs** | **YES.** `Source IP` and `Destination IP` columns present in all CSV files |
| **G. Source/Dest Ports** | **YES.** `Source Port` and `Destination Port` columns present |
| **H. Protocol** | **YES.** `Protocol` column (TCP, UDP, ICMP, etc.) |
| **I. TCP Flags** | **YES.** `FIN Flag Count`, `SYN Flag Count`, `RST Flag Count`, `PSH Flag Count`, `ACK Flag Count`, `URG Flag Count`, `CWE Flag Count`, `ECE Flag Count` |
| **J. Packet/Byte Counts** | **YES.** `Total Fwd Packets`, `Total Bwd Packets`, `Total Length of Fwd Packets`, `Total Length of Bwd Packets`, `Flow Bytes/s`, `Flow Packets/s` |
| **K. Flow Duration** | **YES.** `Flow Duration` column (in seconds) |
| **L. Inter-Arrival-Time** | **YES.** `Flow IAT Mean/Std/Max/Min`, `Fwd IAT Total/Mean/Std/Max/Min`, `Bwd IAT Total/Mean/Std/Max/Min` |
| **M. Attack Labels** | **YES.** 15 attack categories + BENIGN: DoS Hulk, PortScan, DDoS, DoS GoldenEye, FTP-Patator, SSH-Patator, DoS slowloris, DoS Slowhttptest, Bot, Web Attack-Brute Force, Web Attack-XSS, Infiltration, Web Attack-SQL Injection, Heartbleed |
| **N. Temporal Attack Progression** | **YES.** Attacks are scheduled on specific days with documented time windows: Tuesday (Brute Force), Wednesday (DoS/DDoS/Heartbleed), Thursday (Web Attacks/Infiltration), Friday (Botnet/PortScan/DDoS). Monday is benign-only. |
| **O. PCAP Availability** | **YES.** Raw PCAP files for all 5 days (Monday–Friday) |
| **P. Packet-Level Features** | **YES.** PCAPs can be processed with CICFlowMeter or other tools to extract TTL, TCP window size, fragment flags, payload size distribution, retransmission counts |
| **Q. Laptop Feasibility** | **CONDITIONAL.** Full dataset with PCAPs is ~51 GB — challenging on a laptop. CSV-only subset (~2.8M rows, ~500 MB) is feasible on a laptop or Google Colab. A ~101k-row demonstration slice exists on Hugging Face for testing |
| **R. Temporal Transformer Suitability** | **HIGH.** Every flow has a timestamp and ordered sequence. 80+ features per flow enable rich state vectors S_t. Temporal attack progression is clearly documented, enabling multi-step forecasting |

**Key Advantage for WARDEN-T:** CIC-IDS2017 provides 80+ CICFlowMeter-extracted features per flow, with timestamps enabling chronological ordering. The attack schedule is explicitly documented (specific days, specific hours), making it possible to construct sequential windows S(t-9)...S(t) and predict future attack states. The bidirection flow design (fwd/bwd separately computed) gives richer state representations.

**Known Caveats:** Documented labeling issues in original CICFlowMeter output (some TCP flow termination bugs). Researchers have published corrected versions. Some rows contain inf/NaN values requiring cleaning.

---

### 2. CIC-IDS2018 (CSE-CIC-IDS2018)

| Criterion | Finding |
|-----------|---------|
| **A. Name** | CSE-CIC-IDS2018 |
| **B. Official Source** | https://www.unb.ca/cic/datasets/ids-2018.html / AWS S3 (`s3://cse-cic-ids2018/`) |
| **C. Download Format** | `aws s3 sync --no-sign-request --region <region> "s3://cse-cic-ids2018/" dest-dir` |
| **D. Approximate Size** | Very large — 50 attack machines, 420 victim machines, 7 days of traffic. Estimated 100+ GB |
| **E. Timestamps** | **YES.** Flow records have timestamps; attack schedule documented per day per hour |
| **F. Source/Dest IPs** | **YES.** SourceIP, DestinationIP columns |
| **G. Source/Dest Ports** | **YES.** SourcePort, DestinationPort columns |
| **H. Protocol** | **YES.** Protocol column |
| **I. TCP Flags** | **YES.** 83 features including fin_cnt, syn_cnt, rst_cnt, psh_cnt, ack_cnt, urg_cnt, cwe_cnt, ece_cnt |
| **J. Packet/Byte Counts** | **YES.** tot_fw_pk, tot_bw_pk, tot_l_fw_pkt, tot_bwd_pkt, etc. |
| **K. Flow Duration** | **YES.** fl_dur |
| **L. Inter-Arrival-Time** | **YES.** fl_iat_avg/std/max/min, fw_iat_avg/std/max/min, bw_iat_avg/std/max/min |
| **M. Attack Labels** | **YES.** 7 attack types: Brute Force, Heartbleed, Botnet, DoS, DDoS, Web Attacks, Infiltration |
| **N. Temporal Attack Progression** | **YES.** Detailed schedule: Feb 14–Mar 2, 2018. Each attack has documented start/finish times |
| **O. PCAP Availability** | **YES.** Raw PCAPs plus Windows/Ubuntu event logs per machine |
| **P. Packet-Level Features** | **YES.** PCAPs + system logs available |
| **Q. Laptop Feasibility** | **LOW.** Dataset is massive — 50 attack machines × 420 victim machines × 7 days. AWS S3 sync required. Estimated 100+ GB. Not practical for a laptop or Colab |
| **R. Temporal Transformer Suitability** | **HIGH in theory, impractical for prototype.** Same feature richness as CIC-IDS2017 but at much larger scale |

**Key Limitation for 3-Day Prototype:** The sheer scale (100+ GB, 50+ machines, 7 days) makes this impractical for a laptop/Colab prototype. It requires AWS infrastructure. The instruction to not immediately choose CIC-IDS2018 was heeded.

---

### 3. UNSW-NB15

| Criterion | Finding |
|-----------|---------|
| **A. Name** | UNSW-NB15 |
| **B. Official Source** | https://research.unsw.edu.au/projects/unsw-nb15-dataset (404 on direct URL; available via UNSW CIC page) |
| **C. Download Format** | ZIP file from UNSW website containing: 4 CSV files, PCAP files (via IXIA PerfectStorm), BRO files, Argus files |
| **D. Approximate Size** | ~2.5 GB compressed (4 CSV files); ~100 GB raw PCAPs |
| **E. Timestamps** | **YES.** `Stime` (record start time) and `Ltime` (record last time) as Timestamp columns |
| **F. Source/Dest IPs** | **YES.** `srcip`, `dstip` |
| **G. Source/Dest Ports** | **YES.** `sport`, `dsport` |
| **H. Protocol** | **YES.** `proto` |
| **I. TCP Flags** | **NO (not directly).** Has `state` column (TCP state like ACC, CLO, CON, FIN, RST, etc.) but individual TCP flag counts (SYN, ACK, FIN counts) are NOT present. The `state` field indicates connection state but is not a granular flag count |
| **J. Packet/Byte Counts** | **YES.** `spkts`, `dpkts`, `sbytes`, `dbytes`, `sload`, `dload` |
| **K. Flow Duration** | **YES.** `dur` |
| **L. Inter-Arrival-Time** | **YES.** `Sintpkt`, `Dintpkt`, `Sjit`, `Djit` (inter-packet arrival time stats) |
| **M. Attack Labels** | **YES.** 9 attack categories: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms + Normal |
| **N. Temporal Attack Progression** | **CONDITIONAL.** Records are ordered by `Stime` but attack events are not organized into explicit daily schedules with documented attack windows like CIC-IDS2017. The simulation period was Jan 22–Feb 17, 2015 across two phases |
| **O. PCAP Availability** | **YES.** Raw PCAP files from IXIA PerfectStorm |
| **P. Packet-Level Features** | **YES.** PCAPs, BRO files, and Argus files available for detailed extraction |
| **Q. Laptop Feasibility** | **CONDITIONAL.** CSV-only (2.5M records, ~2.5 GB compressed) is feasible. Full PCAPs (~100 GB) are not |
| **R. Temporal Transformer Suitability** | **MODERATE-HIGH.** Timestamps and ordered records enable temporal modelling. However, the lack of explicit daily attack schedules makes it harder to define attack-stage progression. 49 features are comprehensive but TCP flag granularity is lower than CIC datasets |

**Key Limitation:** The `state` field provides TCP connection state but NOT individual TCP flag counts (SYN count, ACK count, FIN count, etc.) which are critical for detecting port scans and certain attack patterns. This is a significant gap compared to CIC-IDS2017/2018.

---

### 4. CTU-13

| Criterion | Finding |
|-----------|---------|
| **A. Name** | CTU-13 (Botnet Dataset) |
| **B. Official Source** | https://www.stratosphereips.org/datasets-ctu13 |
| **C. Download Format** | `CTU-13-Dataset.tar.bz2` (1.9 GB) from mcfp.felk.cvut.cz. Individual scenarios available separately |
| **D. Approximate Size** | ~1.9 GB compressed; ~6+ GB uncompressed |
| **E. Timestamps** | **YES.** Binetflow files contain timestamps for each flow |
| **F. Source/Dest IPs** | **YES.** In binetflow files |
| **G. Source/Dest Ports** | **YES.** In binetflow files |
| **H. Protocol** | **YES.** TCP, UDP, ICMP |
| **I. TCP Flags** | **NO.** Binetflow files contain `state` (TCP connection state) but not individual flag counts |
| **J. Packet/Byte Counts** | **YES.** Packet and byte counts in binetflow |
| **K. Flow Duration** | **YES.** Duration field |
| **L. Inter-Arrival-Time** | **PARTIAL.** Can be computed from timestamps but not directly provided |
| **M. Attack Labels** | **YES.** Botnet, Normal, Background, C&C labels per flow |
| **N. Temporal Attack Progression** | **YES.** 13 scenarios captured on specific dates in August 2011. Each scenario represents a different malware botnet |
| **O. PCAP Availability** | **YES.** Botnet-only PCAP files for each scenario (full PCAP with background/normal/botnet is NOT public due to privacy) |
| **P. Packet-Level Features** | **YES.** Botnet PCAP files available; TShark can extract detailed features |
| **Q. Laptop Feasibility** | **HIGH.** 1.9 GB total, manageable on a laptop |
| **R. Temporal Transformer Suitability** | **MODERATE.** The dataset focuses exclusively on botnet traffic. It does NOT contain diverse attack types (DoS, web attacks, infiltration, brute force). The temporal scope is limited to 13 botnet scenarios, not a continuous network operation timeline. The binetflow files are per-scenario, not a unified chronological sequence |

**Key Limitation:** CTU-13 is a botnet-specific dataset. It lacks the diversity of attack types needed for WARDEN-T's attack forecasting objective. It does not model a "normal network operating" baseline with various attack types interleaved. The scenarios are isolated captures, not a continuous temporal stream.

---

### 5. CICIoT2023

| Criterion | Finding |
|-----------|---------|
| **A. Name** | CICIoT2023 |
| **B. Official Source** | https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| **C. Download Format** | ZIP from CIC website; ~2.7 GB compressed, ~13 GB uncompressed. 169 individual CSV files |
| **D. Approximate Size** | ~2.7 GB compressed / ~13 GB uncompressed CSV |
| **E. Timestamps** | **YES.** `ts` (timestamp) column present in CSV features |
| **F. Source/Dest IPs** | **NO.** The extracted CSV features do NOT contain source IP or destination IP columns. The features are aggregate flow statistics |
| **G. Source/Dest Ports** | **NO.** No source/destination port columns in the CSV features |
| **H. Protocol** | **YES.** `Protocol Type` (integer), `TCP`, `UDP`, `ICMP`, `HTTP`, `HTTPS`, `DNS`, etc. flags |
| **I. TCP Flags** | **YES.** `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number` |
| **J. Packet/Byte Counts** | **YES.** `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `Number`, `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`, `Rate`, `Srate`, `Drate` |
| **K. Flow Duration** | **YES.** `flow_duration` |
| **L. Inter-Arrival-Time** | **YES.** `IAT` (inter-arrival time) |
| **M. Attack Labels** | **YES.** 33 attack types across 7 categories: DDoS, DoS, Recon, Web-based, Brute Force, Spoofing, Mirai |
| **N. Temporal Attack Progression** | **CONDITIONAL.** Timestamps exist but attacks are organized by scenario/file, not as a unified chronological timeline. Features are extracted in fixed-size packet windows (10 or 100 packets) |
| **O. PCAP Availability** | **YES.** Raw PCAP files in the `PCAP/` directory |
| **P. Packet-Level Features** | **YES.** PCAPs available; features extracted via DPKT |
| **Q. Laptop Feasibility** | **CONDITIONAL.** 13 GB uncompressed CSV is heavy for a laptop. Can be subsampled. ~46M records total |
| **R. Temporal Transformer Suitability** | **MODERATE.** Features are extracted in fixed packet windows rather than flow-based. The lack of source/destination IP/port means network state identity is lost — you cannot reconstruct who is communicating with whom. The features are more "per-packet-window statistics" than "per-flow state". This makes it harder to construct a meaningful S_t representing network state |

**Key Limitation:** The pre-extracted CSV features do NOT include source/destination IP addresses or ports. This is a critical gap for WARDEN-T, which needs to represent network state including endpoint identity. The features are aggregated packet-window statistics, losing the bidirectional flow structure that CIC-IDS2017 provides.

---

### 6. LANL Authentication Dataset

| Criterion | Finding |
|-----------|---------|
| **A. Name** | LANL Comprehensive, Multi-Source Cyber-Security Events / LANL Authentication Dataset |
| **B. Official Source** | https://csr.lanl.gov/data/cyber1/ and https://csr.lanl.gov/data/auth/ |
| **C. Download Format** | Manual download request (email required). Files: auth.txt.gz (7.2 GB), flows.txt.gz (1.1 GB), proc.txt.gz (2.2 GB), dns.txt.gz (177 MB), redteam.txt.gz (4.8 KB) |
| **D. Approximate Size** | ~12 GB compressed across 5 data elements; 1.65 billion events total |
| **E. Timestamps** | **YES.** Epoch timestamps with 1-second resolution |
| **F. Source/Dest IPs** | **NO (de-identified).** Network flow data uses `source computer` and `destination computer` identifiers (de-ified). Authentication data uses `user@domain` not IP addresses |
| **G. Source/Dest Ports** | **YES.** `source port`, `destination port` in flows.txt.gz |
| **H. Protocol** | **YES.** `protocol` field in flows.txt.gz |
| **I. TCP Flags** | **NO.** Flow records contain: time, duration, source/dest computer, source/dest port, protocol, packet count, byte count. No TCP flag information |
| **J. Packet/Byte Counts** | **YES.** `packet count`, `byte count` in flows.txt.gz |
| **K. Flow Duration** | **YES.** `duration` field |
| **L. Inter-Arrival-Time** | **NO.** Not directly provided; would need to be computed from sequential timestamps |
| **M. Attack Labels** | **YES.** `redteam.txt.gz` contains 4.8 KB of red teaming event descriptions. Authentication failures and process anomalies are present |
| **N. Temporal Attack Progression** | **CONDITIONAL.** 58 days of data. Red team events are documented but the dataset is primarily authentication-centric, not network-traffic-centric |
| **O. PCAP Availability** | **NO.** No PCAP files |
| **P. Packet-Level Features** | **NO.** Only flow-level summary data |
| **Q. Laptop Feasibility** | **LOW.** 12+ GB compressed; requires manual request; primarily authentication data |
| **R. Temporal Transformer Suitability** | **LOW-MODERATE.** The LANL dataset focuses on authentication events and process events, NOT network traffic flow features. It lacks TCP flags, packet/flow-level detail. It is more suitable for insider-threat detection than network attack forecasting |

**Key Limitation:** This is primarily an authentication/process dataset, not a network traffic dataset. It lacks the flow-level features (TCP flags, packet counts per direction, IAT statistics) needed for WARDEN-T. The de-identification removes source/destination IP addresses which are critical for network state representation.

---

### 7. DARPA Intrusion Detection Dataset (1998/1999)

| Criterion | Finding |
|-----------|---------|
| **A. Name** | DARPA Intrusion Detection Evaluation Dataset (1998/1999) |
| **B. Official Source** | https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset |
| **C. Download Format** | Tcpdump data, BSM audit data, file system dumps — distributed as tar/gzip files. Also available via UCI Machine Learning Repository (KDD Cup 1999) |
| **D. Approximate Size** | ~30 CD-ROMs for full training data (7 weeks). The KDD Cup 10% subset is ~75 MB uncompressed (5M records) |
| **E. Timestamps** | **PARTIAL.** Connection records have `timestamp` but the connection-level data from KDD Cup 1999 does not include precise timestamps per connection in the standard format |
| **F. Source/Dest IPs** | **NO.** The KDD Cup 1999 version (most commonly used) does NOT contain IP addresses — it uses anonymized identifiers |
| **G. Source/Dest Ports** | **NO.** Uses service names (http, telnet, ftp, etc.) not port numbers |
| **H. Protocol** | **YES.** `protocol_type` (tcp, udp, icmp) |
| **I. TCP Flags** | **NO.** Has a `flag` column (single categorical: normal, SF, S0, RE, RQ, etc.) but not individual flag counts |
| **J. Packet/Byte Counts** | **YES.** `src_bytes`, `dst_bytes` |
| **K. Flow Duration** | **YES.** `duration` |
| **L. Inter-Arrival-Time** | **NO.** Not present in the standard dataset |
| **M. Attack Labels** | **YES.** 24 training attack types + 14 test-only types. Categories: DoS, Probe, R2L, U2R |
| **N. Temporal Attack Progression** | **CONDITIONAL.** 7 weeks of training data. Attack types are documented but not organized with precise hourly schedules |
| **O. PCAP Availability** | **YES.** Raw tcpdump data available (38 MB gzip for the 4-hour subset) |
| **P. Packet-Level Features** | **YES.** Tcpdump data can be processed |
| **Q. Laptop Feasibility** | **HIGH.** KDD Cup 10% subset (5M records, ~75 MB) is very feasible |
| **R. Temporal Transformer Suitability** | **LOW-MODERATE.** The dataset is from 1998 — the attack types and network traffic patterns are severely outdated. No source/dest IPs or ports in the commonly used format. The feature set (41 features) is much smaller than modern datasets. The temporal structure is not designed for sequential window modelling |

**Key Limitation:** This dataset is from 1998/1999 — it does not reflect modern network traffic patterns, attack techniques, or protocols. The KDD Cup version (most commonly used) lacks IP addresses and port numbers entirely. It is primarily relevant for historical reference.

---

### 8. MITRE ATT&CK

| Criterion | Finding |
|-----------|---------|
| **A. Name** | MITRE ATT&CK® Framework |
| **B. Official Source** | https://attack.mitre.org/ |
| **C. Download Format** | STIX 2.1 JSON data via GitHub (mitre-attack/attack-stix-data). Free public access |
| **D. Approximate Size** | ~100+ MB of STIX JSON data |
| **E. Temporal Information** | **NO.** ATT&CK describes tactics and techniques, not temporal sequences |
| **F–L. Network Features** | **N/A.** Not a network traffic dataset |
| **M. Attack Labels** | **YES.** Comprehensive taxonomy: 14 Tactics → 190+ Techniques → 400+ Sub-techniques |
| **N. Attack Progression** | **YES (conceptual).** Tactics are ordered in a kill chain: Reconnaissance → Initial Access → Execution → Persistence → Privilege Escalation → Defense Evasion → Credential Access → Discovery → Lateral Movement → Collection → Command & Control → Exfiltration → Impact |
| **O. PCAPs** | **NO** |
| **P. Packet Features** | **NO** |
| **Q. Laptop Feasibility** | **HIGH.** Just JSON files |
| **R. Temporal Transformer Suitability** | **NOT A DATASET — Knowledge Base.** ATT&CK is used as a secondary mapping layer to correlate predicted attack behavior to ATT&CK tactics/techniques |

**Use for WARDEN-T:** MITRE ATT&CK is not a training dataset. It is a **knowledge base** that maps observed/predicted network behavior to attack stages (tactics). It can be used to:
1. Map predicted attack probability to specific ATT&CK tactics
2. Provide interpretable evidence: "The predicted S(t+1) shows patterns consistent with Lateral Movement (T1021) and Discovery (T1082)"
3. Define the attack-stage labels for the forecasting task

---

### 9. CAPEC (Common Attack Pattern Enumeration and Classification)

| Criterion | Finding |
|-----------|---------|
| **A. Name** | CAPEC™ |
| **B. Official Source** | https://capec.mitre.org/ |
| **C. Download Format** | XML/HTML catalog from capec.mitre.org/data/ |
| **D. Approximate Size** | ~559 attack patterns |
| **E–L. Network Features** | **N/A.** Not a network traffic dataset |
| **M. Attack Labels** | **YES.** 559 attack patterns categorized by mechanisms, phases (Explore, Experiment, Exploit), and consequences (CIA) |
| **N. Attack Progression** | **YES (per-pattern).** Each attack pattern has an `Attack_Execution_Flow` with ordered steps and phases |
| **O–P. PCAPs/Packet Features** | **NO** |
| **Q. Laptop Feasibility** | **HIGH** |
| **R. Temporal Transformer Suitability** | **NOT A DATASET — Knowledge Base.** Like ATT&CK, CAPEC provides attack pattern descriptions for mapping |

**Use for WARDEN-T:** CAPEC provides granular attack pattern descriptions. Each CAPEC entry has an execution flow with phases (Explore → Experiment → Exploit). This can complement MITRE ATT&CK for interpretable evidence.

---

### 10. CVE/NVD (Common Vulnerabilities and Exposures / National Vulnerability Database)

| Criterion | Finding |
|-----------|---------|
| **A. Name** | CVE / NVD |
| **B. Official Source** | https://nvd.nist.gov/ and https://cve.mitre.org/ |
| **C. Download Format** | JSON feeds from NVD, CVE list XML |
| **D. Approximate Size** | Hundreds of thousands of entries; NVD JSON feeds are multiple GB |
| **E–L. Network Features** | **N/A.** Not a network traffic dataset |
| **M. Labels** | **YES.** Vulnerability identifiers with severity scores, descriptions, references |
| **N. Attack Progression** | **NO (per-vulnerability)**. CVEs describe vulnerabilities, not attack sequences |
| **O–R. PCAPs/Suitability** | **N/A.** Not suitable as a primary or secondary dataset for temporal forecasting |

**Use for WARDEN-T:** CVE/NVD provides context about known vulnerabilities that attackers exploit. It can be used to annotate which attacks in the temporal forecast correspond to known CVEs, but it cannot serve as a training dataset.

---

## TASK 2: PRACTICALITY COMPARISON TABLE

| Dataset | Size/Complexity | Temporal Info | Flow Features | Packet Info | Attack Labels | Temporal Attack Progression | PCAP Availability | Laptop Feasibility | Suitability for WARDEN-T |
|---------|----------------|---------------|---------------|-------------|---------------|---------------------------|-------------------|--------------------|-----------------------|
| **CIC-IDS2017** | 51 GB (PCAP), 2.8M rows CSV | ✅ Timestamps per flow | ✅ 80+ features incl. src/dst IP, ports, protocol, flags | ✅ PCAPs + flow features | ✅ 15 attack categories | ✅ Documented daily attack schedule | ✅ Full PCAPs | ⚠️ CSV-only feasible; full PCAPs challenging | ✅ **HIGH** |
| **CIC-IDS2018** | 100+ GB, 7 days, 50+ machines | ✅ Timestamps per flow | ✅ 83 features incl. all CICFlowMeter | ✅ PCAPs + logs | ✅ 7 attack types | ✅ Detailed per-day schedule | ✅ Full PCAPs + logs | ❌ Requires AWS infrastructure | ⚠️ High in theory, impractical for prototype |
| **UNSW-NB15** | 2.5 GB CSV, 2.5M rows | ✅ Stime/Ltime timestamps | ✅ 49 features; src/dst IP, ports, protocol, state | ✅ PCAPs, BRO, Argus | ✅ 9 attack categories | ⚠️ Records ordered but no explicit attack schedule | ✅ PCAPs | ✅ CSV feasible on laptop | ⚠️ Moderate-High; lacks TCP flag counts |
| **CTU-13** | 1.9 GB compressed | ✅ Timestamps in binetflow | ⚠️ Binetflow; no TCP flag counts | ✅ Botnet PCAPs | ✅ Botnet/Normal/C&C | ⚠️ 13 isolated botnet scenarios | ✅ Botnet PCAPs only | ✅ Feasible on laptop | ⚠️ Moderate; botnet-only, lacks diversity |
| **CICIoT2023** | 13 GB CSV, 46M rows | ✅ ts timestamp | ⚠️ 46 features but NO src/dst IP/ports | ✅ PCAPs | ✅ 33 attack types | ⚠️ Attack scenarios, not unified timeline | ✅ PCAPs | ⚠️ Heavy; can be subsampled | ⚠️ Moderate; lacks endpoint identity |
| **LANL** | 12 GB compressed | ✅ Epoch timestamps | ⚠️ Flow summaries only; no TCP flags | ❌ No PCAPs | ⚠️ Red team events only | ⚠️ 58 days, authentication-centric | ❌ | ❌ Too large; de-identified | ⚠️ Low; auth-focused, not network-traffic |
| **DARPA 1998/99** | 75 MB (10% subset) | ⚠️ Limited timestamps | ❌ No src/dst IP/ports | ✅ Tcpdump data | ✅ 24+ attack types | ⚠️ 7-week timeline but not hourly | ✅ Tcpdump | ✅ Very feasible | ❌ Low; outdated, no IP/port features |
| **MITRE ATT&CK** | ~100 MB STIX JSON | ❌ N/A | ❌ N/A | ❌ N/A | ✅ 190+ techniques | ✅ Kill chain tactics | ❌ | ✅ Trivial | ✅ **Knowledge base for mapping** |
| **CAPEC** | ~559 patterns | ❌ N/A | ❌ N/A | ❌ N/A | ✅ 559 patterns | ✅ Per-pattern execution flow | ❌ | ✅ Trivial | ✅ **Knowledge base for mapping** |
| **CVE/NVD** | Multiple GB | ❌ N/A | ❌ N/A | ❌ N/A | ✅ CVE IDs | ❌ | ❌ | ⚠️ Large feeds | ✅ **Knowledge base for context** |

---

### Trade-off Analysis

**CIC-IDS2017 vs CIC-IDS2018:**
- Both have identical feature extraction methodology (CICFlowMeter) and similar attack types
- CIC-IDS2018 is 2× larger with more machines and a longer capture period
- CIC-IDS2017 is more practical for a laptop/Colab prototype while retaining all necessary features
- **Trade-off:** Scale vs. feasibility. CIC-IDS2018 offers more data but requires AWS; CIC-IDS2017 fits on a laptop

**CIC-IDS2017 vs UNSW-NB15:**
- CIC-IDS2017 has TCP flag counts (FIN, SYN, RST, PSH, ACK, URG, CWE, ECE); UNSW-NB15 has only a `state` field
- UNSW-NB15 has 49 features vs CIC-IDS2017's 80+; UNSW-NB15's feature set is more compact
- UNSW-NB15 is smaller (2.5 GB CSV vs CIC-IDS2017's ~500 MB CSV for 2.8M rows)
- CIC-IDS2017 has explicit attack schedules; UNSW-NB15 does not
- **Trade-off:** Feature richness + temporal structure (CIC-IDS2017) vs. smaller size + modern attack types (UNSW-NB15)

**CIC-IDS2017 vs CICIoT2023:**
- CICIoT2023 has more attack categories (33 vs 15) but lacks source/dest IP/port in extracted features
- CIC-IDS2017 preserves bidirectional flow identity; CICIoT2023 aggregates packet windows
- CICIoT2023 is heavier (13 GB uncompressed)
- **Trade-off:** Attack diversity (CICIoT2023) vs. network state identity preservation (CIC-IDS2017)

**CIC-IDS2017 vs CTU-13:**
- CTU-13 is smaller and more feasible but is botnet-only
- CIC-IDS2017 has diverse attack types spanning the full attack lifecycle
- **Trade-off:** Feasibility (CTU-13) vs. attack diversity and temporal structure (CIC-IDS2017)

---

## TASK 3: MVP DATASET RECOMMENDATION

### Primary Dataset: **CIC-IDS2017**

**Justification:**

1. **Temporal Information:** Every flow record has a timestamp. The dataset spans 5 consecutive days (Mon–Fri, July 3–7, 2017) with documented attack windows on specific days and hours. This directly supports constructing sequential windows S(t-9), ..., S(t).

2. **Network Features:** 80+ CICFlowMeter features per flow, including: source/destination IP, source/destination port, protocol, 8 TCP flag counts, forward/backward byte and packet counts, flow duration, bidirectional IAT statistics (mean/std/max/min), TCP window sizes, and more. This enables a rich state vector S_t.

3. **Attack Labels:** 15 distinct attack categories plus benign, providing granular labels for attack-stage mapping. The multi-class label structure allows predicting specific attack types, not just binary attack/no-attack.

4. **Sequential Windows:** Flows are ordered chronologically by timestamp. This enables creating time-ordered sequences where S_t represents the network state at time window t, and we can observe S(t-9)...S(t) to predict S(t+1).

5. **Future Attack-State Targets:** Since attacks are documented with specific timing (e.g., "FTP-Patator: 9:20–10:20 AM Tuesday"), we can define the target S(t+1) as the attack type active in the next time window, enabling multi-step forecasting.

6. **Dataset Size:** CSV-only subset (~2.8M rows, ~500 MB) is feasible on a laptop or Google Colab. A Hugging Face subset (~101k rows) exists for rapid prototyping.

7. **Ease of Preprocessing:** CICFlowMeter has already extracted all features. The CSV format is well-documented. The `CICFlowMeter` GitHub repo and `flowprep` tooling on Hugging Face provide ready-to-use preprocessing scripts.

8. **Laptop/Colab Feasibility:** The CSV-only version fits in Google Colab's default runtime (12.5 GB drive, 12 GB RAM). A downsampled version (~100k rows) enables rapid experimentation.

### Secondary Dataset: **UNSW-NB15**

**Justification for Cross-Validation:**
- Different network environment (IXIA PerfectStorm-generated traffic vs. CIC's B-Profile simulated traffic)
- 49 features covering flow, content, time, and connection-level statistics
- 9 attack categories including fuzzers, shellcode, and worms (not present in CIC-IDS2017)
- Same timestamp ordering enables comparable temporal modelling
- Smaller footprint (2.5 GB CSV) makes it an ideal validation dataset
- **Important caveat:** Lacks individual TCP flag counts (has `state` field instead), which may limit some attack-type detection capabilities

---

## MITRE ATT&CK AS SECONDARY KNOWLEDGE SOURCE

**Assessment:** MITRE ATT&CK is **highly suitable** as a secondary knowledge source for mapping observed/predicted behaviour to attack stages.

**How it maps to WARDEN-T:**

The ATT&CK framework's tactical lifecycle provides natural attack-stage labels:
- **Reconnaissance** → Corresponds to scanning/port scan activity (detectable via SYN scans, unusual port access patterns)
- **Initial Access** → Corresponds to brute force, exploitation (detectable via failed login spikes, anomalous connection patterns)
- **Execution** → Corresponds to payload delivery (detectable via process/network anomalies)
- **Persistence** → Corresponds to backdoor installation (detectable via unusual sustained connections)
- **Discovery/Lateral Movement** → Corresponds to internal network scanning (detectable via internal IP sweeps)
- **Exfiltration** → Corresponds to data theft (detectable via unusual outbound traffic volumes)
- **Impact** → Corresponds to DoS/DDoS (detectable via traffic volume spikes)

**Practical application in WARDEN-T:**
1. After the Temporal Transformer predicts S(t+1), map the predicted feature vector to the most likely ATT&CK tactic
2. Use ATT&CK sub-technique IDs as interpretable output labels
3. Provide evidence: "S(t+1) predicted as Exfiltration (T1041) based on elevated outbound byte rate and unusual destination IP patterns"
4. CAPEC can provide additional attack pattern-level detail for the same mapping

---

## DATASET DOWNLOAD AND PLACEMENT INSTRUCTIONS

### For CIC-IDS2017 (Primary):

1. **Which dataset/file to download:**  
   Go to https://www.unb.ca/cic/datasets/ids-2017.html and click "Download this dataset"  
   You will get a ZIP containing:
   - `Monday-WorkingHours.pcap` through `Friday-WorkingHours.pcap`
   - `GeneratedLabelledFlows/` (PCAP-derived CSVs per day)
   - `MachineLearningCSV.zip` (the CSV-only subset for ML)

2. **Where to download it from:**  
   `http://cicresearch.ca/CICDataset/CIC-IDS-2017/` (requires account creation on the CIC website)

3. **Where to place it in the project:**  
   ```
   warden-t/data/raw/cic-ids2017/
   ├── pcap/          (all 5 .pcap files)
   ├── csv/           (MachineLearningCSV.zip contents)
   └── labelled_flows/ (GeneratedLabelledFlows contents)
   ```

4. **Command to run after downloading:**  
   ```bash
   python src/data/inspect_dataset.py --data-dir data/raw/cic-ids2017/csv/
   ```

### For UNSW-NB15 (Secondary):

1. **Which dataset/file to download:**  
   Go to https://research.unsw.edu.au/projects/unsw-nb15-dataset and download the source files (CSV files)

2. **Where to download it from:**  
   UNSW CIC datasets page (direct download link provided on the project page)

3. **Where to place it in the project:**  
   ```
   warden-t/data/raw/unsw-nb15/
   ├── csv/           (UNSW-NB15_1.csv through UNSW-NB15_4.csv)
   └── pcap/          (optional: raw PCAP files)
   ```

4. **Command to run after downloading:**  
   ```bash
   python src/data/inspect_dataset.py --data-dir data/raw/unsw-nb15/csv/
   ```

### For MITRE ATT&CK (Knowledge Base):

1. **Which dataset/file to download:**  
   Clone from GitHub: `https://github.com/mitre-attack/attack-stix-data`

2. **Where to place it:**  
   ```
   warden-t/data/raw/attack-stix-data/
   ```

3. **Command:**  
   ```bash
   git clone https://github.com/mitre-attack/attack-stix-data.git data/raw/attack-stix-data
   ```

---

## DATASET INSPECTION SCRIPT

The script `src/data/inspect_dataset.py` has been created and provides:
- Number of rows, columns, column names, data types
- Missing values count
- Infinite values count
- Label distribution
- Timestamp range
- Unique attack labels
- Unique protocols
- Approximate file size

**Usage:**
```bash
# For CIC-IDS2017 CSV files
python src/data/inspect_dataset.py --data-dir data/raw/cic-ids2017/csv/

# For UNSW-NB15 CSV files  
python src/data/inspect_dataset.py --data-dir data/raw/unsw-nb15/csv/ --label-column Label --timestamp-column Stime

# For CICIoT2023 (merged CSV)
python src/data/inspect_dataset.py --data-dir data/raw/ciciot2023/ --label-column label
```

---

## CONCLUSION

**CIC-IDS2017 is the recommended primary dataset** for the WARDEN-T 3-day prototype because it uniquely satisfies all requirements for temporal network behaviour modelling:
- ✅ Timestamps on every flow enabling S(t-9)...S(t) window construction
- ✅ Source/destination IPs and ports for network state identity
- ✅ Complete TCP flag counts for port scan and attack signature detection
- ✅ Bidirectional flow features for rich state vectors
- ✅ 80+ features per flow for deep temporal modelling
- ✅ Documented attack schedules enabling future-state target definition
- ✅ PCAPs available for packet-level feature extraction if needed
- ✅ CSV-only subset feasible on laptop/Colab

**UNSW-NB15 is recommended as secondary** for cross-validation due to its different traffic generation methodology and complementary attack categories.

**MITRE ATT&CK is recommended as the knowledge-base layer** for mapping predicted attack behaviour to interpretable attack stages.

No model implementation will proceed until this reconnaissance is reviewed and approved.
