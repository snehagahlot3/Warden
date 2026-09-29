# WARDEN-T Data Pipeline Documentation

## Source Dataset

**CIC-IDS2017** (Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017)

- **Source:** Hugging Face mirror (`bvk/CICIDS-2017`), originally from `http://cicresearch.ca/CICDataset/CIC-IDS-2017/`
- **Paper:** Sharafaldin et al., "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018
- **Raw CSV:** `data/raw/CIC-IDS2017/cic_ids2017_full.csv` (825.10 MB, 2,099,971 rows, 89 columns)

## Raw Files Used

- **File:** `cic_ids2017_full.csv`
- **Size:** 825.10 MB (865,180,471 bytes)
- **Rows:** 2,099,971 flow records
- **Columns:** 89 features
- **Missing values:** 0
- **Infinite values:** 0
- **Duplicate rows:** 3,167

## Selected Features

20 features selected from 89 available columns. 19 numerical, 1 categorical (protocol).

| Feature Name | Original Column | Why Useful | Preprocessing |
|---|---|---|---|
| flow_duration | Flow Duration | Connection duration indicates attack type | log_transform + standard_scale |
| total_fwd_packets | Total Fwd Packet | Forward packet count | standard_scale |
| total_bwd_packets | Total Bwd packets | Backward packet count | standard_scale |
| total_fwd_bytes | Total Length of Fwd Packet | Forward bytes volume | log_transform + standard_scale |
| total_bwd_bytes | Total Length of Bwd Packet | Backward bytes volume | log_transform + standard_scale |
| flow_iat_mean | Flow IAT Mean | Mean inter-arrival time | log_transform + standard_scale |
| flow_iat_std | Flow IAT Std | IAT variance indicates burstiness | log_transform + standard_scale |
| fwd_iat_mean | Fwd IAT Mean | Forward inter-arrival time | log_transform + standard_scale |
| bwd_iat_mean | Bwd IAT Mean | Backward inter-arrival time | log_transform + standard_scale |
| down_up_ratio | Down/Up Ratio | Download/upload ratio | standard_scale |
| fwd_packets_per_sec | Fwd Packets/s | Forward throughput | standard_scale |
| bwd_packets_per_sec | Bwd Packets/s | Backward throughput | standard_scale |
| syn_flag_count | SYN Flag Count | Connection attempts; port scans | standard_scale |
| rst_flag_count | RST Flag Count | Connection resets | standard_scale |
| fin_flag_count | FIN Flag Count | Graceful closes | standard_scale |
| psh_flag_count | PSH Flag Count | Push flag activity | standard_scale |
| packet_length_mean | Packet Length Mean | Average packet size | standard_scale |
| packet_length_std | Packet Length Std | Packet size variance | standard_scale |
| avg_packet_size | Average Packet Size | Overall packet size | standard_scale |
| protocol | Protocol | Transport protocol (TCP/UDP/ICMP) | one_hot_encode |

### Features NOT Available in CIC-IDS2017 CSV

The following features from the problem statement are **NOT available** in the CIC-IDS2017 CSV:

- **TTL variance:** CIC-IDS2017 does not extract TTL fields
- **TCP window size:** Only FWD Init Win Bytes / Bwd Init Win Bytes are available (initial window, not per-packet)
- **Fragment flags:** Not present in CICFlowMeter output
- **Payload size distribution:** Payloads are anonymized in the dataset
- **Retransmission counts:** Not directly available (can be approximated from IAT anomalies)

## Temporal Aggregation

### Network State Definition

Each network state `S_t` represents the aggregated behavior of **500 consecutive flows**.

`S_t` is a vector of **84 dimensions**:
- 80 numerical features (20 features x 4 statistics: mean, std, max, min)
- 4 derived features: `window_flow_count`, `window_total_bytes`, `window_total_packets`, `window_attack_ratio`

### Window Size Rationale

- CIC-IDS2017 has ~2.1M flows over 5 days
- ~4.86 flows per minute
- 500 flows ≈ 103 minutes of traffic
- Each window captures a meaningful temporal pattern

## Sequence Construction

- **Sequence length:** 10 time steps
- **Input:** `X = [S(t-9), S(t-8), ..., S(t)]` — shape `(batch, 10, 81)`
- **Target:** `S(t+1)` — the next network state's attack/benign label
- **Total sequences:** 4,189 (from 4,199 network states, minus 10 for lookback)
- **Feature tensor shape:** `(4189, 10, 81)`

## Label Construction

**Rule:** Majority vote within each 500-flow window.

- **BENIGN:** 0% attack flows in window
- **MIXED_LIGHT:** <10% attack flows (borderline)
- **ATTACK:** 10-50% attack flows
- **HEAVY_ATTACK:** >50% attack flows

For the forecasting task, labels are binary:
- `0` = BENIGN
- `1` = ATTACK (includes MIXED_LIGHT, ATTACK, HEAVY_ATTACK)

**Note:** The binary distribution is skewed toward attack (68.3% attack overall) because attacks create concentrated traffic bursts that fill more windows with attack traffic.

## Temporal Train/Validation/Test Split

**Time-aware split** — no random shuffling to avoid temporal leakage:

| Split | Range | Sequences | Attack % |
|---|---|---|---|
| **Train** | Sequences 0-2512 | 2,513 | 62.9% |
| **Validation** | Sequences 2513-3350 | 838 | 70.8% |
| **Test** | Sequences 3351-4188 | 838 | 82.9% |

The split is chronological: earlier data → train, middle → validation, later → test. The test set has higher attack ratio because CIC-IDS2017 has more attacks concentrated in the later days (Tuesday-Friday vs Monday which is benign-only).

## Final Tensor Shapes

| File | Shape | Size |
|---|---|---|
| X_train.npy | (2513, 10, 81) | 15.53 MB |
| y_train.npy | (2513,) | 20 KB |
| X_val.npy | (838, 10, 81) | 5.18 MB |
| y_val.npy | (838,) | 7 KB |
| X_test.npy | (838, 10, 81) | 5.18 MB |
| y_test.npy | (838,) | 7 KB |
| train_categories.npy | (2513,) | — |
| val_categories.npy | (838,) | — |
| test_categories.npy | (838,) | — |

## Class Distribution

| Split | Total | Benign | Attack | Attack % |
|---|---|---|---|---|
| Train | 2,513 | 932 (37.1%) | 1,581 (62.9%) | 62.9% |
| Validation | 838 | 245 (29.2%) | 593 (70.8%) | 70.8% |
| Test | 838 | 143 (17.1%) | 695 (82.9%) | 82.9% |

## Limitations

1. **Relative timestamps:** Timestamps are `MM:SS.mmm` format (relative within ~60 min windows), not absolute wall-clock. Row ordering is used for temporal sequence construction.
2. **No IP/port features:** Raw IP addresses are decimal-encoded and excluded to avoid high cardinality. Port numbers are excluded as individual features.
3. **Missing packet-level features:** TTL variance, fragment flags, TCP window size, payload distribution, and retransmission counts are NOT available in the CIC-IDS2017 CSV output from CICFlowMeter.
4. **Non-overlapping windows:** Windows are non-overlapping (500-flow steps). Overlapping windows could produce more training samples but increase redundancy.
5. **Class imbalance:** The temporal distribution of attacks in CIC-IDS2017 creates attack-heavy windows in later time periods.
6. **Label derivation:** Attack labels are derived from majority vote within windows, which may not capture single anomalous flows in predominantly benign windows.

## Configuration

All pipeline parameters are saved in `data/processed/pipeline_config.json`.

The configuration is also reflected in `configs/config.yaml` under the `cic_ids2017` section.
