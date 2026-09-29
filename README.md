# WARDEN-T: Temporal Transformer for Network Attack Forecasting

**Project:** Smart India Hackathon 2026 — Problem Statement ID 26153  
**Title:** AI-based Network Attack Forecasting from Network Traffic Data  
**Architecture:** WARDEN-T — Temporal Transformer for Multi-Step Attack Forecasting  

## Overview

WARDEN-T is a Temporal Transformer architecture that learns temporal network-state transitions and performs multi-step forecasting of attack/infiltration probability. The system:

1. Represents network state at time `t` as a structured feature vector `S_t` (87 features: flow + packet-level)
2. Observes a temporal sequence: `S(t-9), S(t-8), ..., S(t)`
3. Learns temporal state transitions
4. Predicts `S(t+1)`
5. Recursively rolls forward for multiple future windows
6. Produces attack/infiltration probability for future horizons
7. Maps predicted behaviour to attack stages (MITRE ATT&CK)
8. Provides interpretable evidence (attention + Integrated Gradients)
9. Supports PCAP input (Scapy/cicflowmeter converter)
10. Includes what-if simulation, surprise signal, and stage-aware playbook

## Key Features

- **87 network-level features** per window (flow statistics + packet-level: TCP window bytes, port entropy, SYN/RST ratios)
- **No label leakage** — `window_attack_ratio` removed from input features
- **Persistence baseline** comparison (computed, not hardcoded)
- **Onset detection / lead time** metrics (forecasting vs classification)
- **Calibration curves** with threshold analysis
- **Integrated Gradients** explainability
- **Ensemble uncertainty** (3-seed)
- **Leave-one-attack-family-out** evaluation
- **Per-host attribution** from flagged windows
- **World-model surprise signal** (prediction error = novel behaviour)
- **What-if simulation** (adjust features, re-roll forecast)
- **Stage-aware playbook** (ATT&CK-mapped mitigations)
- **PCAP support** (Scapy/cicflowmeter converter)
- **CII fit**: ~80K parameters, CPU-only, fully offline — suits air-gapped SCADA/utility networks

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/snehagahlot3/Warden.git
cd Warden
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download CIC-IDS2017 dataset
Download from [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) and place `cic_ids2017_full.csv` in `data/raw/CIC-IDS2017/`.

### 5. Run the data pipeline
```bash
python src/data/phase3_9_pipeline.py
```

### 6. Train the model
```bash
python -m src.models.train
```

### 7. Evaluate
```bash
python -m src.models.evaluate
```

### 8. Launch dashboard
```bash
streamlit run app.py
```

## Project Structure

```
warden-t/
├── data/
│   ├── raw/CIC-IDS2017/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── phase3_9_pipeline.py
│   │   ├── pcap_converter.py
│   │   └── time_based_windows.py
│   ├── features/
│   │   └── attack_mapping.py
│   ├── models/
│   │   ├── warden_transformer.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── evaluate_loo_family.py
│   │   └── evaluate_ensemble.py
│   ├── forecasting/
│   │   ├── rollout.py
│   │   ├── csv_inference.py
│   │   └── per_host_attribution.py
│   └── explainability/
│       └── attention.py
├── configs/config.yaml
├── app.py
├── styles.py
├── requirements.txt
└── README.md
```

## Processed Data Summary

| Metric | Value |
|--------|-------|
| Raw flows | 2,099,971 |
| Window size | 500 flows per state |
| Network states | 4,199 |
| Features per state | 87 (flow + packet-level) |
| Sequence length | 10 |
| Total sequences | 4,189 |
| X_train shape | (2513, 10, 87) |
| X_val shape | (838, 10, 87) |
| X_test shape | (838, 10, 87) |
| Train attack % | 62.9% |
| Val attack % | 70.8% |
| Test attack % | 82.9% |



## License

All datasets are publicly available under their respective licenses. See individual dataset pages for details.
