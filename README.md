# WaterGuardX

## Adaptive Water-System Anomaly Detection Using a Self-Supervised Temporal Transformer

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.64-brightgreen.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, production-grade machine learning system designed to detect physical anomalies, pipe bursts, and operational irregularities in municipal water distribution networks from multivariate SCADA sensor time-series data. 

WaterGuardX integrates a **Temporal Transformer Encoder** with **self-supervised masked sequence reconstruction pretraining**, a non-parametric **two-sample Kolmogorov-Smirnov distribution-shift detector**, and a **controlled replay-buffered adaptation mechanism** that suppresses false alarms under seasonal and operational drift without catastrophic forgetting.

---

## Overview

Municipal water distribution networks rely on Supervisory Control and Data Acquisition (SCADA) telemetry monitoring pressures, flow rates, and tank levels at high temporal frequencies. Detecting physical pipeline failures, valve faults, and leakage events in these systems is fraught with challenges:
1. **Multivariate Temporal Dependencies:** Pipe hydraulics exhibit tight spatial correlation across network nodes and strong diurnal cyclical rhythms.
2. **Extreme Class Imbalance:** Physical failure events are rare, representing a small fraction of annual operational cycles.
3. **Distribution Shift:** Seasonal demand variations, pipe roughness degradation, and pump staging induce severe distribution shift that degrades static machine learning models into complete false-alarm saturation ($FPR \to 100\%$).
4. **Catastrophic Forgetting:** Naive online fine-tuning on shifted streams causes models to forget nominal baseline operational dynamics.

**WaterGuardX** solves these challenges by combining self-supervised representation pretraining with automated drift gating and safe replay-buffered adaptation.

---

## Problem Statement

Given continuous multivariate sensor observations $\mathbf{X}_t \in \mathbb{R}^{D}$ from 12 physical SCADA channels and 4 cyclical temporal encodings across sliding windows of length $T = 48$ timesteps (4 hours at 5-minute sampling resolution), the objective is to predict binary anomaly status $y_t \in \{0, 1\}$ while:
- Maximizing detection recall ($> 98\%$) to prevent catastrophic infrastructure failures.
- Maintaining low operational false-positive rates ($FPR < 2\%$).
- Maintaining ultra-low inference latency ($< 0.2\text{ ms}$ per window) suitable for edge SCADA RTUs.
- Detect distribution shift and automatically recover performance without human intervention or manual relabeling.

---

## Key Features

- **Temporal Transformer Sequence Modeling:** 2-layer, 2-head Transformer Encoder with Sinusoidal Positional Encoding capturing multi-scale temporal dependencies.
- **Self-Supervised Masked Reconstruction (SSL):** Pretrained on unannotated sensor sequences via 15% random sensor-timestep masking to learn intrinsic hydraulic conservation laws prior to anomaly classification.
- **Distribution-Shift Detection Gating:** Non-parametric Kolmogorov-Smirnov (KS) test across all 16 features with data-driven bootstrap thresholding ($\tau_{\text{drift}} = 0.009148$), eliminating unwarranted continuous retraining.
- **Safe Replay-Buffered Adaptive Learning:** Controlled fine-tuning leveraging a 20% historical nominal replay buffer that suppresses false alarms under distribution shift by **99.1%** while strictly preserving nominal baseline performance.
- **Evidence-Based Interpretability & Diagnostic Engine:** Permutation feature importance, sliding-window temporal occlusion sensitivity, multi-head attention mapping, and Expected Calibration Error (ECE).
- **Interactive Streamlit Web Application:** Production web interface for CSV ingestion, validation, real-time sequential scoring, dynamic sensor inspection, drift diagnostics, and CSV export.

---

## System Architecture

```
Input SCADA Sensor Stream (5-min intervals)
        ↓
Validation & StandardScaler Normalization
        ↓
Cyclical Temporal Feature Derivation (sin/cos hour, sin/cos dow)
        ↓
Sliding Sequence Generation (T = 48 timesteps × D = 16 features)
        ↓
Linear Feature Projection (16 → 32) + Sinusoidal Positional Encoding
        ↓
2-Layer Multi-Head Temporal Transformer Encoder (d_model=32, nhead=2, d_ff=64)
        ↓
Mean Temporal Pooling & Latent Representation
        ↓
Linear Classification Head + Sigmoid Activation
        ↓
Continuous Anomaly Score p ∈ [0, 1] vs Fixed Threshold (τ = 0.0100)
        ↓
Kolmogorov-Smirnov Shift Detection Gating (τ_drift = 0.009148)
        ↓
[Normal Stream: Nominal Model]  |  [Shifted Stream: Replay-Adapted Model]
```

### Self-Supervised Pretraining Mechanism
Before supervised classification, the backbone encoder is pretrained on unlabeled raw sensor streams via masked autoencoding:
$$\mathcal{L}_{\text{SSL}} = \frac{1}{|\mathcal{M}|} \sum_{(t, d) \in \mathcal{M}} \left( \hat{X}_{t, d} - X_{t, d} \right)^2$$
where $\mathcal{M}$ represents a random 15% mask of sequence elements. Pretraining builds a dense representation of diurnal demand curves and pressure correlations that substantially improves fine-tuning convergence and adaptation plasticity.

---

## Dataset

Synthesized based on physical SCADA telemetry and ground-truth network characteristics from the **BattLeDIM 2020 Benchmark**:
- **Temporal Span:** Full year 2018 (105,120 consecutive timesteps at 5-minute sampling interval).
- **Sensors ($D=12$):**
  - 8 Pressure Junctions: `n1`, `n54`, `n105`, `n163`, `n215`, `n332`, `n458`, `n549` (meters of head)
  - 2 Pipe Flow Meters: `p227`, `p235` (liters / second)
  - 1 Pump Station Status: `PUMP_1` (operational state)
  - 1 Storage Reservoir Tank Level: `T1` (water column height in meters)
- **Engineered Temporal Features ($D=4$):** `sin_hour`, `cos_hour`, `sin_dow`, `cos_dow`.
- **Chronological Split (Strictly Leakage-Free):**
  - **Train:** January 1 – August 31, 2018 ($69,984$ timesteps, 66.6%)
  - **Validation:** September 1 – October 31, 2018 ($17,568$ timesteps, 16.7%)
  - **Held-Out Test:** November 1 – December 31, 2018 ($17,568$ timesteps, 16.7%)
  - **Untouched Final Test Subsets:** December 1 – December 31, 2018 ($8,928$ timesteps; $8,881$ evaluation sequences).

---

## Official Research Performance & Benchmarks (Phase 10)

Comprehensive evaluation conducted on untouched held-out December 1–31, 2018 test data ($N = 8,881$ test sequences):

| Metric | Normal Test Stream (Nominal Model) | Shifted Test Stream (Pre-Adaptation) | Shifted Test Stream (Post-Adaptation) | Normal Stream Retention (Post-Adaptation) |
|---|---|---|---|---|
| **Accuracy** | 98.27% | 18.93% | **98.96%** | **98.96%** |
| **Precision** | 92.44% | 18.93% | **96.33%** | **96.22%** |
| **Recall (Sensitivity)** | **98.93%** | 100.00% | **98.27%** | **98.39%** |
| **Specificity** | 98.11% | 0.00% | **99.12%** | **99.10%** |
| **F1-Score** | **0.9557** | 0.3183 | **0.9729** | **0.9729** |
| **ROC-AUC** | **0.9979** | 0.9824 | **0.9956** | **0.9972** |
| **PR-AUC** | **0.9943** | 0.9579 | **0.9891** | **0.9926** |
| **False Positive Rate (FPR)** | **1.89%** | 100.00% | **0.88%** | **0.90%** |
| **False Negative Rate (FNR)** | **1.07%** | 0.00% | **1.73%** | **1.61%** |

### Cross-Model Comparison Across Entire Project
| Model Architecture | Parameters | Model Size | Latency | F1-Score | Recall | PR-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** (Phase 3) | 17 | 1.31 KB | 0.0003 ms | 0.8848 | 0.8463 | 0.9179 |
| **Random Forest** (Phase 3) | 100 trees | 8,123.7 KB | 0.0094 ms | 0.9779 | 0.9885 | 0.9870 |
| **XGBoost** (Phase 3) | 100 trees | 614.6 KB | 0.0020 ms | 0.9777 | 0.9935 | 0.9911 |
| **LSTM Baseline** (Phase 4) | 15,105 | 86.23 KB | 1.0951 ms | 0.4647 | 0.3784 | 0.5806 |
| **Standard Transformer** (Phase 5) | 17,729 | 87.77 KB | 0.6995 ms | 0.9221 | 0.9392 | 0.9769 |
| **SSL Transformer** (Phase 6) | 19,313 | 95.23 KB | 0.4257 ms | 0.9320 | 0.9730 | 0.9851 |
| **WaterGuardX Nominal** (Phase 10) | **19,313** | **95.15 KB** | **0.1451 ms** | **0.9557** | **0.9893** | **0.9943** |
| **WaterGuardX Adapted** (Phase 10) | **19,313** | **95.15 KB** | **0.1451 ms** | **0.9729** | **0.9827** | **0.9891** |

---

## Diagnostic Error Analysis & Interpretability (Phase 11)

- **Sensor Permutation Sensitivity (Test Set):** Permuting feature columns revealed that primary downstream junction `n332` (Average Precision drop $= 0.4775$), critical transmission node `n105` (AP drop $= 0.2124$), and pump state `PUMP_1` (AP drop $= 0.2350$) exert the highest empirical influence on model predictions. *(Statistical sensitivity; does not assert physical causality)*.
- **Temporal Occlusion Dynamics:** Occluding sliding temporal blocks across the 48-step sequence showed that the **most recent 12 timesteps ($t-60$ to $t$ minutes)** induce the greatest prediction shift ($|\Delta \hat{p}| \approx 0.15$), confirming that immediate hydraulic shocks outweigh distant historical context.
- **False Positive Root Cause:** High-confidence false alarms ($\hat{p} > 0.95$) are concentrated around midnight pump staging events where sudden pressure switching steps ($>4\sigma$) temporally mimic physical pipe bursts.
- **False Negative Root Cause:** Missed anomalies ($\hat{p} < 0.001$) correspond to low-magnitude incipient pipe leakages ($<0.6\sigma$) where signal attenuation remains within diurnal bounds.
- **Calibration (Expected Calibration Error):** Normal Stream $\text{ECE} = 0.00473$, Adapted Stream $\text{ECE} = 0.01252$.

---

## Project Structure

```
WaterGuardX/
├── app/
│   ├── app.py                      # Main Streamlit web application
│   ├── components/
│   │   ├── data_upload.py          # Data ingestion & validation component
│   │   ├── prediction_view.py      # Real-time inference & drift status
│   │   ├── charts.py               # Interactive Plotly time-series charts
│   │   └── metrics_view.py         # Official benchmarks & diagnostic tabs
│   └── utils/
│       └── app_utils.py            # Artifact loaders, preprocessing & inference
├── data/
│   ├── raw/                        # Ground-truth BattLeDIM SCADA data
│   │   ├── water_sensor_data.csv   # Synthesized full 2018 dataset (105,120 rows)
│   │   ├── anomaly_events.json     # Detailed injected failure metadata
│   │   └── sensor_stats.json       # SCADA physical distribution statistics
│   └── processed/                  # Preprocessed splits & shifted datasets
│       ├── scaler.pkl              # Fitted StandardScaler artifact
│       └── preprocessing_meta.json # Preprocessing feature ordering & metadata
├── src/
│   ├── config.py                   # Global constants and paths
│   ├── data_loader.py              # Raw data loading utilities
│   ├── preprocessing.py            # Chronological splitting and feature scaling
│   ├── datasets.py                 # PyTorch sequence window datasets
│   ├── visualization.py            # Diagnostic plotting functions
│   ├── baselines.py                # Logistic Regression, Random Forest, XGBoost
│   ├── lstm.py                     # Recurrent baseline architecture
│   ├── transformer.py              # Temporal Transformer backbone
│   ├── self_supervised.py          # Masked autoencoder pretraining & fine-tuning
│   ├── drift_detection.py          # Two-sample KS shift detection
│   ├── adaptation.py               # Replay-buffered fine-tuning & gating
│   ├── training.py                 # Training loops & early stopping
│   ├── evaluation.py               # Threshold optimization & metrics computation
│   ├── error_analysis.py           # Phase 11 diagnostic error extraction
│   └── interpretability.py         # Attention, occlusion, & permutation importance
├── models/
│   ├── final_model.pt              # Official Phase 10 Nominal Model (19,313 weights)
│   └── final_model_adapted.pt      # Official Phase 10 Shift-Adapted Model
├── results/
│   ├── final_results.json          # Master evaluation metrics
│   ├── final_model_selection.json  # Architecture justification and ablation evidence
│   ├── final_threshold.json        # Calibrated decision thresholds
│   ├── drift/                      # Phase 7 distribution shift metrics
│   ├── ablation/                   # Phase 9 component contribution analysis
│   └── error_analysis/             # Phase 11 predictions and error taxonomy
├── figures/                        # Publication figures across all phases
├── notebooks/                      # 11 interactive research Jupyter notebooks
├── requirements.txt                # Production Python dependencies
└── README.md                       # Master project documentation
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/jpreethika40-oss/WaterGuardX.git
cd WaterGuardX
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Streamlit Application

Launch the interactive web application locally:
```bash
streamlit run app/app.py
```
Access the application in your browser at `http://localhost:8501`.

### Application Features:
1. **Live SCADA Ingestion:** Upload arbitrary continuous `.csv` sensor recordings or select pre-packaged held-out December 2018 streams.
2. **Automated Validation & Preprocessing:** Automatic schema validation, unit detection, standard scaling, and cyclical feature engineering.
3. **Distribution Shift Monitor:** Continuous evaluation of Kolmogorov-Smirnov statistics to detect operational drift in real time.
4. **Adaptive Gating:** Automatic alert recommending the Replay-Adapted model when shift is confirmed.
5. **Interactive Anomaly Timeline:** Zoomable Plotly trajectory showing anomaly probability vs the fixed $\tau = 0.0100$ threshold with downloadable predictions.

---

## Reproducing Research Phases

To execute any phase independently:

| Phase | Description | Command |
|---|---|---|
| **Phase 2** | Exploratory Data Analysis & Preprocessing | `python run_eda.py` |
| **Phase 3** | Classical Baselines (LR, RF, XGB) | `python -m src.baselines` |
| **Phase 4** | LSTM Recurrent Baseline | `python -m src.lstm` |
| **Phase 5** | Standard Temporal Transformer | `python train_transformer.py` |
| **Phase 6** | Self-Supervised Pretraining (SSL) | `python train_ssl.py` |
| **Phase 7** | Distribution Shift Detection | `python train_drift.py` |
| **Phase 8** | Adaptive Learning with Replay Buffer | `python train_adaptation.py` |
| **Phase 9** | Systematic Ablation Study | `python train_ablation.py` |
| **Phase 10** | Final Optimization & Comprehensive Evaluation | `python train_final.py` |
| **Phase 11** | Final Error Analysis & Model Interpretability | `python run_error_analysis.py` |

---

## Phase Status

- [x] **Phase 1:** Dataset Discovery & Physical Water Network Profiling
- [x] **Phase 2:** Exploratory Data Analysis & Leakage-Free Preprocessing
- [x] **Phase 3:** Classical Machine Learning Baselines (LR, RF, XGBoost)
- [x] **Phase 4:** LSTM Recurrent Baseline
- [x] **Phase 5:** Temporal Transformer Encoder
- [x] **Phase 6:** Self-Supervised Masked Sequence Pretraining
- [x] **Phase 7:** Distribution Shift Detection & Robustness Evaluation
- [x] **Phase 8:** Adaptive Learning Under Distribution Shift
- [x] **Phase 9:** Systematic Ablation & Component Contribution Analysis
- [x] **Phase 10:** Final Optimization & Comprehensive Model Evaluation
- [x] **Phase 11:** Final Error Analysis & Model Interpretability
- [x] **Phase 12:** WaterGuardX Final Streamlit Application & GitHub Integration

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
