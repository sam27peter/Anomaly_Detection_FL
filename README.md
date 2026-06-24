<div align="center">

# 🛰️ Anomaly Detection using Federated Learning

**A comprehensive research framework for privacy-preserving anomaly detection on distributed spacecraft telemetry data**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)]()

*IIT Palakkad Internship Project · June 2026*

</div>

---

## 📖 About

This framework investigates how anomaly detection models can be trained **collaboratively across distributed clients without ever sharing raw data** — a critical requirement for privacy-sensitive domains such as spacecraft telemetry monitoring.

The project supports both **centralized** and **federated** learning paradigms for multivariate time-series anomaly detection using NASA spacecraft telemetry datasets (SMAP and MSL), and provides a full research pipeline from raw data ingestion to statistical reporting and interactive dashboard visualisation.

### Implemented Capabilities

| Category | Details |
|---|---|
| **Learning Paradigms** | Centralized CNN baseline · FedAvg · FedProx |
| **Data Distributions** | IID · Non-IID client partitioning |
| **Datasets** | SMAP (25 features) · MSL (55 features) |
| **Evaluation** | Accuracy · Precision · Recall · F1 · ROC-AUC · PR-AUC · Fairness metrics |
| **Analysis** | Statistical analysis · Confidence intervals · Centralized vs Federated comparison |
| **Outputs** | Automated experiment tracking · Visualisation generation · Interactive dashboard |

---

## ✨ Framework Highlights

- 🔒 **Privacy-preserving** — raw telemetry never leaves the client; only model parameters are exchanged
- ⚙️ **One-command orchestration** — `python run_orchestrator.py` runs the entire research pipeline end-to-end
- 📦 **Reproducible experiments** — every run is automatically saved with its config, metrics, history, model, and plots
- 📊 **Interactive dashboard** — explore results through a Streamlit web interface
- ⚖️ **Fairness analysis** — quantifies per-client performance consistency across heterogeneous distributions
- 📐 **Statistical rigour** — 95% confidence intervals, mean/std, min/max across experiment runs
- 🧩 **Modular architecture** — designed for straightforward extension with new FL algorithms, models, and datasets

---

## 📂 Repository Structure

```
Anomaly_Detection_FL/
│
├── client/                        # Federated client logic
│   ├── fl_client.py               #   Local training (FedAvg + FedProx support)
│   └── partitioner.py             #   IID / Non-IID data partitioning
│
├── server/                        # Federation server
│   ├── fl_server.py               #   Aggregation & round coordination
│   └── strategies.py             #   FedAvg and FedProx aggregation strategies
│
├── models/                        # Model definitions
│   ├── cnn_model.py               #   1D CNN for multivariate time-series
│   ├── architecture.py            #   BaseModel interface
│   └── model_selector.py          #   Model factory
│
├── train/                         # Training & preprocessing
│   ├── centralized.py             #   Centralized (non-federated) baseline trainer
│   ├── centralized_runner.py      #   Entry point for centralized training
│   ├── preprocessor_v2.py         #   Data normalisation and sliding-window generation
│   └── data_manager.py            #   Dataset loading and statistics
│
├── evaluation/                    # Evaluation engine
│   ├── metrics.py                 #   Core metrics (accuracy, F1, ROC-AUC, PR-AUC, etc.)
│   ├── fairness.py                #   Per-client fairness metrics
│   ├── comparison.py              #   Cross-experiment comparison utilities
│   ├── centralized_vs_fl.py       #   Centralized vs federated performance gap
│   └── statistical_analysis.py   #   Confidence intervals and summary statistics
│
├── visualization/                 # Plot generation
│   ├── centralized_plots.py       #   Centralized training curves and confusion matrices
│   ├── federated_plots.py         #   Per-round accuracy / loss / per-client plots
│   ├── comparison_plots.py        #   Side-by-side algorithm comparisons
│   ├── comparison_table.py        #   Tabular comparison outputs
│   └── statistical_plots.py       #   Distribution and confidence interval charts
│
├── config/
│   └── federated_config.py        #   Centralised hyperparameter configuration
│
├── utils/
│   ├── logger.py                  #   Project-wide logging setup
│   ├── experiment_logger.py       #   Per-experiment logging
│   └── experiment_tracker.py      #   Experiment registry and metadata management
│
├── analysis/                      #   Post-hoc research analysis notebooks/scripts
│
├── data/
│   ├── raw/                       #   NASA telemetry files (not tracked in git)
│   │   ├── train/
│   │   ├── test/
│   │   └── labeled_anomalies.csv
│   ├── processed/                 #   Preprocessed datasets and reports
│   └── partitions/                #   Client data partitions
│
├── results/                       #   All experiment outputs (auto-created)
├── logs/                          #   Runtime logs (auto-created)
│
├── dashboard.py                   # Streamlit interactive dashboard
├── run_orchestrator.py            # End-to-end pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## 🔄 End-to-End Workflow

```
Raw NASA Telemetry (SMAP / MSL)
             │
             ▼
  Preprocessing & Sliding-Window Generation
             │
             ▼
     Client Data Partitioning (IID / Non-IID)
             │
        ┌────┴────┐
        ▼         ▼
  Centralized   Federated Training
   Baseline     (FedAvg / FedProx)
        │         │
        └────┬────┘
             ▼
     Experiment Tracking & Logging
             │
             ▼
    Metrics Evaluation (Accuracy, F1, AUC …)
             │
             ▼
     Fairness Analysis (per-client variance)
             │
             ▼
    Statistical Analysis (CI, mean, std)
             │
             ▼
   Visualisation Generation (plots, tables)
             │
             ▼
     Interactive Streamlit Dashboard
```

---

## 🚀 Quick Start

### 1 — Clone and install

```bash
git clone https://github.com/sam27peter/Anomaly_Detection_FL.git
cd Anomaly_Detection_FL
pip install -r requirements.txt
```

### 2 — Prepare the dataset

Place the NASA telemetry files inside `data/raw/`:

```
data/raw/
├── train/                  # channel_*.npy  (one file per telemetry channel)
├── test/                   # channel_*.npy
└── labeled_anomalies.csv   # ground-truth anomaly labels
```

Generate preprocessed datasets and client partitions:

```bash
python -m train.preprocessor_v2
python -m client.partitioner
```

### 3 — Run the full research pipeline

```bash
python run_orchestrator.py
```

This single command automatically:

1. Trains the centralised CNN baseline on SMAP and MSL
2. Runs all federated experiments (FedAvg / FedProx × IID / Non-IID)
3. Computes all evaluation metrics
4. Performs fairness analysis across clients
5. Generates all visualisations and comparison reports
6. Produces statistical summaries with confidence intervals
7. Writes all outputs ready for the dashboard

---

## 🖥️ Run Individual Components

### Centralised baseline

```bash
python -m train.centralized_runner SMAP
python -m train.centralized_runner MSL
```

### Federated training

```bash
# FedAvg — IID distribution on SMAP
python -m server.fl_server fedavg SMAP iid

# FedProx — Non-IID distribution on MSL
python -m server.fl_server fedprox MSL non_iid
```

**Argument reference**

| Position | Values |
|---|---|
| Algorithm | `fedavg` · `fedprox` |
| Dataset | `SMAP` · `MSL` |
| Partition | `iid` · `non_iid` |

---

## 🧠 Model Architecture

The anomaly detector uses a **1D CNN** designed for multivariate time-series:

```
Input  (batch, sequence_length, num_features)
  │
  ▼
Conv1d → BatchNorm → ReLU → MaxPool       [feature extraction]
  │
  ▼
Conv1d → BatchNorm → ReLU → MaxPool       [deeper pattern extraction]
  │
  ▼
AdaptiveAvgPool1d                          [handles variable-length sequences]
  │
  ▼
Linear(64 → 128) → ReLU → Dropout(0.3)
  │
  ▼
Linear(128 → 1) → Sigmoid

Output  0 = Normal  ·  1 = Anomaly
```

---

## 🌐 Federated Learning Protocol

```
Initialise global model on server
         │
         ▼
┌── For each communication round ──────────────────────────────┐
│                                                               │
│  Server broadcasts current global weights to all clients     │
│         │                                                     │
│         ▼                                                     │
│  Each client trains locally on its private partition          │
│  (FedProx adds proximal term: μ/2 · ‖w − w_global‖²)       │
│         │                                                     │
│         ▼                                                     │
│  Clients return updated weights (no raw data shared)         │
│         │                                                     │
│         ▼                                                     │
│  Server aggregates → new global model                        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
         │
         ▼
Evaluate global model · log metrics · save checkpoint
```

> **Privacy guarantee:** raw telemetry data never leaves the originating client. Only model weight tensors are transmitted.

---

## 📊 Interactive Dashboard

Launch the Streamlit dashboard to explore all experiment results interactively:

```bash
streamlit run dashboard.py
```

**Dashboard panels**

- Experiment selector (algorithm · dataset · partition)
- Training curves (accuracy and loss per round)
- Global evaluation metrics
- Per-client fairness statistics and accuracy bar chart
- Algorithm comparison tables
- Centralised vs federated performance gap analysis

---

## 🧪 Experiment Tracking

Every experiment run is automatically persisted under `results/experiments/`:

```
results/experiments/
└── fedavg_iid_SMAP_20260624_164557/
    ├── experiment_config.json     # Full hyperparameter snapshot
    ├── metrics.json               # Final evaluation metrics
    ├── history.json               # Per-round training history
    ├── global_model.pth           # Trained global model weights
    ├── accuracy_curve.png
    └── loss_curve.png
```

Stored artefacts per experiment:

- Full hyperparameter configuration (learning rate, rounds, clients, µ, etc.)
- Per-round and final metrics (accuracy, precision, recall, F1, AUC)
- Trained model checkpoint
- All generated plots

---

## 📈 Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Fraction of flagged anomalies that are true anomalies |
| Recall | Fraction of true anomalies that are detected |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under receiver operating characteristic curve |
| PR-AUC | Area under precision–recall curve |
| Confusion Matrix | Per-class breakdown (TP / TN / FP / FN) |

---

## ⚖️ Fairness Analysis

Federated learning introduces potential **performance disparity across clients** (especially under Non-IID distributions). The framework quantifies this with:

| Fairness Metric | Description |
|---|---|
| Mean client accuracy | Average performance across all clients |
| Standard deviation | Spread of per-client performance |
| Best client accuracy | Upper bound of performance |
| Worst client accuracy | Lower bound of performance |
| Accuracy gap | Best minus worst — key fairness indicator |

Low accuracy gap → model is fair; high gap → some clients benefit significantly more than others.

---

## 📐 Statistical Analysis

Automated statistical reports are written to `results/statistics/`:

```
results/statistics/
├── statistics.csv          # Per-algorithm summary table
└── mean_accuracy.png       # Accuracy distribution chart
```

Each report includes mean, standard deviation, min, max, and **95% confidence interval** across multiple runs — enabling statistically rigorous comparison between FedAvg, FedProx, and the centralised baseline.

---

## 📝 Logging

Separate log files are maintained per subsystem for full traceability:

```
logs/
├── centralized.log
├── federated.log
├── dashboard.log
├── project.log
└── experiments/
    └── <experiment_name>.log
```

---

## 📁 Results Layout

```
results/
├── experiments/           # Full per-run artefacts (model, metrics, plots)
├── comparison/            # Cross-algorithm comparison outputs
├── centralized_vs_fl/     # Centralised vs federated gap analysis
├── statistics/            # Statistical summaries and CI reports
└── single_machine/        # Centralised baseline results
```

---

## 🔮 Future Work

- [ ] **Flower-based deployment** — migrate federation layer to the `flwr` framework
- [ ] **Communication channel simulation** — AWGN and Polar-NRZ channel models to study degraded uplinks
- [ ] **Differential privacy** — add DP-SGD noise for formal privacy guarantees
- [ ] **Secure aggregation** — cryptographic protection of client updates
- [ ] **Transformer-based detector** — replace CNN with a time-series Transformer
- [ ] **Additional FL algorithms** — SCAFFOLD · FedAdam · FedYogi
- [ ] **Hyperparameter optimisation** — automated search over FL-specific hyperparameters
- [ ] **Real-time dashboard updates** — live streaming of in-progress experiment metrics

---

## 📚 References

**Federated Learning**
- McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg)
- Li et al. (2020) — *Federated Optimization in Heterogeneous Networks* (FedProx)

**Dataset**
- NASA Spacecraft Telemetry Anomaly Detection Dataset — SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Please open an issue or submit a pull request.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">

**👨‍💻 Sam Peter & Steve CJ· IIT Palakkad Internship Project · June 2026**

</div>
