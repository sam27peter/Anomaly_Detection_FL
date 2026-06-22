# Anomaly Detection with Federated Learning

A PyTorch-based federated learning framework for **time-series anomaly detection** in distributed environments. Implements **FedAvg** and **FedProx** algorithms to train anomaly detection models across multiple clients while preserving data privacy.

Built on **NASA satellite telemetry data** with support for both centralized and federated training paradigms.

---

## 🎯 Overview

### What It Does
- Trains **1D CNN models** on distributed time-series data without centralizing raw data
- Supports **two federation strategies:**
  - **FedAvg:** Simple weight averaging across clients
  - **FedProx:** Proximal-regularized averaging for heterogeneous client data
- Generates comprehensive **metrics & visualizations** (confusion matrices, accuracy curves, per-client performance)
- Provides **centralized baseline** for comparison

### Key Features
✅ **Privacy-preserving:** Data stays on clients; only model updates are aggregated  
✅ **Configurable:** Adjustable rounds, epochs, batch size, learning rate  
✅ **Multi-dataset support:** Easily swap datasets with different feature dimensions  
✅ **Data heterogeneity:** Supports IID and non-IID data partitioning  
✅ **Comprehensive metrics:** Accuracy, precision, recall, F1, confusion matrices  
✅ **Visualization-ready:** Generates loss curves, accuracy plots, client comparisons  

---

## 📋 Requirements

- **Python 3.8+**
- **PyTorch 2.0+** (with CUDA support optional but recommended)
- **scikit-learn**, **Pandas**, **NumPy**, **Matplotlib**, **Plotly**
- **Flower (flwr)** – Federated Learning framework

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Hardware
- **GPU:** Recommended (NVIDIA/CUDA). Code auto-detects and falls back to CPU.
- **RAM:** ≥8GB (depends on dataset size and batch size)
- **Disk:** ≥2GB (for data + results)

---

## 📂 Directory Structure

```
Anomaly_Detection_FL/
│
├── client/                           # Federated Learning Clients
│   ├── fl_client.py                 # Client training logic (FedAvg + FedProx support)
│   ├── client_runner.py             # Standalone client entry point
│   └── partitioner.py               # Data partitioning (IID/non-IID distribution)
│
├── server/                           # Federated Learning Server
│   ├── fl_server_avg.py             # FedAvg aggregation & coordination
│   ├── fl_server_prox.py            # FedProx aggregation (with proximal term)
│   ├── fedavg_runner.py             # FedAvg wrapper
│   └── fedprox_runner.py            # FedProx wrapper
│
├── models/                           # Model Architectures
│   ├── cnn_model.py                 # 1D CNN for time-series (Conv1d + FC layers)
│   ├── architecture.py              # BaseModel interface
│   └── model_selector.py            # Model factory function
│
├── train/                            # Training & Analysis Utilities
│   ├── centralized.py               # Centralized training (non-federated baseline)
│   ├── data_manager.py              # Dataset loader & statistics
│   ├── preprocessor_v2.py           # Data normalization & windowing
│   ├── cent_plots.py                # Visualization for centralized training
│   ├── feature_analysis.py          # Feature importance analysis
│   ├── failure_analysis.py          # Anomaly detection failure modes
│   ├── fusion_data_analysis.py      # Multi-sensor data fusion analysis
│   └── centralized_runner.py        # Centralized training entry point
│
├── data/
│   ├── raw/
│   │   ├── train/                   # NASA satellite training telemetry (.npy)
│   │   ├── test/                    # NASA satellite test telemetry (.npy)
│   │   └── labeled_anomalies.csv    # Ground truth anomaly labels
│   ├── processed/
│   │   ├── dataset_*                # Preprocessed datasets
│   │   ├── partitions/              # Client data partitions
│   │   └── reports/                 # Dataset statistics & analysis
│   └── .gitkeep
│
├── results/                          # Output directory (auto-created)
│   ├── single_machine/              # Centralized training results
│   │   ├── models/                  # Trained models (.pth)
│   │   ├── metrics/                 # Metrics JSON files
│   │   └── history/                 # Training history JSON
│   └── federated/
│       ├── fedavg/                  # FedAvg results by partition type
│       ├── fedprox/                 # FedProx results
│       ├── local_clients/           # Per-client local results
│       └── *_summary.csv            # Aggregated metrics across runs
│
├── requirements.txt                  # Python dependencies
├── run_orchestrator.py              # (Placeholder) Main orchestrator
├── dashboard.py                     # (Placeholder) Dash visualization UI
└── README.md                        # This file
```

---

## 🚀 Quick Start

### 1. Prepare Your Environment
```bash
git clone https://github.com/sam27peter/Anomaly_Detection_FL.git
cd Anomaly_Detection_FL
pip install -r requirements.txt
```

**Note:** You'll need NASA satellite telemetry data. Place raw data in `data/raw/` with structure:
```
data/raw/
├── train/              # channel_001.npy, channel_002.npy, ...
├── test/               # channel_001.npy, channel_002.npy, ...
└── labeled_anomalies.csv
```

### 2. Analyze Your Dataset
```bash
python train/data_manager.py
```
**Output:** Dataset statistics in `data/processed/reports/`
- `channel_statistics.csv` – Per-channel stats (min, max, mean, std, anomaly counts)
- `dataset_summary.json` – Aggregate dataset info

### 3. Run Centralized Baseline (Optional)
Train a single CNN on all data:
```bash
python train/centralized.py 25
```
**Arguments:**
- `25` – Number of features in dataset (default: "25")

**Output in `results/single_machine/`:**
- `models/cnn_25.pth` – Trained model weights
- `metrics/cnn_25.json` – Accuracy, precision, recall, F1, confusion matrix
- `history/cnn_25_history.json` – Training loss & accuracy per epoch

### 4. Run Federated Learning (FedAvg)
Distribute training across 5 clients over 3 rounds:
```bash
python server/fedavg_runner.py 25 iid
```

**Arguments:**
- `25` – Number of features
- `iid` – Data distribution type ("iid" = identically distributed, "non_iid" = heterogeneous)

**Output in `results/federated/fedavg/iid_25/`:**
- `global_model.pth` – Aggregated global model after all rounds
- `metrics.json` – Final global metrics (accuracy, F1, confusion matrix)
- `history.json` – Per-round accuracy & loss
- `global_confusion_matrix.png` – Visualization
- `accuracy_curve.png` – Global accuracy vs. round
- `loss_curve.png` – Global loss vs. round
- `client_accuracy.png` – Per-client accuracy comparison

**Additional outputs in `results/federated/local_clients/iid_25/client_{1-5}/`:**
- `local_model.pth` – Client's final local model
- `metrics.json` – Client's local metrics
- `loss_curve.png` – Client's training loss curve
- `confusion_matrix.png` – Client's test confusion matrix

### 5. Run Federated Learning (FedProx)
More stable for non-IID data (adds proximal regularization):
```bash
python server/fedprox_runner.py 25 non_iid
```

**Arguments & outputs:** Same as FedAvg, results in `results/federated/fedprox/`

---

## 📊 Understanding the Architecture

### Model: 1D CNN for Time-Series
Located in `models/cnn_model.py`:

```
Input (batch_size, sequence_length, num_features)
  ↓
Conv1d(num_features → 32, kernel=5) + BatchNorm + ReLU + MaxPool
  ↓
Conv1d(32 → 64, kernel=3) + BatchNorm + ReLU + MaxPool
  ↓
AdaptiveAvgPool1d (global aggregation)
  ↓
FC(64 → 128) + ReLU + Dropout(0.3)
  ↓
FC(128 → 1) + Sigmoid
  ↓
Output (batch_size, 1) – Anomaly probability [0, 1]
```

**Why this design?**
- **Conv1d layers:** Extract temporal patterns from multivariate time-series
- **MaxPool:** Reduce dimensionality, capture important temporal features
- **Batch Norm:** Stabilize training in federated setting
- **AdaptiveAvgPool:** Handle variable-length sequences
- **Sigmoid:** Binary classification (normal vs. anomaly)

### Federated Learning Flow

#### FedAvg (Simple Averaging)
```
Round 1:
  Server → broadcast global_model_weights to [Client_1, ..., Client_5]
  Client_i: load_weights(global) → train_locally(20 epochs) → return local_weights
  Server: avg_weights = mean([Client_1_weights, ..., Client_5_weights])
  Server: update global_model_weights = avg_weights

Round 2, 3: Repeat
```

#### FedProx (Proximal Regularization)
```
Same as FedAvg, but during client training:

Loss = Classification_Loss + (μ/2) * ||local_weights - global_weights||²

This regularization term pulls client weights toward global model,
reducing divergence in heterogeneous (non-IID) settings.
```

### Data Partitioning

**IID (Independent & Identically Distributed):**
- Shuffle all data randomly and distribute equally to clients
- Simulates realistic scenario where each device has similar data distribution

**Non-IID (Heterogeneous):**
- Partition by class/label (each client sees fewer classes)
- Simulates reality where devices have different local data characteristics
- Tests algorithm robustness to distribution skew

---

## 🔧 Configuration & Customization

### Modify Training Parameters

**Centralized training** (`train/centralized.py`):
```python
BATCH_SIZE = 64          # Batch size for training
EPOCHS = 20              # Number of training epochs
LEARNING_RATE = 0.001    # Adam optimizer learning rate
DEVICE = "cuda" / "cpu"  # Automatic
```

**Federated client** (`client/fl_client.py`):
```python
LOCAL_EPOCHS = 20        # Local training epochs per round
BATCH_SIZE = 64
LEARNING_RATE = 0.001
mu = 0.01                # FedProx regularization coefficient (larger = more regulation)
```

**Federated server** (`server/fl_server_avg.py`):
```python
NUM_CLIENTS = 5          # Number of federated clients
ROUNDS = 3               # Communication rounds
DATASET_TYPE = "25"      # Feature dimension
PARTITION_TYPE = "iid"   # "iid" or "non_iid"
```

### Future: Move to Config File
Hardcoded values should be replaced with a YAML config:
```yaml
# config.yaml
centralized:
  batch_size: 64
  epochs: 20
  learning_rate: 0.001

federated:
  num_clients: 5
  rounds: 3
  local_epochs: 20
  mu_fedprox: 0.01
```

Usage: `python server/fedavg_runner.py --config config.yaml`

---

## 📈 Interpreting Results

### Metrics Explained

| Metric | Meaning | Target |
|--------|---------|--------|
| **Accuracy** | % of correct predictions | High (>0.95) |
| **Precision** | % of predicted anomalies that are real | High (>0.9) |
| **Recall** | % of real anomalies detected | High (>0.9) |
| **F1 Score** | Harmonic mean of precision & recall | High (>0.9) |
| **Confusion Matrix** | [TN, FP; FN, TP] – break down by class | Diagonal high, off-diagonal low |

### Files Generated

**JSON Metrics** (`results/*/metrics.json`):
```json
{
  "dataset_type": "25",
  "accuracy": 0.9543,
  "precision": 0.9512,
  "recall": 0.9574,
  "f1_score": 0.9543,
  "confusion_matrix": [[234, 12], [10, 256]]
}
```

**Training History** (`results/*/history.json`):
```json
{
  "accuracy": [0.82, 0.88, 0.92, ...],
  "loss": [0.45, 0.35, 0.25, ...]
}
```

**Plots:**
- **accuracy_curve.png** – Global accuracy improving over FL rounds
- **loss_curve.png** – Global loss decreasing over rounds
- **client_accuracy.png** – Bar chart of per-client accuracy
- **global_confusion_matrix.png** – Final confusion matrix heatmap

### Comparing Centralized vs. Federated

```
Scenario: 25 features, IID data
─────────────────────────────────────────
               Accuracy   F1 Score
─────────────────────────────────────────
Centralized     0.9540     0.9542  (all data, non-private)
FedAvg          0.9512     0.9511  (distributed, private)
FedProx         0.9518     0.9517  (distributed, more robust)
─────────────────────────────────────────
```

**Insights:**
- FedAvg: Slight degradation due to data partitioning
- FedProx: Better than FedAvg on heterogeneous data
- Privacy gain: Model weights never centralized (only aggregated)

---

## 🛠️ Advanced Usage

### Train on Custom Dataset

1. **Prepare data in `data/raw/`:**
   ```
   train/: X_train samples (e.g., channel_001.npy shape: [sequence_length])
   test/:  X_test samples (e.g., channel_001.npy shape: [sequence_length])
   labeled_anomalies.csv: with columns [chan_id, anomaly_sequences, class]
   ```

2. **Update `data_manager.py`** to match your data format

3. **Run with your feature dimension:**
   ```bash
   python server/fedavg_runner.py <YOUR_NUM_FEATURES> iid
   ```

### Extend Model Architecture

Edit `models/cnn_model.py` or create new model:
```python
class LSTMModel(nn.Module, BaseModel):
    def __init__(self, num_features):
        super().__init__()
        self.lstm = nn.LSTM(num_features, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.sigmoid(self.fc(lstm_out[:, -1, :]))
```

Update `models/model_selector.py`:
```python
def get_model(model_type, num_features):
    if model_type == "cnn":
        return CNNModel(num_features)
    elif model_type == "lstm":
        return LSTMModel(num_features)
```

### Run Custom FL Algorithm

Modify `server/fl_server_avg.py` aggregation logic:
```python
def federated_average(local_weights):
    # Current: simple mean
    # Custom: weighted by client dataset size, median instead of mean, etc.
    ...
```

---

## ❌ Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError: data/raw/train`
- **Fix:** Download NASA satellite data; ensure raw data is in `data/raw/` directory

**Issue:** `RuntimeError: Expected 3D input (got 2D)`
- **Fix:** Ensure input data shape is `(batch_size, sequence_length, num_features)`; check `preprocessor_v2.py`

**Issue:** `CUDA out of memory`
- **Fix:** Reduce `BATCH_SIZE` (try 32 or 16); use CPU with `torch.device("cpu")`

**Issue:** Model not improving after rounds
- **Possible causes:**
  - Learning rate too high → reduce `LEARNING_RATE`
  - Too few local epochs → increase `LOCAL_EPOCHS`
  - Non-IID data → use FedProx instead of FedAvg
  - Data imbalance → check anomaly ratio in dataset

**Issue:** Different results on re-run
- **Fix:** Set random seeds for reproducibility:
  ```python
  import torch, numpy as np, random
  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)
  ```

---

## 📚 Key References

### Federated Learning Papers
- **FedAvg:** McMahan et al. (2017) – "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- **FedProx:** Li et al. (2020) – "Federated Optimization in Heterogeneous Networks"

### Time-Series Anomaly Detection
- Anomaly detection in satellite telemetry (NASA data context)
- 1D CNN effectiveness for temporal pattern extraction

### Libraries Used
- **Flower (flwr):** Lightweight federated learning framework
- **PyTorch:** Deep learning backend
- **scikit-learn:** Metrics & preprocessing

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Complete `run_orchestrator.py` (unified CLI entry point)
- [ ] Build `dashboard.py` (Dash web UI for results visualization)
- [ ] Add unit tests (`tests/`)
- [ ] Implement differential privacy
- [ ] Add async FL for stragglers
- [ ] Support for Transformer models

---

## 📝 License

[Specify your license here – e.g., MIT, Apache 2.0]

---

## 📧 Contact & Questions

For questions or issues, please open a GitHub issue or contact the repository maintainer.

---

**Last Updated:** 2026-06-22
