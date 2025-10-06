# Federated IDS (LSTM) — Sequential Windowed Training Across Domains

This repo trains a **federated** LSTM classifier over multiple **IDS “domains”** (each domain = a folder of CSVs).
Per round, the global model is sent to each domain, trained locally, then **aggregated by data-size weighting** (FedAvg style).
Evaluation is done **per domain** with Accuracy, F1, AUC (binary), Confusion Matrix, and (optionally) loss.


### requirements.txt

```text
numpy
pandas
scikit-learn
scipy
matplotlib
torch
wandb
```

> Note: Installing `torch` sometimes depends on your CUDA version. If you need GPU support, follow the command from [https://pytorch.org](https://pytorch.org) for your system (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu121`).

---

## Features

* Sliding-window sequence creation from raw per-timestep CSVs
* Train/test split per domain (file-level)
* Per-domain **min–max normalization** (using *train* files only; safe, NaN/Inf-proof)
* **LSTMClassifier** (two-layer MLP head) and an optional **LSTMModelWithAttention**
* Federated loop with **size-weighted model aggregation**
* Metrics: Accuracy, F1 (binary or macro), ROC-AUC (binary), Confusion Matrix
* CUDA/MPS detection (GPU if available)

---

## Folder Structure

```
repo/
├─ src/
│  ├─ main.py                 # entry point (training + federation + eval)
│  ├─ evaluate_model.py       # evaluation utilities
│  ├─ models.py               # LSTMClassifier & LSTMModelWithAttention
│  └─ utils.py                # data loading, windowing, normalization, helpers
├─ attack_data/               # (you create this) domains and CSVs live here
│  ├─ domain_A/
│  │  ├─ ..._1_60_sec.csv
│  │  ├─ ..._2_60_sec.csv
│  │  └─ ...
│  └─ domain_B/
│     └─ ...
├─ results/                   # (created automatically if you save JSON later)
├─ requirements.txt
└─ README.md
```

---

## Data Expectations

* Place all data under: `attack_data/<domain_name>/`
* Each CSV **must** contain a column named `label` with integer values:

  * `0` = benign
  * `1` = attack
* Other columns are numeric features.
* Filenames are sorted and (optionally) parsed with a pattern like `..._<index>_60_sec.csv` so that the code can keep a consistent ordering per domain.

**Windowing**
From each CSV, we build sliding windows:

* `window_size` = sequence length
* `step_size` = stride between windows
* Each window becomes a single training sample, labeled with the **last** time step’s label.

---

## Installation

```bash
# (Recommended) Use Python 3.9+ and a fresh virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# If you need a specific CUDA build of PyTorch, install it from pytorch.org instead of the generic `torch` above.
```

---

## Quick Start

From the `src/` directory:

```bash
python main.py \
  --input_size 140 \
  --hidden_size 64 \
  --num_layers 1 \
  --output_size 2 \
  --window_size 10 \
  --step_size 3 \
  --batch_size 64 \
  --global_iters 5 \
  --local_epochs 5 \
  --lr 0.001 \
  --save_dir ./results
```

You should see logs like:

```
Using device: cuda
--- Global Iteration 1 / 5 ---
Training on domain: blackhole_var10_base
Epoch 1/5, Loss: 0.6934
...
[blackhole_var10_base] n=12345 | acc=0.91 | f1=0.90 | auc=0.95 | loss=0.21 | cm=[[...]]
...
=== Aggregates (computed from per-domain) ===
Accuracy: 92.31%
F1: 0.9031
AUC : 0.9487
Loss : 0.2142
```

---

## Command-Line Arguments

| Arg              |     Default | Description                                                                                                                                                                                 |
| ---------------- | ----------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--input_size`   |         140 | Number of flattened features per time step (after windowing this becomes your time dimension × features, but code currently flattens time into features and uses seq len 1; see **Notes**). |
| `--hidden_size`  |          64 | LSTM hidden size.                                                                                                                                                                           |
| `--num_layers`   |           1 | LSTM layers.                                                                                                                                                                                |
| `--output_size`  |           2 | Number of classes (binary IDS).                                                                                                                                                             |
| `--window_size`  |          10 | Sliding window length (per CSV).                                                                                                                                                            |
| `--step_size`    |           3 | Stride between windows.                                                                                                                                                                     |
| `--batch_size`   |          64 | Mini-batch size.                                                                                                                                                                            |
| `--global_iters` |           5 | Federated rounds.                                                                                                                                                                           |
| `--local_epochs` |           5 | Local epochs per domain per round.                                                                                                                                                          |
| `--lr`           |       0.001 | Local optimizer LR (Adam).                                                                                                                                                                  |
| `--seed`         |          42 | Random seed.                                                                                                                                                                                |
| `--save_dir`     | `./results` | Where to store outputs if you add saving.                                                                                                                                                   |

---

## Notes on the Model Shapes

* In `utils.load_data`, windows are flattened and then reshaped to `(B, 1, feature_dim)`, so the LSTM sequence length is **1** and `feature_dim = (#features × window_size)`.

  * This keeps your model as an LSTM while effectively acting like a 1-step sequence with a large feature vector.
  * If you prefer a **true temporal** LSTM (seq len = `window_size`), change the data prep to keep shape `(B, window_size, #features)` and set `input_size = #features`.

* `LSTMClassifier` returns `(logits, (h_n, c_n))`.

* `LSTMModelWithAttention` returns `(logits, context)` and can be swapped into `main.py` if desired.

---

## Evaluation Details

`evaluate_model.eval_global` computes:

* **Accuracy**
* **F1**: binary if 2 classes; macro otherwise
* **AUC**: only for binary when both classes are present in the batch
* **Confusion Matrix**
* **Loss** (Cross-Entropy), when tensors are compatible

Per-domain results are printed each round. A simple weighted aggregate (by number of samples) is printed at the end of each round.

---

## Data & Naming Conventions

* Domains are discovered from subfolders in `attack_data/`.
* Per domain, files are sorted; first 16 files → **train**, last 4 files → **test** (you can change this in `utils.load_data`).
* CSVs must include a **`label`** column; all other columns are treated as numeric features.

---

## Troubleshooting

* **`FileNotFoundError: .../attack_data`**
  Ensure your data directory exists: `attack_data/<domain>/*.csv`.

* **`'label' column missing`**
  Each CSV must have a `label` column with 0/1 values.

* **AUC is `None`**
  Happens when a test batch contains only one class (all 0s or all 1s). That’s expected behavior.

* **CUDA not used**
  You’ll see `Using device: cpu`. Check PyTorch install and GPU drivers.

---

## Optional: .gitignore (suggested)

Create a `.gitignore` in repo root:

```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
.env
.ipynb_checkpoints/

# Data & results
attack_data/
results/
wandb/

# OS
.DS_Store
Thumbs.db
```

> Keep `attack_data/` out of Git if it’s large or sensitive. You can version a small sample instead.

---
