# Federated IDS — Event-Driven Continual Federated Learning (EFL)

An **Event-Driven Continual Federated Learning** system for IoT Intrusion Detection.  
Clients use generative models (TabDDPM / ZI-RVAE) to synthesize attack data locally, share only model weights (no raw data), and train a distillation-based student classifier that is resilient to catastrophic forgetting.

---

## System Overview

```
Phase 1  →  Each client trains a generative model (TabDDPM / ZI-RVAE) on local attack data
Phase 2a →  Server fingerprints clients via MMD² on attack samples (admission gate)
Phase 2b →  Server trains a multi-class Teacher model on pooled synthetic attack data
Phase 3  →  Teacher logits broadcast back to clients
Phase 4  →  Each client trains a binary Student via task loss + KL distillation (Distilled Replay)
```

Metrics reported per client: Accuracy, F1 (macro + weighted), Baseline Student vs Distilled Student comparison.

---

## Folder Structure

```
Federated_IDS/
├── src/
│   ├── main.py              # entry point
│   ├── client_EFL.py        # EFL client: generator training, student training, distillation
│   ├── server_EFL.py        # EFL server: MMD admission, Teacher training, Phase 4
│   ├── models.py            # TabDDPM, ZI-RVAE, TeacherModel, StudentModel
│   ├── utils.py             # data loading, windowing, normalization, CLI args
│   └── evaluate_model.py    # evaluation utilities
├── toy/
│   └── tsne_tabddpm_vs_real.py   # t-SNE + precision/recall overlap analysis
├── attack_data/
│   └── <domain>/            # one subfolder per attack domain, each with CSVs
├── saved_models_efl/
│   └── <attacktype>/        # denoiser checkpoints saved here
├── results/
│   └── <attacktype>/        # JSON metrics saved here
├── requirements.txt
├── SETUP.md                 # full setup instructions
└── README.md
```

---

## Quick Setup

See [SETUP.md](SETUP.md) for full instructions. Short version:

```bash
git clone https://github.com/sourasb05/Federated_IDS.git
cd Federated_IDS
git checkout toy-experiments

# Conda (recommended for Apple Silicon / CUDA GPU)
conda create -n fedids python=3.11 -y && conda activate fedids
conda install pytorch==2.0.1 -c pytorch -y   # add MPS/CUDA flags as needed — see SETUP.md
pip install -r requirements.txt
```

---

## Running the EFL System

```bash
cd src

# TabDDPM only, localrepair domains, 1 time step
python main.py \
    --algorithm efl \
    --generator tabddpm \
    --attacktype localrepair \
    --tot_time_steps 1 \
    --tabddpm_epochs 500

# Both generators, all domains, 4 time steps
python main.py \
    --algorithm efl \
    --generator both \
    --tot_time_steps 4 \
    --zirvae_epochs 200 \
    --tabddpm_epochs 500

# Skip Phase 1 — load saved denoiser checkpoints directly
python main.py \
    --algorithm efl \
    --generator tabddpm \
    --attacktype localrepair \
    --load_decoders
```

---

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--algorithm` | `efl` | FL algorithm: `efl`, `fedavg`, `replay`, … |
| `--generator` | `both` | Generator: `tabddpm`, `zirvae`, or `both` |
| `--attacktype` | all | Filter domains by substring, e.g. `localrepair`, `worstparent` |
| `--tot_time_steps` | `4` | Number of EFL time steps |
| `--tabddpm_epochs` | `500` | TabDDPM training epochs per class per client |
| `--zirvae_epochs` | `200` | ZI-RVAE training epochs per client |
| `--overlap_p` | `0.0` | Domain overlap across clients (0 = disjoint, 1 = fully shared) |
| `--load_decoders` | off | Skip Phase 1, load saved checkpoints |
| `--window_size` | `10` | Sliding window length |
| `--batch_size` | `64` | Mini-batch size |
| `--lr` | `0.001` | Learning rate |
| `--seed` | `42` | Random seed |

---

## Generators

### TabDDPM (recommended)
Gaussian diffusion over tabular/time-series windows.  
- Architecture: sinusoidal time embedding → 4 × residual MLP blocks (hidden=512) → output  
- ~2.57M parameters per class denoiser, ~9.84 MB per class  
- Conditional: one denoiser per class (benign / attack)  
- Checkpoint saved immediately after each class: `saved_models_efl/<attacktype>/tabddpm_client<id>_<domain>_<class>.pt`

### ZI-RVAE
Zero-Inflated Recurrent VAE with GRU encoder.  
- Monotone KL schedule with free-bits to prevent posterior collapse  
- ~0.8 MB per decoder — much lighter than TabDDPM  

---

## t-SNE Overlap Analysis

Visualise and quantify how well TabDDPM synthetic attack samples match real ones:

```bash
# Edit DATA_DIR, CKPT_PATH, OUT_PATH at the top of the script to select domain/client
python toy/tsne_tabddpm_vs_real.py
```

**Metrics printed:**

| Metric | Meaning |
|---|---|
| Precision | % synthetic timesteps inside real data manifold (fidelity) |
| Recall | % real timesteps covered by synthetic manifold (coverage) |
| F1 overlap | Harmonic mean of precision and recall |
| MMD² | RBF-kernel distribution distance (lower = closer) |

Colors follow the Wong (2011) color-blind-friendly palette: **blue = real**, **orange = synthetic**.

---

## Data Format

- Place data under `attack_data/<domain_name>/` — one subfolder per attack scenario  
- Each CSV must have a `label` column: `0` = benign, `1` = attack  
- All other columns are numeric features  
- Files are sorted; first 16 → train, last 4 → test  
- 20 files per domain expected (code takes `[:20]`)

---

## Saved Model Paths

```
saved_models_efl/<attacktype>/tabddpm_client<id>_<domain>_attack.pt
saved_models_efl/<attacktype>/tabddpm_client<id>_<domain>_benign.pt
saved_models_efl/<attacktype>/zirvae_decoder_client<id>_<domain>.pt
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `FileNotFoundError: attack_data` | Create `attack_data/<domain>/` and place CSVs there |
| `'label' column missing` | Every CSV must have a `label` column with 0/1 values |
| Phase 1 prints `ZI-RVAE` when using `--generator tabddpm` | Ensure `client_EFL.py` is up to date — pull latest |
| Denoiser checkpoint not found with `--load_decoders` | Run Phase 1 first (without `--load_decoders`) to save checkpoints |
| CUDA/MPS not detected | Check PyTorch install — see [SETUP.md](SETUP.md) |

---

## .gitignore (suggested)

```
__pycache__/
*.pyc
.venv/
venv/
attack_data/
saved_models_efl/
results/
*.png
*.DS_Store
efl_metrics.json
```
