# Environment Setup

Tested on Python 3.11. Python 3.10–3.12 should work.

---

## 1. Clone the repo

```bash
git clone https://github.com/sourasb05/Federated_IDS.git
cd Federated_IDS
git checkout toy-experiments
```

---

## 2. Create a virtual environment

### Option A — plain `venv` (CPU / any platform)

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — Conda (recommended for Apple Silicon MPS or CUDA GPU)

```bash
conda create -n fedids python=3.11 -y
conda activate fedids

# PyTorch — pick ONE line that matches your hardware:

# Apple Silicon (MPS)
conda install pytorch==2.0.1 torchvision torchaudio -c pytorch -y

# NVIDIA GPU (CUDA 11.8)
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# CPU only
conda install pytorch==2.0.1 torchvision torchaudio cpuonly -c pytorch -y

# Remaining dependencies
pip install -r requirements.txt
```

---

## 3. Verify installation

```bash
python - <<'EOF'
import torch, sklearn, numpy, pandas, matplotlib, scipy, tqdm, flask
print("torch   :", torch.__version__)
print("sklearn :", sklearn.__version__)
print("numpy   :", numpy.__version__)
print("pandas  :", pandas.__version__)
print("MPS     :", torch.backends.mps.is_available())
print("CUDA    :", torch.cuda.is_available())
EOF
```

---

## 4. Run the EFL system

```bash
cd src

# TabDDPM only, localrepair attack type, 1 time step
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

# Skip Phase 1 — load saved denoisers directly
python main.py \
    --algorithm efl \
    --generator tabddpm \
    --attacktype localrepair \
    --load_decoders
```

---

## 5. Run the t-SNE overlap analysis

```bash
# From the project root
python toy/tsne_tabddpm_vs_real.py
```

Output saved to `toy/tsne_tabddpm_localrepair_var15_base.png` (path set inside the script).
Edit `DATA_DIR`, `CKPT_PATH`, and `OUT_PATH` at the top of the script to switch domain/client.

---

## Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--generator` | `both` | `zirvae`, `tabddpm`, or `both` |
| `--attacktype` | all domains | filter by substring, e.g. `localrepair`, `worstparent` |
| `--tot_time_steps` | `4` | number of EFL time steps to run |
| `--zirvae_epochs` | `200` | ZI-RVAE training epochs per client |
| `--tabddpm_epochs` | `500` | TabDDPM training epochs per class per client |
| `--overlap_p` | `0.0` | domain overlap fraction across clients (0=disjoint, 1=shared) |
| `--load_decoders` | off | skip Phase 1, load saved checkpoints instead |

---

## Saved model paths

Denoiser checkpoints are saved under:
```
saved_models_efl/<attacktype>/tabddpm_client<id>_<domain>_<class>.pt
```

Example:
```
saved_models_efl/localrepair/tabddpm_client0_localrepair_var20_dec_attack.pt
saved_models_efl/localrepair/tabddpm_client0_localrepair_var20_dec_benign.pt
```
