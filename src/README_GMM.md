# GMM Generative Replay — Architecture

## Overview

Federated continual learning where each client uses **per-class Bayesian Gaussian Mixture Models (GMM)** to remember past domains and replay synthetic data when learning new ones.

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                        server_GMM.py                            ║
║                                                                  ║
║   Global LSTM Model  ──────────────────────────────────────┐    ║
║        │  FedAvg aggregate ←── local weights               │    ║
║        ↓                                                    │    ║
║   ┌─────────┐   ┌─────────┐   ┌─────────┐                 │    ║
║   │Client 0 │   │Client 1 │   │Client 2 │  ← ClientGMM    │    ║
║   └────┬────┘   └────┬────┘   └────┬────┘                 │    ║
║        └─────────────┴─────────────┘                        │    ║
║                   local weights                              │    ║
╚══════════════════════════════════════════════════════════════╝
```

---

## ClientGMM — Continual Learning Flow

```
╔══════════════════════════════════════════════════════════════════╗
║                  ClientGMM  (one per client)                    ║
║                                                                  ║
║   Assigned domains: [domain_A, domain_B, domain_C, ...]         ║
║                                                                  ║
║   TIME STEP t=0                                                  ║
║   ┌──────────────────────────────────────────────────────────┐  ║
║   │  Domain A  (real data, T=10 timesteps, F features)       │  ║
║   │                                                           │  ║
║   │  load_data() → (B, T, F) tensors                         │  ║
║   │       ↓  flatten to (N, T*F)                             │  ║
║   │  _train_generator()                                       │  ║
║   │       ↓                                                   │  ║
║   │  ┌────────────────────────────────┐                       │  ║
║   │  │  Per-class BayesianGMM fitting │                       │  ║
║   │  │                                │                       │  ║
║   │  │  X[y==0] → GMM_0  (benign)    │  ← EM algorithm       │  ║
║   │  │  X[y==1] → GMM_1  (attack)    │  ← no gradients       │  ║
║   │  │                                │                       │  ║
║   │  │  Each GMM: up to 10 Gaussian   │                       │  ║
║   │  │  components, Bayesian prior    │                       │  ║
║   │  │  prunes unused ones            │                       │  ║
║   │  └────────────────────────────────┘                       │  ║
║   │       ↓  frozen — never updated again                     │  ║
║   │  generators[domain_A] = {0: GMM_0, 1: GMM_1}             │  ║
║   │                                                           │  ║
║   │  Train LSTM on real Domain A data                         │  ║
║   └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║   TIME STEP t=1                                                  ║
║   ┌──────────────────────────────────────────────────────────┐  ║
║   │  Domain B  (new real data arrives)                        │  ║
║   │                                                           │  ║
║   │  _train_generator(domain_B)  → GMM_0, GMM_1 for B        │  ║
║   │                                                           │  ║
║   │  _build_replay_loader()                                   │  ║
║   │       ↓                                                   │  ║
║   │  ┌────────────────────────────────────────────────────┐  │  ║
║   │  │  _synthesize_proportional(domain_A, n=500)         │  │  ║
║   │  │                                                     │  │  ║
║   │  │  P(class=0) = 0.7  → sample 350 from GMM_0         │  │  ║
║   │  │  P(class=1) = 0.3  → sample 150 from GMM_1         │  │  ║
║   │  │                                                     │  │  ║
║   │  │  Labels are EXACT  (no random marginal draw)        │  │  ║
║   │  │  GMM_0.sample() → always class 0 windows           │  │  ║
║   │  │  GMM_1.sample() → always class 1 windows           │  │  ║
║   │  │                                                     │  │  ║
║   │  │  reshape (N, T*F) → (N, T, F)                      │  │  ║
║   │  └─────────────────────────────┬──────────────────────┘  │  ║
║   │                                ↓                          │  ║
║   │  Train LSTM on:                                           │  ║
║   │    ├── real Domain B batches                              │  ║
║   │    └── synthetic Domain A replay batches  ←──────────────┘  ║
║   └──────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║   TIME STEP t=2  (same pattern, now replay A + B)               ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Inside BayesianGaussianMixture

```
   Class 0 windows  (N_0, T*F)
         │
         ▼
   ┌─────────────────────────────────────────┐
   │         EM Algorithm (iterative)        │
   │                                         │
   │  K=10 Gaussian components               │
   │                                         │
   │  Component 1: μ₁, Σ₁, weight w₁        │
   │  Component 2: μ₂, Σ₂, weight w₂        │
   │  ...                                    │
   │  Component K: μₖ, Σₖ, weight wₖ        │
   │                                         │
   │  Bayesian prior → weights near 0        │
   │  get pruned automatically               │
   │  (e.g. only 3 of 10 survive)            │
   └──────────────────┬──────────────────────┘
                      │
             at replay time
                      │
                      ▼
         GMM_0.sample(n=350)
              │
              ├── Pick component k with prob w_k
              └── Sample x ~ N(μ_k, Σ_k)
                       ↓
              (350, T*F)  all class 0
```

---

## Data Flow

```
  Real data (CSV)
       │
       ▼
  load_data()  →  (B, T, F)  DataLoader
       │
       ├──→  LSTM training  (current domain, real)
       │
       └──→  flatten to (N, T*F)
                  │
                  ▼
             BayesianGMM.fit()   [one per class, per domain]
                  │
                  ▼   [next time step]
             GMM.sample()
                  │
                  ▼
             reshape (N, T*F) → (N, T, F)
                  │
                  ▼
             LSTM training  (past domains, synthetic replay)
```

---

## Key Difference vs TVAE / CTVAE

```
  TVAE/CTVAE replay:
    z ~ N(0,I)  →  Decoder  →  x̂  →  label sampled from P(Y)
                                         ↑
                                    can be wrong class

  GMM replay:
    GMM_0.sample()  →  x̂  →  label = 0  ✓ always correct
    GMM_1.sample()  →  x̂  →  label = 1  ✓ always correct
```

---

## Why BayesianGMM over plain GMM

| Property | Plain GMM | BayesianGMM |
|---|---|---|
| Number of components | Must specify exactly | Set large, model prunes unused |
| Overfitting risk | High (all K used) | Low (sparse Dirichlet prior) |
| Mode collapse | Possible | Less likely |
| Cross-validation needed | Yes | No |

---

## Generator Registry Structure

```python
generators = {
    "domain_A": {
        0: {"gmm": BayesianGaussianMixture, "n_obs": 3200},  # benign
        1: {"gmm": BayesianGaussianMixture, "n_obs": 800},   # attack
    },
    "domain_B": {
        0: {"gmm": BayesianGaussianMixture, "n_obs": 2900},
        1: {"gmm": BayesianGaussianMixture, "n_obs": 1100},
    },
}
```

Each GMM is **frozen after fitting** — it is never updated in later time steps.
Past domain knowledge is locked in at the time it was first seen.

---

## Files

| File | Role |
|---|---|
| `client_GMM.py` | Per-client GMM fitting, replay sampling, LSTM training |
| `server_GMM.py` | FedAvg aggregation, dynamic client selection, BWT tracking |
| `main.py` | Entry point — `--algorithm gmm` |
| `utils.py` | `load_data()` returns `(B, T, F)` tensors |

---

## How to Run

```bash
python src/main.py --algorithm gmm --window_size 10 --global_iters 20 --local_epochs 5
```

Results saved to `results/gmm_metrics.json`.
Generator quality plots saved to `results/gmm_plots/`.
