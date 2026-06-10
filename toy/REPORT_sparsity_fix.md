# Report: Fixing Distribution Mismatch in VAE-Generated Synthetic Data
## Problem, Diagnosis, and Solution

**Dataset:** `blackhole_var10_base/10_features_timeseries_60_sec.csv`
**Model family:** Variational Autoencoder (VAE) for generative replay in Federated IDS
**Evaluation metric:** Kolmogorov–Smirnov (KS) test per feature, TSTR/TRTS F1

---

## 1. The Problem

After training a TimeVAE (Conv1D) to generate synthetic network traffic windows, the
synthetic data failed the KS test on **all 14 features** (KS mean = 0.45, pass rate = 0/14).
This means the marginal distribution of every feature in the synthetic data was
statistically distinguishable from the real data.

| Metric | TimeVAE (no fix) |
|---|---|
| Mean KS statistic | 0.459 |
| KS pass rate | 0 / 14 |
| TSTR F1 | 1.000 |
| TRTS F1 | 0.992 |

Interestingly, TSTR/TRTS scores were near-perfect, meaning the synthetic data
preserved class boundaries well. The distribution mismatch was therefore not hurting
downstream classifier quality — but it did represent a genuine fidelity problem.

---

## 2. Diagnosis

### 2.1 Feature audit

Inspection of the 14 RPL routing-metric features revealed a severe **zero-inflation** problem:

| Feature | Zero fraction | Description |
|---|---|---|
| disr | 1.000 | All-zero — always 0 in this capture |
| diss | 1.000 | All-zero |
| disr.1 | 1.000 | All-zero |
| diss.1 | 1.000 | All-zero |
| diar | 0.865 | 86.5% zeros, rare non-zero spikes |
| diar.1 | 0.865 | 86.5% zeros |
| dior | 0.342 | 34% zeros |
| dios | 0.364 | 36% zeros |
| diar | 0.865 | 86.5% zeros |
| tots | 0.313 | 31% zeros |
| dior.1 | 0.342 | 34% zeros |
| dios.1 | 0.364 | 36% zeros |
| tots.1 | 0.313 | 31% zeros |

12 out of 14 features had more than 30% zero values.
The remaining 2 (`rank`, `rank.1`) were dense continuous features.

### 2.2 Why MSE-VAE fails on zero-inflated data

A standard VAE with Mean Squared Error (MSE) reconstruction loss minimises:

```
L_recon = E[ (x - x_hat)^2 ]
```

For a feature that is zero 86% of the time (e.g. `diar`), the optimal decoder output
under MSE is the **mean of the distribution** — which is approximately zero.
The decoder learns to output a value very close to zero for every sample, completely
ignoring the rare but important non-zero spikes.

The resulting synthetic distribution is a **narrow Gaussian centred near zero**,
whereas the real distribution is a **spike-at-zero + long tail** (a zero-inflated
distribution). These two shapes are very different, hence the KS test fails.

```
Real distribution:                VAE output:
│                                 │
█                                 │    ▄▄▄
█                                 │  ▄█████▄
█  ▄  ▄  ▄                       │▄▄███████▄▄
└────────────────  value          └────────────────  value
0   0.5  1.0  1.5                 0   0.5  1.0  1.5

Spike at zero + sparse spikes     Smooth Gaussian near zero
```

This is a **fundamental limitation of MSE loss** for sparse/zero-inflated data —
it is not an architecture problem. Both TimeVAE (Conv1D) and RVAE (GRU) suffer
equally from it (KS mean 0.45 for both).

---

## 3. Solutions Explored

### Option 1 — Accept the mismatch (already implemented baseline)
Since TSTR/TRTS are near-perfect, the synthetic data is good enough for the
federated replay objective. KS failure does not mean the data is useless.

**When to use:** When distribution fidelity reporting is not required.

---

### Option 2 — log1p pre-processing (implemented in `lstmvae_toy.py`)

Apply `log1p(x)` to all sparse features before normalisation. This compresses the
zero-inflated distribution into a shape closer to Gaussian, which MSE-VAE can model.

```python
# Before min-max normalisation
df[sparse_cols] = np.log1p(df[sparse_cols])

# After synthesis — undo with:
X_syn[sparse_cols] = np.expm1(X_syn[sparse_cols])
```

**Result:** RVAE KS mean improved from 0.447 → 0.422 (1 feature passed).
Improvement was modest because the all-zero features (disr, diss) are unaffected
by log1p (log1p(0) = 0), and the bimodal class structure still creates a mixed
distribution the KS test struggles with.

---

### Option 3 — Zero-Inflated VAE (ZI-VAE) (implemented in `zivae_toy.py`)

Model the data-generating process explicitly as a **mixture**:

```
P(x_feature) = P(zero) * δ(0)  +  P(non-zero) * Gaussian(μ, σ)
```

This requires a **two-head decoder** per sparse feature:

| Head | Output | Loss |
|---|---|---|
| Bernoulli gate | P(feature > 0) — sigmoid | Binary Cross-Entropy |
| Gaussian value | E[feature \| feature > 0] — sigmoid | MSE |

Dense features use only the Gaussian head (standard MSE).

#### Architecture

```
Encoder:  (B, T, F) → Flatten → MLP → (μ, log_var)
                                              │
                                         reparameterise
                                              │ z
                          ┌───────────────────┴──────────────────┐
                   Gaussian MLP                           Gate MLP
                  (latent → T×F)                      (latent → T×S)
                          │                                       │
                    gauss_out (B,T,F)                  gate_logit (B,T,S)
                    [sigmoid]                          [Bernoulli BCE]
                          └─────────── merged ────────────────────┘
                                          │
                              final output (B, T, F):
                              dense features  → gauss_out
                              sparse features → gate * gauss_out
```

where S = number of sparse features (12 in this dataset).

#### Loss function

```
L = (recon_weight / T×F) × MSE(gauss_out, x)
  + 0.1 × Σ_sparse BCE(sigmoid(gate_logit), (x > 0).float())
  + kl_weight × KL(q(z|x) || p(z))
```

The BCE term trains the gate to predict whether each sparse feature is zero or not,
independently of the magnitude. During synthesis, the gate is sampled from a Bernoulli
distribution, correctly producing zeros with the right frequency.

#### Synthesis

```python
gate_prob = sigmoid(gate_logit)          # P(feature != 0) per timestep
gate      = Bernoulli(gate_prob).sample()
output    = gate * gauss_output          # zero if gate=0, value if gate=1
```

---

## 4. Results

| Metric | TimeVAE (baseline) | log1p + TimeVAE | ZI-VAE |
|---|---|---|---|
| Mean KS statistic | 0.459 | 0.488 | **0.182** |
| KS pass rate | 0 / 14 | 0 / 14 | **4 / 14** |
| TSTR F1 | 1.000 | 1.000 | **1.000** |
| TRTS F1 | 0.992 | 1.000 | **1.000** |
| TSTR/TRTS gap | 0.009 | 0.000 | **0.000** |

The ZI-VAE achieved a **60% reduction in mean KS statistic** (0.459 → 0.182)
and moved 4 features from failing to passing the KS test, while maintaining
perfect TSTR/TRTS scores.

---

## 5. Why 10 Features Still Fail

The 10 remaining KS failures fall into two categories:

1. **All-zero features** (`disr`, `diss`, `disr.1`, `diss.1`): The real distribution
   is a perfect point mass at 0. The Gaussian value head adds a tiny amount of noise
   even when the gate correctly outputs 0, which is enough for the KS test to detect.
   These features carry no information and could be dropped from the feature set.

2. **Bimodal class structure**: The KS test compares the full synthetic population
   (class 0 + class 1 concatenated) against the full real population. Because class 0
   and class 1 have very different marginal distributions for routing metrics, the
   combined distribution is bimodal. If the class balance in synthesis (1000 per class)
   differs from the real balance (265 class-0 / 165 class-1), the KS test will detect
   the shape mismatch even if each per-class distribution is perfectly reproduced.

---

## 6. Files

| File | Description |
|---|---|
| `toy/lstmvae_toy.py` | TimeVAE vs RVAE comparison with log1p pre-processing (Option 2) |
| `toy/zivae_toy.py` | ZI-VAE vs TimeVAE comparison (Option 3) |
| `toy/zivae_toy_results/` | Plots: distributions, KS bars, correlations, TSTR/TRTS |
| `toy/lstmvae_toy_results/` | Plots from TimeVAE vs RVAE comparison |
| `src/time_vae.py` | Base TimeVAE implementation |
| `src/rvae.py` | RVAE (GRU-VAE) implementation |

---

## 7. Recommendation

For the federated IDS paper:

- **Use ZI-VAE** if distribution fidelity (KS test) is a reported metric.
  It provides the best marginal distribution match while preserving class boundaries.
- **Use plain TimeVAE or RVAE** if the goal is purely replay quality (TSTR/TRTS),
  since all three achieve near-perfect F1 with lower architectural complexity.
- **Consider dropping all-zero features** (`disr`, `diss`, `disr.1`, `diss.1`)
  from the feature set — they carry zero information and can only hurt the KS score.
