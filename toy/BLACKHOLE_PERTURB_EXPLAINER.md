# blackhole_perturb.py — What It Does and Why

## The Problem

We want to generate **synthetic Blackhole attack traffic** that is realistic enough to:
1. Train a classifier on synthetic data and have it generalise to real data (**TSTR**)
2. Have a classifier trained on real data correctly identify synthetic attacks (**TRTS**)

Standard generative models (VAE, GAN) failed because Blackhole is an extremely subtle attack.

---

## Feature Statistics (blackhole_var10_base, raw values)

14 features total. `disr`, `diss`, `disr.1`, `diss.1` are completely dead — all-zero in both classes.

| Feature | Zero% (all) | Zero% normal | Zero% attack | Normal mean ± std | Attack mean ± std | KS stat | RF importance |
|---------|-------------|--------------|--------------|-------------------|-------------------|---------|---------------|
| `rank`    | 0%  | 0%  | 0%  | 464 ± 46   | 409 ± 55   | **0.423** | **35.0%** |
| `rank.1`  | 0%  | 0%  | 0%  | 189 ± 48   | 147 ± 44   | **0.360** | **36.1%** |
| `disr`    | 100% | 100% | 100% | 0 ± 0    | 0 ± 0      | 0.000 | 0.0% |
| `diss`    | 100% | 100% | 100% | 0 ± 0    | 0 ± 0      | 0.000 | 0.0% |
| `dior`    | 30% | 29%  | 30%  | 0.28 ± 0.36 | 0.26 ± 0.27 | 0.051 | 5.4% |
| `dios`    | 33% | 33%  | 34%  | 0.19 ± 0.23 | 0.19 ± 0.20 | 0.074 | 3.1% |
| `diar`    | 85% | 88%  | 83%  | 0.05 ± 0.18 | 0.07 ± 0.19 | 0.047 | 1.6% |
| `tots`    | 28% | 29%  | 28%  | 0.22 ± 0.25 | 0.23 ± 0.22 | 0.083 | 3.7% |
| `disr.1`  | 100% | 100% | 100% | 0 ± 0   | 0 ± 0      | 0.000 | 0.0% |
| `diss.1`  | 100% | 100% | 100% | 0 ± 0   | 0 ± 0      | 0.000 | 0.0% |
| `dior.1`  | 30% | 29%  | 30%  | 0.39 ± 0.32 | 0.38 ± 0.30 | 0.051 | 6.2% |
| `dios.1`  | 33% | 33%  | 34%  | 0.30 ± 0.24 | 0.31 ± 0.24 | 0.075 | 3.4% |
| `diar.1`  | 85% | 88%  | 83%  | 0.09 ± 0.27 | 0.11 ± 0.27 | 0.047 | 1.8% |
| `tots.1`  | 28% | 29%  | 28%  | 0.34 ± 0.24 | 0.35 ± 0.25 | 0.082 | 3.7% |

**Key observations:**
- `rank` + `rank.1` together hold **71.1%** of discriminative power (RF importance)
- `rank` lag-1 autocorrelation: normal φ = 0.959, attack φ = 0.981 — highly persistent sequences
- `rank.1` lag-1 autocorrelation: normal φ = 0.984, attack φ = 0.986
- All other features have KS < 0.09 — the normal and attack distributions are nearly identical
- `disr`, `diss`, `disr.1`, `diss.1` are structurally all-zero across both classes (dead features)
- `diar` / `diar.1` are highly sparse (85% zero) with negligible class difference

---

## Why Blackhole Is Hard to Synthesise

**94% of Blackhole rows are statistically indistinguishable from normal traffic.**
The only consistent signal is a **persistent downward shift in routing rank** (`rank` and `rank.1`).

The attack mechanism explains this directly: a Blackhole node advertises a falsely low routing rank to attract traffic from neighbours, then silently drops it. The result is that the node's observed rank is systematically lower than a legitimate node's rank — but everything else about the traffic looks normal.

A generative model that does not explicitly encode this mechanism will either:
- Average out the rank shift (VAE posterior collapse) and produce samples that look normal, or
- Memorise the shift plus all the noise features and generate detectable-but-unrealistic attacks

---

## Why a Standard VAE Failed

A two-stage VAE was tried first (`blackhole_vae.py`). It failed because:
- The residual VAE trained on attack deltas over-amplified the dead features and generated attacks that were "too obvious" (cond_precision = 1.0, TSTR = 0.33)
- The model could not isolate the subtle rank-level shift from all the near-identical features

---

## Why the Preprocessing Destroyed the Signal (Critical Bug)

The original preprocessing pipeline (from `zirvae_multifile.py`) applied **differencing** to rank columns:

```
rank_preprocessed = sign(Δrank) × log1p(|Δrank|)
```

This converts an **absolute level** into a **rate of change**. For Blackhole:

| Stage | Normal rank | Attack rank | Difference |
|-------|-------------|-------------|------------|
| Raw values | 464 | 409 | **55 units** |
| After diff + log1p + min-max | 0.822 | 0.827 | **0.005** |

The 55-unit level difference is the entire attack signal. Differencing converts it into a near-zero rate-of-change that is identical in both classes.

**Fix:** `blackhole_perturb.py` does NOT difference rank columns. Rank goes directly from raw integer → min-max normalised [0, 1]:

| Stage | Normal rank | Attack rank | Difference |
|-------|-------------|-------------|------------|
| Raw values | 464 | 409 | 55 units |
| Min-max normalised (no diff) | 0.582 | 0.395 | **0.187** ✓ |

After this fix, Rank-only F1 jumps from 0.38 → **0.66** — close to the full-feature baseline.

---

## The Synthesis Strategy

### Why Two-Component Generation?

The data has two structurally different parts:
1. **Background features** (`dior`, `dios`, `tots`, `diar`, + `.1` variants): continuous, zero-inflated, similar in both classes. A VAE can model these well.
2. **Signal features** (`rank`, `rank.1`): strongly autocorrelated time series whose class difference is purely a **level shift**, not a shape change. A VAE will average this out. An AR(1) model directly encodes the level and autocorrelation.

This motivates a hybrid: **ZICVAE for background + AR(1) perturbation for signal**.

---

## Component 1 — ZICVAE (Zero-Inflated Conditional GRU-VAE)

### Why ZICVAE instead of a plain VAE?

Three design choices are each motivated by the data:

**Conditional (C):** The CVAE is conditioned on the class label y ∈ {0, 1}. This lets a single model learn both P(features | normal) and P(features | attack) simultaneously. When sampling attack backgrounds (y=1), the non-rank features (dior, dios, tots, ...) are drawn from the true attack conditional distribution rather than the normal-class distribution. This matters because even though KS < 0.09, the joint distribution can still differ subtly.

**Zero-Inflated (ZI):** Features like `dior`, `dios`, `diar`, `tots` have 28–85% zeros. A Gaussian VAE decoder will try to model these with a continuous distribution centred away from zero, producing small non-zero values instead of true structural zeros. The ZI gate treats each timestep as a mixture: with probability p_zero the output is exactly 0 (structural absence), otherwise the reconstructed continuous value is used. This is essential to preserve the sparsity pattern that distinguishes normal from attack traffic in these features.

**GRU encoder/decoder:** The data is a time series of window length T=10. A GRU captures temporal dependencies within each window (lag-1 autocorrelations of 0.46–0.55 in the background features). A plain MLP VAE would treat each timestep independently and miss these dependencies.

### Architecture

```
Input
  x  : (B, T, F_active)   windowed time series, active features only (F=10)
  y  : (B,)  long         class label  {0 = normal, 1 = attack}
```

#### Label Embedding (shared)
```
y  ──→  Embedding(num_classes=2, label_dim=4)  ──→  e  (B, 4)
```
The same embedding table is used by both encoder and decoder, so the latent space
is conditioned on class identity end-to-end.

#### Encoder
```
e_exp = e.unsqueeze(1).expand(-1, T, -1)        (B, T, 4)  ← broadcast over time

[x ; e_exp]  (B, T, F+4)
      │
     GRU(input=F+4, hidden=H, n_layers=1, batch_first=True)
      │
  last hidden h[-1]  (B, H)
      │
  ┌───┴────┐
fc_mu     fc_lv              Linear(H, Z) each
  │         │                fc_lv bias initialised to -1.0 (avoid collapse)
  μ (B,Z)  log σ² (B,Z)
      │
  reparameterise:  z = μ + ε · exp(½ log σ²),   ε ~ N(0, I)
      │
      z  (B, Z)
```

#### Decoder
```
e  (B, 4)
[z ; e]  (B, Z+4)  =  zy
    │
    ├──→  fc_h0   Linear(Z+4, n_layers·H)
    │             reshape  →  h0  (n_layers, B, H)   ← initial GRU hidden state
    │
    └──→  fc_input  Linear(Z+4, F)
                   unsqueeze + expand  →  inp  (B, T, F)  ← same input at every step

inp, h0  ──→  GRU(input=F, hidden=H, n_layers=1, batch_first=True)
                        │
                   gru_out  (B, T, H)
                        │
              ┌─────────┴──────────────┐
           fc_out                  fc_gate
         Linear(H, F)           Linear(H, |ZI|)
         + sigmoid               (logits, one per ZI feature)
              │                       │
         x̂ ∈ (0,1)  (B,T,F)     gate_logits (B,T,|ZI|)
```

#### ZI Masking (sampling only)
```
gate_prob = sigmoid(gate_logits)              (B, T, |ZI|)
gate_bin  = Bernoulli(gate_prob)              1 = emit value, 0 = force zero

for each ZI feature f:
    result[:,:,f] = gate_bin[:,:,f] ? x̂[:,:,f] : 0
```

### Feature Buckets

Features are automatically classified at runtime:

| Bucket | Criterion | Features (Blackhole) | Loss term |
|--------|-----------|----------------------|-----------|
| `rank_idx` | column name in `RANK_COLS` | `rank`, `rank.1` | MSE |
| `zi_idx` | not rank AND zero% > 30% in normal | `diar`, `diar.1` (and others depending on variant) | gate BCE + masked MSE |
| `lognorm_idx` | everything else active | `dior`, `dios`, `tots`, `dior.1`, `dios.1`, `tots.1` | MSE |

`disr`, `diss`, `disr.1`, `diss.1` are excluded entirely (always zero — kept as structural zeros).

### Loss Function

```
L_recon  =  (T·F / n_terms) ·
             [ Σ_{f ∈ rank ∪ lognorm}  MSE(x̂_f, x_f)
             + Σ_{f ∈ ZI}  [ 0.5 · BCE(gate_f, 1{x_f > 0})
                            + 0.5 · MSE(x̂_f, x_f | x_f > 0) ] ]

L_KL     =  Σ_z  max( −½(1 + log σ²_z − μ²_z − σ²_z),  free_bits )

L_total  =  L_recon  +  β(epoch) · L_KL
```

**Free-bits KL:** Each latent dimension is allowed up to `free_bits` nats of KL for free. This prevents posterior collapse on low-information dimensions without switching off the encoder entirely.

**Cyclical β annealing:** β(epoch) follows a sawtooth schedule over 4 cycles within the total epoch budget:

```
cycle_len  = n_epochs / 4
phase      = (epoch mod cycle_len) / cycle_len
β(epoch)   = min(1.0,  2 · phase)
```

β ramps 0→1 within each cycle, then resets. This gives the decoder repeated opportunities to learn a good reconstruction before the KL penalty tightens.

**Denoising:** Input to the encoder is x + N(0, 0.05²), clipped to [0,1]. Exact zeros on ZI features are restored before encoding so the gate learns the true zero-inflation pattern.

### Hyperparameters (auto-scaled per variant)

| Parameter | Formula | Typical value |
|-----------|---------|---------------|
| Latent dim Z | `max(8, min((T·F_active)//8, 32))` | 12 |
| Hidden dim H | `max(64, F_active × 6)` | 60–96 |
| Batch size | `min(512, max(64, N_train//10))` | 256–512 |
| Free-bits | `min(0.5, 8/Z)` | 0.5 |
| Epochs | 300 (fixed) | 300 |
| Noise std | 0.05 (fixed) | 0.05 |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-5 | — |
| Gradient clip | 5.0 | — |

---

## Component 2 — AR(1) Perturbation for Rank

### Why AR(1) instead of letting the VAE model rank?

The rank signal is a **level shift**: attack rank sequences have the same shape and autocorrelation as normal rank sequences, but their mean is ~55 units lower. A VAE encodes both classes into a shared latent space and will tend to generate samples near the posterior mean, which blurs the level difference. An AR(1) model directly parameterises the level (μ), autocorrelation (φ), and noise (σ_inn), making the level shift exact by construction.

Additionally, rank sequences have φ ≈ 0.96–0.98 — extremely high autocorrelation within a 10-step window. An AR(1) model captures this with one parameter; a GRU decoder would need many epochs to learn it implicitly.

### Model

An AR(1) process with mean reversion:

```
x_t = φ · (x_{t-1} − μ) + μ + ε_t
ε_t ~ N(0, σ_inn²)
```

The marginal distribution is:

```
x_t ~ N(μ,  σ²)
σ_inn = σ · √(1 − φ²)
```

Parameters are fitted **on raw (unpreprocessed) rank values** from training attack windows, separately per class:

```
μ     = sample mean of attack rank sequences
σ     = sample std  of attack rank sequences
φ     = mean lag-1 autocorrelation across all attack windows in training set
σ_inn = σ · √(1 − φ²)
```

Fitted values (var10_base):

| Parameter | Normal | Attack |
|-----------|--------|--------|
| μ (rank) | 464 | 409 |
| σ (rank) | 46 | 55 |
| φ (rank) | 0.959 | 0.981 |
| σ_inn (rank) | 12.6 | 10.8 |
| μ (rank.1) | 189 | 147 |
| σ (rank.1) | 48 | 44 |
| φ (rank.1) | 0.984 | 0.986 |
| σ_inn (rank.1) | 8.4 | 7.5 |

### Sampling

Each synthetic sequence is initialised from the marginal:

```
x_0 ~ N(μ, σ²)

for t = 1, …, T−1:
    x_t = φ · (x_{t-1} − μ) + μ + ε_t,    ε_t ~ N(0, σ_inn²)
```

After sampling in raw rank space, the sequence is converted to the preprocessed [0,1] space via min-max normalisation (no log1p, no differencing — rank is not log1p transformed):

```
x_preprocessed = clip((x_raw − g_min) / (g_max − g_min),  0, 1)
```

where `g_min`, `g_max` are fitted on the training set.

### Why fit in raw space, not preprocessed space?

The class-level difference (normal μ=464, attack μ=409) is large and interpretable in raw space. After min-max normalisation the difference compresses to ~0.19, and the normalisation constants themselves depend on the data split. Fitting in raw space and then applying the same preprocessing chain used on real data ensures the synthetic rank sequences end up in exactly the right position within the normalised feature space.

---

## Synthesis Pipeline

```
Training data
      │
      ├── preprocess (log1p sparse + min-max, NO differencing on rank)
      │         └── windowing  →  (N, T=10, F=14)
      │
      ├── ZICVAE training  (active features F=10, both classes, 300 epochs)
      │
      └── AR(1) fitting on raw rank  (attack class, per RANK_COL)

Synthesis
      │
      ├── Normal class:
      │     vae.sample(N_SYNTH, y=0)            (N, T, F_active)
      │     embed into full F=14 space
      │     dead features stay 0
      │
      └── Attack class:
            vae.sample(N_SYNTH, y=1)            ← CVAE attack background
            embed into full F=14 space
            for col in [rank, rank.1]:
                sample AR(1) in raw space
                convert to preprocessed space
                overwrite col in attack windows
            clip to [0, 1]
```

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **Baseline F1** | LSTM trained and tested on real data (upper bound) |
| **Rank-only F1** | LSTM trained only on `rank`/`rank.1` — confirms rank signal survived preprocessing |
| **TSTR F1** | Train on Synthetic, Test on Real — main quality metric |
| **TRTS F1** | Train on Real, Test on Synthetic — checks synthetic attacks are classifiable |
| **Cond. Precision** | Fraction of synthetic attacks labelled as attack by the real-data classifier |
| **KS cls0/cls1** | KS statistic per feature: how well synthetic matches real distribution per class |
| **Tail KS / Coverage** | KS and range coverage restricted to windows where rank < 10th percentile of normal (most extreme attack region) |

**Good synthesis:** TSTR ≈ Baseline, TRTS > 0.5, Cond. Precision > 0.5, KS mean < 0.10.

---

## Results (var10_base, AR1-CVAE)

```
Baseline F1    = 0.7409
Rank-only F1   = 0.6772   ← was 0.38 before fix, now close to baseline
TSTR F1        = 0.7421   ← matches baseline ✓
TRTS F1        = 0.8084   ← synthetic attacks are classifiable ✓
Cond Precision = 0.8150   ← 81% of synthetic attacks recognised by real clf ✓
KS-Normal mean = 0.0941
KS-Attack mean = 0.1042
```

---

## Method Comparison: AR(1)+ZICVAE vs ZIRVAE

Two generators are implemented in `blackhole_perturb.py` and can be run against the same variants:

| | AR(1)+ZICVAE | ZIRVAE |
|---|---|---|
| **Rank preprocessing** | Level preserved (min-max only, no diff) | Differenced (sign-log1p) |
| **Generator structure** | One conditional model (both classes) + AR(1) for rank | Two independent per-class models |
| **Rank synthesis** | AR(1) sampled in raw space, converted to preprocessed | VAE must learn from temporal change pattern |
| **Feature-type detection** | 3-way: rank / ZI / lognorm | 4-way: bernoulli / zi\_lognorm / lognormal / continuous |
| **Dead feature handling** | Excluded from VAE input, kept as structural zeros | Bernoulli head natively produces all-zero output |
| **Run command** | `python blackhole_perturb.py` | `python blackhole_perturb.py --zirvae` |

### Side-by-Side Results (all 12 variants)

**Data sources:** ZIRVAE values from `blackhole_perturb_results/`; AR1-CVAE values from `blackhole_perturb_results_new approach/`.

**Key:** TSTR/TRTS/Cond.Prec — higher is better ↑. KS-Normal/KS-Attack — lower is better ↓. Bold = winner per cell.

| Variant | Baseline | TSTR ↑ | | TRTS ↑ | | Cond.Prec ↑ | | KS-Normal ↓ | | KS-Attack ↓ | |
|---------|----------|--------|--------|--------|--------|-------------|-------------|-------------|-------------|-------------|-------------|
| | | ZIRVAE | AR1-CVAE | ZIRVAE | AR1-CVAE | ZIRVAE | AR1-CVAE | ZIRVAE | AR1-CVAE | ZIRVAE | AR1-CVAE |
| var5\_base  | 0.702 | **0.591** | 0.326 | **0.600** | 0.228 | **0.835** | 0.000 | 0.147 | **0.120** | 0.179 | **0.087** |
| var5\_dec   | 0.723 | **0.609** | 0.323 | **0.608** | 0.311 | **0.752** | 0.000 | 0.148 | **0.096** | 0.168 | **0.065** |
| var5\_oo    | 0.671 | **0.525** | 0.403 | **0.487** | 0.327 | **0.042** | 0.000 | 0.144 | **0.114** | 0.148 | **0.095** |
| var10\_base | 0.741 | 0.570 | **0.742** | 0.673 | **0.808** | 0.421 | **0.815** | 0.164 | **0.094** | 0.161 | **0.104** |
| var10\_dec  | 0.609 | **0.541** | 0.482 | **0.663** | 0.657 | **0.751** | 0.561 | 0.183 | **0.115** | 0.163 | **0.117** |
| var10\_oo   | 0.662 | **0.652** | 0.648 | **0.759** | 0.631 | 0.376 | **0.540** | 0.158 | **0.143** | **0.134** | 0.140 |
| var15\_base | 0.575 | 0.608 | **0.633** | 0.646 | **0.768** | **0.722** | 0.530 | 0.191 | **0.114** | 0.169 | **0.149** |
| var15\_dec  | 0.674 | 0.519 | **0.666** | 0.646 | **0.736** | **0.876** | 0.656 | 0.202 | **0.153** | 0.247 | **0.172** |
| var15\_oo   | 0.540 | 0.479 | **0.502** | **0.426** | 0.335 | **0.030** | 0.002 | 0.249 | **0.207** | **0.244** | 0.262 |
| var20\_base | 0.619 | 0.523 | **0.615** | 0.638 | **0.919** | **0.930** | 0.835 | 0.221 | **0.138** | 0.217 | **0.142** |
| var20\_dec  | 0.713 | 0.499 | **0.570** | 0.626 | **0.580** | **0.895** | 0.670 | 0.208 | **0.147** | 0.228 | **0.168** |
| var20\_oo   | 0.736 | 0.304 | **0.612** | 0.372 | **0.459** | 0.211 | **0.167** | 0.242 | **0.175** | 0.152 | **0.182** |
| **Mean**    | **0.647** | **0.535** | **0.543** | **0.596** | **0.563** | **0.570** | **0.398** | **0.188** | **0.135** | **0.184** | **0.140** |
| **Wins**    | — | 5 ZI | **7 AR1** | **7 ZI** | 5 AR1 | **10 ZI** | 2 AR1 | 0 ZI | **12 AR1** | 3 ZI | **9 AR1** |

### Win-count summary

| Metric | ZIRVAE wins | AR1-CVAE wins |
|--------|:-----------:|:-------------:|
| TSTR ↑ | 5 | **7** |
| TRTS ↑ | **7** | 5 |
| Cond. Precision ↑ | **10** | 2 |
| KS-Normal ↓ | 0 | **12** |
| KS-Attack ↓ | 3 | **9** |

### Aggregate (mean ± std across 12 variants)

| Metric | ZIRVAE | AR1-CVAE | Δ (AR1 − ZIRVAE) |
|--------|--------|----------|-----------------|
| TSTR ↑ | 0.535 ± 0.089 | **0.543 ± 0.137** | +0.008 |
| TRTS ↑ | **0.596 ± 0.111** | 0.563 ± 0.227 | −0.032 |
| Cond. Precision ↑ | **0.570 ± 0.337** | 0.398 ± 0.338 | −0.172 |
| KS-Normal ↓ | 0.188 ± 0.037 | **0.135 ± 0.033** | −0.053 (AR1 better) |
| KS-Attack ↓ | 0.184 ± 0.039 | **0.140 ± 0.053** | −0.044 (AR1 better) |

> Data sources: ZIRVAE from `blackhole_perturb_results/*/zirvae_metrics.json`; AR1-CVAE from `blackhole_perturb_results_new approach/*/metrics.json`.

### Key takeaways

**AR(1)+ZICVAE wins on distribution fidelity (KS) and TSTR.**
By preserving the rank level-shift and encoding it directly via AR(1), synthetic windows land in the correct region of feature space. The VAE background handles non-rank features. This produces tighter KS statistics (0.135 vs 0.188 on normal class) and higher TSTR on variants with enough attack data (var10+, var15+, var20+).

**ZIRVAE wins on Cond. Precision and TRTS.**
Its data-driven approach is more robust to data scarcity. On all three `var5_*` variants, AR(1)+ZICVAE completely collapses: Cond. Precision = 0.000 on all three, meaning the real classifier sees none of the synthetic attacks as attacks at all. ZIRVAE achieves 0.835, 0.752, 0.042 on the same variants. With few attack windows the AR(1) parameters are poorly estimated and the rank perturbation misses the target region entirely.

**The var5 failure is the critical distinction.**
With a small attack training set, the AR(1) mean (μ_attack) is estimated from too few windows and lands too close to μ_normal. The synthetic rank then falls in the normal range, producing synthetic "attacks" that are indistinguishable from normal traffic. ZIRVAE avoids this because it learns directly from the data distribution without relying on a point estimate of the rank mean.

**Recommendation by use case:**

| Situation | Recommended method |
|-----------|-------------------|
| Large attack training set (var10+) | AR(1)+ZICVAE — better KS and TSTR |
| Small attack training set (var5) | ZIRVAE — AR(1) collapses here |
| Need tight distribution match | AR(1)+ZICVAE |
| Need recognisable synthetic attacks | ZIRVAE |

---

## File Structure

```
toy/
├── blackhole_perturb.py          # Main script
├── blackhole_perturb_results/
│   ├── blackhole_var*/
│   │   ├── metrics.json                   # All metrics for this variant
│   │   ├── rank_distributions_*.png       # Rank histograms: real vs synthetic
│   │   ├── distributions_*.png            # All features: real vs synthetic
│   │   ├── ks_*.png                       # KS bars per feature
│   │   └── ar1_validation_*.png           # Lag-1 scatter: real vs synthetic attack
│   ├── summary.csv               # One row per variant
│   ├── summary.json
│   └── tstr_trts_all_variants.png
```

---

## Key Functions

| Function | Purpose |
|----------|---------|
| `split_files()` | 70/30 train/test file split (fixed seed) |
| `preprocess_and_window()` | log1p on sparse cols, min-max normalise, NO differencing on rank, sliding windows |
| `fit_rank_ar1_raw()` | Fit AR(1) on **raw** rank values per class |
| `sample_ar1()` | Generate AR(1) sequences: `x_t = φ(x_{t-1} − μ) + μ + ε` |
| `raw_rank_seq_to_preprocessed()` | Convert raw rank → normalised [0,1] via min-max (no log1p) |
| `generate_attack_windows()` | Build synthetic attacks: CVAE attack background + AR(1) rank |
| `train_zicvae()` | ZI Conditional GRU-VAE on all windows, both classes |
| `signal_only_f1()` (→ `rank_only_f1`) | Diagnostic: how much signal is in rank alone |
| `tail_ks_and_coverage()` | Evaluation on extreme low-rank windows only |
