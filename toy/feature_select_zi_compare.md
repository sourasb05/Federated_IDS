# Feature Selection + Zero-Inflated Generative Model Comparison

**Script:** `toy/feature_select_zi_compare.py`  
**Output directory:** `toy/feature_select_zi_results/`

---

## Overview

The pipeline has three stages:

1. **Feature Selection** — aggregate importance scores across 16 files and 4 methods to identify the most discriminative features.
2. **Generative Model Training** — train three Zero-Inflated VAE architectures on the selected features only.
3. **Evaluation** — compare synthetic data quality using KS tests, TSTR, and TRTS.

The motivation for combining feature selection with zero-inflated generation is twofold: (a) irrelevant or near-constant features add noise to the VAE reconstruction objective without contributing to the class-discriminative signal, and (b) the ZI decoder is most effective when it is focused on the sparse features that actually carry attack signatures.

---

## Data

| Item | Value |
|---|---|
| Dataset | `attack_data/blackhole_var20_base/` |
| Total files | 20 CSV files (`1_features_timeseries_60_sec.csv` … `20_features_timeseries_60_sec.csv`) |
| Raw features | 14 continuous features + `label` (0 = Normal, 1 = Attack) |
| Feature names | `rank`, `disr`, `diss`, `dior`, `dios`, `diar`, `tots`, `rank.1`, `disr.1`, `diss.1`, `dior.1`, `dios.1`, `diar.1`, `tots.1` |
| File split (seed=42) | 16 files for feature selection; 14 for training; 6 for testing |
| Window size | 10 timesteps |
| Window labelling | A window is labelled 1 (Attack) if any of its 10 rows has `label=1` |

---

## Stage 1 — Feature Selection

### Preprocessing (applied before scoring)

Each file is preprocessed independently, fitting all statistics on that file's data only, to avoid leakage between files during the aggregation step.

**Step 1 — Differencing + sign-log1p for rank features**

`rank` and `rank.1` are near-monotonic global counters. Their absolute value encodes time position, not network behaviour. First-differencing removes the trend and leaves the per-step change rate, which is stationary:

```
Δrank[t] = rank[t] − rank[t−1]
rank_transformed[t] = sign(Δrank[t]) × log(1 + |Δrank[t]|)
```

Sign-log1p compresses heavy tails while preserving the sign and the exact zero at no change.

**Step 2 — Sparse feature detection**

A feature is considered sparse if more than 30% of its values are exactly zero after step 1. This threshold (`SPARSE_THRESH = 0.30`) reflects the structure of RPL network metrics, where many traffic counters are zero during normal operation and become non-zero during attacks.

**Step 3 — log1p on sparse features**

Non-negative sparse features (network counters) have a highly right-skewed distribution. `log1p` compresses the tail and makes the non-zero mass more learnable.

**Step 4 — Min-max normalisation**

Fitted on the current file's data and clipped to `[0, 1]`.

**Step 5 — Symmetric rescaling for rank columns**

After sign-log1p, rank columns span `[−A, +A]` with a spike at 0. Standard min-max maps 0 to an interior value (approximately 0.56 for typical data), hiding the spike. The fix scales by the absolute maximum so that 0 maps exactly to 0.5:

```
rank_scaled = rank_transformed / (2 × max(|rank_transformed|)) + 0.5
```

Negative changes → `[0, 0.5)`, no change → `0.5`, positive changes → `(0.5, 1]`.

### Four Scoring Methods

After preprocessing, each file is scored with four methods. All scores are normalised to `[0, 1]` within each method using min-max scaling so they are on a comparable scale before averaging.

#### Filter: ANOVA F-statistic

The F-statistic measures the ratio of between-class variance to within-class variance for each feature independently:

```
F_i = (between-class variance of feature i) / (within-class variance of feature i)
```

High F → the class means are far apart relative to within-class spread → the feature separates Normal from Attack in a linear sense. This is the standard univariate filter for binary classification.

**Limitation:** assumes normally distributed features and linear separability. Used here as a fast, interpretable baseline.

#### Filter: Mutual Information

Mutual information measures the general statistical dependence between a feature and the label, without any linearity assumption:

```
MI(X_i; Y) = Σ_{x,y} P(X_i=x, Y=y) × log[ P(X_i=x, Y=y) / (P(X_i=x) × P(Y=y)) ]
```

Estimated via k-nearest-neighbours (sklearn implementation, `random_state=0` for reproducibility). Captures non-linear and non-monotonic relationships that ANOVA misses.

**Limitation:** sensitive to the number of samples; can be noisy on small files. Averaging across 16 files mitigates this.

#### Filter: Spearman |ρ|

Spearman rank correlation measures monotonic association between a feature and the label:

```
ρ_i = 1 − (6 × Σ d_j²) / (n(n²−1))
```

where `d_j` is the difference in ranks of feature value and label for observation `j`. The absolute value is taken because the direction of correlation is irrelevant — both strong positive and strong negative correlation indicate a useful feature.

**Advantage over Pearson:** robust to outliers and captures monotonic non-linear relationships (e.g., a feature that increases with attacks but not linearly).

#### Embedded: Random Forest Gini Importance

A Random Forest (100 trees, `max_depth=6`, `class_weight='balanced'`, `random_state=0`) is trained on the row-level (non-windowed) feature matrix. The Gini importance of feature `i` is the total reduction in node impurity weighted by the fraction of samples reaching each node, averaged across all trees:

```
Importance_i = (1/n_trees) × Σ_trees Σ_nodes [p(node) × ΔGini(node, i)]
```

This is an **embedded** method — importance is derived from the fitted model, so it captures interaction effects between features that filter methods miss. The `class_weight='balanced'` setting corrects for class imbalance.

**Limitation:** can be biased towards high-cardinality continuous features. Using it alongside filter methods prevents over-reliance on it.

### Score Aggregation

For each file `f` and method `m`, a normalised score vector `s_{f,m} ∈ [0,1]^F` is computed.

The final importance score for feature `i` is:

```
score_i = (1 / (16 × 4)) × Σ_f Σ_m  s_{f,m}[i]
```

Averaging across files reduces noise from individual files with constant or near-constant features (which produce degenerate ANOVA values and are handled by `nan_to_num`). Averaging across methods ensures that no single method dominates the selection — a feature must score consistently across multiple perspectives to be selected.

The top-`K` features by `mean_score` are retained (`TOP_K_FEATURES = 8` by default).

### Outputs

| File | Content |
|---|---|
| `feature_importance.csv` | Table: feature, anova_f, mutual_info, spearman_rho, rf_gini, mean_score, rank |
| `feature_importance.png` | 5-panel horizontal bar chart — one panel per method + mean score, sorted by mean score, with top-K cut-off line |

---

## Stage 2 — Zero-Inflated Generative Models

### Why Zero-Inflation

Network traffic metrics have a bimodal distribution: a point mass at exactly zero (feature is inactive) and a continuous positive distribution (feature is active). A standard Gaussian decoder cannot represent this structure — it maps the latent code to a single continuous value per feature, producing a smeared near-zero mass instead of exact zeros.

The Zero-Inflated (ZI) decoder uses two heads per sparse feature:

1. **Gaussian head** — produces the value when the feature is active (sigmoid output in `[0, 1]`)
2. **Bernoulli gate head** — produces the probability that the feature is non-zero

At synthesis time: `output = gate × value`, where `gate ~ Bernoulli(gate_prob)`.

Features with ≤ 30% zeros (dense features) use only the Gaussian head.

### Shared ZIDecoder

All three architectures use the same decoder. The encoder is the only part that differs.

```
z  (B, latent_dim)
 └─ fc_h0  → GRU initial hidden state h0  (n_layers, B, hidden_dim)
 └─ fc_input → repeated input  (B, T, n_features)
 └─ GRU(inp, h0)  → hidden states  (B, T, hidden_dim)
 └─ fc_out → sigmoid → Gaussian output  (B, T, n_features)          [all features]
 └─ gate_fc → Bernoulli logits  (B, T, |sparse_idx|)                [sparse features only]
```

The GRU decoder generates the window **autoregressively** in hidden-state space (not in output space) — each timestep's hidden state depends on all previous timesteps. This preserves temporal autocorrelation in the synthetic window, which an MLP decoder cannot do.

### ZI-RVAE

**Encoder:** GRU encoder (`REncoder` from `src/rvae.py`)

```
x  (B, T, F)
 └─ GRU  →  last hidden state  (B, hidden_dim)
 └─ fc_mu, fc_var  →  (mu, log_var)  (B, latent_dim)
```

The GRU processes the window as a sequence, compressing all T timesteps into the final hidden state. This is the simplest recurrent encoder — appropriate as a baseline for the ZI decoder comparison.

**Decoder:** ZIDecoder (shared)

### ZI-LSTM-Attn

**Encoder:** LSTM with dot-product self-attention over all T timesteps

```
x  (B, T, F)
 └─ LSTM  →  all hidden states  (B, T, hidden_dim)
 └─ attn_w  →  scores  (B, T, 1)  →  softmax  →  weights  (B, T, 1)
 └─ weighted sum  →  context  (B, H)
 └─ fc_mu, fc_var  →  (mu, log_var)
```

Instead of using only the final hidden state (which may lose early timestep information in longer sequences), the attention mechanism creates a context vector as a weighted combination of all T hidden states. Each timestep is scored by a learned linear layer.

**Why it may help:** In windows where the attack signature appears at the beginning (e.g., a routing disruption affecting the first few timesteps), the GRU's final hidden state carries a diluted signal. Attention can re-weight those early timesteps more heavily.

**Decoder:** ZIDecoder (shared)

### ZI-Transformer

**Encoder:** Transformer encoder with sinusoidal positional encoding + mean pooling

```
x  (B, T, F)
 └─ Linear(F → d_model)  →  (B, T, d_model)
 └─ Positional encoding (sinusoidal, no learned params)
 └─ TransformerEncoder (n_layers=2, n_heads=4, dim_feedforward=hidden_dim)
    └─ Multi-head self-attention over T  →  (B, T, d_model)
 └─ Mean pooling over T  →  (B, d_model)
 └─ fc_mu, fc_var  →  (mu, log_var)
```

Multi-head self-attention attends to all pairs of timesteps simultaneously — not sequentially. This is fundamentally different from both GRU and LSTM: all T positions interact in parallel and the attention weights are query-dependent (different for each sample), not learned globally.

Sinusoidal positional encoding injects position information without adding learnable parameters, which is important given the relatively small dataset size.

**Constraint:** `d_model` must be divisible by `n_heads=4`. The script computes `d_model = max(32, (latent_dim // 4) * 4)` to satisfy this automatically.

**Decoder:** ZIDecoder (shared)

### Shared Training Details

All three models are trained with the same procedure on each class separately (one generator per class, class-conditional generation).

**ZI Loss Function**

```
L = loss_factor × MSE(gauss_out, x)
  + 0.1 × BCE(gate_logit, (x > 0).float())    [sparse features only]
  + kl_weight × KL_free_bits(mu, log_var)
```

- **MSE term** — reconstruction of the Gaussian head against the clean target. Scaled by `loss_factor = T × F` to make it commensurable with the KL term (which sums over `latent_dim` dimensions).
- **BCE gate term** — trains the Bernoulli gate directly on the observed zero/non-zero pattern. Weight `0.1` keeps it as a regulariser; the MSE term remains the primary signal.
- **Free-bits KL** — each latent dimension is clamped to a minimum of `free_bits` nats. Dimensions below the threshold produce zero KL gradient, preventing posterior collapse. `free_bits = min(0.5, 8.0 / latent_dim)` scales with latent dimension size.

**Anti-Posterior-Collapse Measures**

| Mechanism | Description |
|---|---|
| Free bits | Per-dim KL clamped to `free_bits` nats — encoder cannot collapse a dimension below this threshold |
| Cyclical KL annealing | `kl_weight` cycles 0 → 1 → 0 → 1 (4 cycles). Each cycle: linear ramp for first 50%, hold at 1.0 for last 50%. Periodic resets let the decoder re-learn reconstruction before KL pressure is applied again |
| Denoising | Gaussian noise (`noise_std=0.05`) added to encoder input; decoder reconstructs the clean target. Forces the latent code to carry sample-specific signal rather than the dataset mean |

**Hyperparameters (auto-scaled to selected feature count F)**

| Parameter | Formula | Rationale |
|---|---|---|
| `latent_dim` | `max(4, min((T×F)//10, 32))` | Proportional to total input dimensionality |
| `hidden_dim` | `max(64, F×8)` | Wide enough to capture per-step interactions |
| `free_bits` | `min(0.5, 8.0/latent_dim)` | Total free-bits budget ≈ 8 nats regardless of latent size |
| `batch_size` | `min(256, max(32, N//20))` | Scales with dataset size |
| `d_model` | `max(32, (latent_dim//4)×4)` | Divisible by 4 heads |
| `epochs` | 1000 | With cyclical annealing, 4 full cycles of 250 epochs each |

---

## Stage 3 — Evaluation

### KS Test per Class

For each feature, a two-sample Kolmogorov-Smirnov test is run separately for Normal (class 0) and Attack (class 1) samples:

```
H0: real and synthetic distributions are the same
Pass: p > 0.05  (no significant difference detected)
```

Before the test, windows are averaged over the time dimension `(N, T, F) → (N, F)` to produce per-sample feature vectors. Reported metrics:

- **mean_KS** — average KS statistic across all F features (lower is better)
- **KS pass** — number of features that pass the test (higher is better)

### TSTR — Train on Synthetic, Test on Real

A lightweight LSTM classifier (1 layer, 64 hidden units, 30 epochs) is trained on the synthetic data and evaluated on the real held-out test set. This measures whether the synthetic data's decision boundary matches the real boundary.

```
TSTR F1 ≈ Baseline F1  →  synthetic data is a faithful substitute for real data
TSTR F1 << Baseline F1 →  synthetic data distorts the class boundary
```

### TRTS — Train on Real, Test on Synthetic

The same LSTM classifier is trained on real training data and evaluated on the synthetic data. This measures whether the synthetic data is consistent with the classifier trained on real data — i.e., whether synthetic samples are recognisable as their intended class.

```
TRTS F1 ≈ Baseline F1  →  synthetic samples lie in the correct class region
TRTS F1 << Baseline F1 →  synthetic samples are misclassified — wrong distribution
```

### Baseline F1

The same LSTM trained and tested on real data. All TSTR and TRTS scores are interpreted relative to this value.

### Outputs

| File | Content |
|---|---|
| `feature_importance.csv` | Feature scores and ranks from Stage 1 |
| `feature_importance.png` | 5-panel importance bar chart |
| `ks_ZI*.png` | KS bar chart per architecture (blue = pass, red = fail) |
| `distributions_ZI*.png` | Real vs synthetic distribution overlay per feature and class |
| `correlations_ZI*.png` | Real correlation, synthetic correlation, and absolute difference heatmaps |
| `tstr_trts_comparison.png` | Grouped bar chart comparing Baseline / TSTR / TRTS across all three architectures |
| `zi_comparison_metrics.json` | All numeric results in structured JSON |

---

## Architecture Comparison Summary

| Aspect | ZI-RVAE | ZI-LSTM-Attn | ZI-Transformer |
|---|---|---|---|
| Encoder type | GRU | LSTM + attention | Multi-head self-attention |
| Temporal compression | Final hidden state | Attention-weighted sum of all states | Mean pool of all states |
| Decoder | ZIDecoder (GRU) | ZIDecoder (GRU) | ZIDecoder (GRU) |
| Zero-inflation | Yes | Yes | Yes |
| Parameter count | Lowest | Medium | Highest |
| Inductive bias | Local sequential | Local sequential + global weighting | Global pairwise interactions |

All three models are evaluated on the same selected feature subset with identical training hyperparameters, so differences in results are attributable to the encoder architecture alone.

---

## Key Design Decisions

**Why 16 files for feature selection, 14 for training?**  
Feature selection and model training use overlapping but distinct file sets. The 16-file selection set gives a more stable importance estimate (more data, more files to average over), while keeping the 14/6 train/test split identical to the baseline experiments for fair comparison.

**Why per-file scoring rather than pooled scoring?**  
Individual files have different class distributions and can have constant features (e.g., if an attack is absent in a particular capture). Scoring per file and averaging prevents any single file from dominating the importance estimate and naturally handles missing/constant features (they receive a score of 0 from that file).

**Why average across methods rather than taking the best method?**  
No single method is universally optimal for this data. ANOVA is fast and interpretable but linear; MI captures non-linear effects but is noisy; Spearman handles monotonic non-linearity robustly; RF captures interaction effects. A feature that ranks highly across all four methods is robustly important.

**Why the same ZIDecoder for all three architectures?**  
The decoder is what generates the data. Keeping it identical isolates the effect of the encoder on latent representation quality. Any difference in synthetic data quality between the three models is a direct reflection of how well each encoder compresses the input window into a useful latent code.

**Why `free_bits = min(0.5, 8.0 / latent_dim)` rather than a fixed value?**  
The total free-bits budget (free_bits × latent_dim) determines the minimum information content the encoder must preserve. Setting a budget of 8 nats regardless of latent size ensures that (a) small latent spaces are not over-constrained and (b) large latent spaces do not get so many free dimensions that KL regularisation becomes ineffective.
