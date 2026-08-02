# TimeKAN — Problem & Method Formulation

**TimeKAN:** Frequency-Decomposed Kolmogorov–Arnold Networks for Long-Term Time Series Forecasting.

> Note: this document formulates **TimeKAN** (KAN = Kolmogorov–Arnold Network), not k-NN.

---

## 1. Problem statement

Given a multivariate lookback window

$$
X = \{x_1, x_2, \ldots, x_L\} \in \mathbb{R}^{L \times C}
$$

with lookback length $L$ and $C$ channels, predict the next $H$ steps:

$$
\hat{Y} = \{\hat{x}_{L+1}, \ldots, \hat{x}_{L+H}\} \in \mathbb{R}^{H \times C}.
$$

**Hypothesis.** Different frequency components of a series follow different dynamics (trend ≈ low frequency, seasonality ≈ mid, noise/microstructure ≈ high). A single shared model that fits all bands jointly underfits some and overfits others. TimeKAN therefore:

$$
\text{decompose} \rightarrow \text{specialize} \rightarrow \text{fuse}.
$$

---

## 2. Architecture (4 stages)

```
X --> [A] RevIN --> [B] Frequency decomp --> [C] Per-band M-KAN --> [D] Attention fusion --> Linear --> RevIN^{-1} --> Yhat
                     |  X^(1)..X^(K)              |  Z^(1)..Z^(K)
                     +----------------------------+
```

| Stage | Module | Role |
|-------|--------|------|
| A | RevIN | Instance norm / denorm for distribution shift |
| B | Fixed FFT or Adaptive soft masks | Split $X$ into $K$ band signals |
| C | Chebyshev M-KAN (per band) | Band-specific nonlinear forecasting maps |
| D | Cross-band attention | Fuse $Z^{(k)}$ into $\hat{Y}$ |

---

## 3. Stage A — Reversible Instance Normalization (RevIN)

For each channel $c$, over the lookback axis:

$$
\mu_c = \frac{1}{L}\sum_{t=1}^{L} X_{t,c},
\qquad
\sigma_c = \sqrt{\frac{1}{L}\sum_{t=1}^{L}(X_{t,c}-\mu_c)^2 + \varepsilon}.
$$

Normalize (optional affine $\gamma_c$, $\beta_c$):

$$
\tilde{X}_{t,c} = \gamma_c \cdot \frac{X_{t,c}-\mu_c}{\sigma_c} + \beta_c.
$$

After the forecast head, **denormalize** with the same $(\mu_c, \sigma_c)$ (and invert affine). This is standard for LTSF under non-stationarity.

---

## 4. Stage B — Adaptive frequency decomposition

Work in the real DFT domain along time. Let

$$
\mathcal{F}(\tilde{X}) \in \mathbb{C}^{F \times C},
\qquad F = \left\lfloor \frac{L}{2} \right\rfloor + 1.
$$

### 4.1 Fixed bands

Partition frequency bins into $K$ contiguous bands with cutoffs at fractions of Nyquist (default: equal thirds). Hard mask $M^{(k)} \in \{0,1\}^F$:

$$
X^{(k)} = \mathcal{F}^{-1}\Big( M^{(k)} \odot \mathcal{F}(\tilde{X}) \Big) \in \mathbb{R}^{L \times C}.
$$

### 4.2 Adaptive / learnable bands (novelty knob)

A small MLP reads the pooled log-magnitude spectrum and emits soft masks over bands:

$$
s = \log\big(1 + |\mathcal{F}(\tilde{X})|\big)_{\mathrm{avg\ over\ }C} \in \mathbb{R}^{F},
$$

$$
\alpha = \mathrm{softmax}\big(\mathrm{MLP}(s)\big) \in \mathbb{R}^{K \times F}
$$

(softmax over the band axis $k$). Mix with a fixed prior $P$ for stability (implementation default $0.7\alpha + 0.3P$), then

$$
X^{(k)} = \mathcal{F}^{-1}\Big( \alpha_{k,:} \odot \mathcal{F}(\tilde{X}) \Big).
$$

**Output of Stage B:** $\{X^{(1)},\ldots,X^{(K)}\}$, each in $\mathbb{R}^{L \times C}$.

---

## 5. Stage C — Per-band Chebyshev M-KAN

### 5.1 Chebyshev edge function

Instead of slow B-spline KANs, use Chebyshev polynomials $T_i$ of the first kind, with $\tanh$ to keep inputs in $[-1,1]$:

$$
\phi(x) = \sum_{i=0}^{d} c_i \, T_i\big(\tanh(x)\big).
$$

Recurrence: $T_0 = 1$, $T_1 = u$, $T_n = 2u\,T_{n-1} - T_{n-2}$.

A Chebyshev-KAN linear layer maps $\mathbb{R}^{d_{\mathrm{in}}} \to \mathbb{R}^{d_{\mathrm{out}}}$ by applying learnable $\phi_{j \leftarrow i}$ on each input dimension and summing (efficient-kan style, GPU-friendly).

### 5.2 Temporal block per band

For band $k$, with degree $d_k$:

1. Channel lift: $H = \tilde{X}^{(k)} W_{\mathrm{in}} \in \mathbb{R}^{L \times d}$
2. Stack $\ell$ Chebyshev-KAN layers + residual LayerNorm
3. Time projection: $Z^{(k)} = \mathrm{LN}\big( (H^{\top} W_{\mathrm{time}})^{\top} \big) \in \mathbb{R}^{H \times d}$

### 5.3 Band-specific capacity (design point)

Low frequency → smoother / lower degree; high frequency → higher degree:

| $K$ | Degree schedule $(d_1,\ldots,d_K)$ |
|-----|--------------------------------------|
| 1 | $(4)$ |
| 2 | $(2, 6)$ |
| 3 | $(2, 4, 6)$ |
| 4 | $(2, 3, 5, 7)$ |
| 5 | $(2, 3, 4, 5, 7)$ |

Ablation: uniform $d_k \equiv d$ for all bands.

**PlainKAN ablation:** one undecomposed Chebyshev backbone on $\tilde{X}$ (no Stage B/D specialization) — isolates whether gains come from decomposition vs “KAN > MLP”.

---

## 6. Stage D — Cross-band attention fusion

Each band yields $Z^{(k)} \in \mathbb{R}^{H \times d}$. Let

$$
\bar{Z} = \frac{1}{K}\sum_{k=1}^{K} Z^{(k)}.
$$

At each forecast step $t = 1, \ldots, H$:

$$
\mathrm{score}_{t,k}
=
\frac{
\big(W_q Z^{(k)}_{t,:}\big)^{\top}
\big(W_k \bar{Z}_{t,:}\big)
}{\sqrt{d}},
\qquad
\alpha_{t,:} = \mathrm{softmax}(\mathrm{score}_{t,:}).
$$

$$
U_{t,:} = \sum_{k=1}^{K} \alpha_{t,k} \, (W_v Z^{(k)}_{t,:}),
\qquad
\hat{Y}^{\mathrm{norm}} = U W_{\mathrm{out}} \in \mathbb{R}^{H \times C}.
$$

Finally RevIN denorm:

$$
\hat{Y} = \mathrm{RevIN}^{-1}(\hat{Y}^{\mathrm{norm}}).
$$

**Fusion ablations:** replace attention by concat+linear, mean, or last-band-only.

---

## 7. Training objective

Minimize multivariate MSE on the forecast window:

$$
\mathcal{L}
=
\frac{1}{BHC}
\sum_{b=1}^{B}
\sum_{t=1}^{H}
\sum_{c=1}^{C}
\big(\hat{Y}_{b,t,c} - Y_{b,t,c}\big)^2.
$$

Optimizer: Adam; early stop on validation MSE; standard LTSF sliding windows.

---

## 8. Datasets & protocol

| Dataset | Role |
|---------|------|
| ETTh1 / ETTh2 / ETTm1 / ETTm2 | Standard LTSF benchmark |
| Weather | High-dimensional meteorological |
| Electricity | Many clients (channel-capped on 8GB GPU) |
| Finance (yfinance) | Log-returns (+ vol); not raw prices |

Horizons: $H \in \{96, 192, 336, 720\}$ (Finance demo uses shorter $H$).

Splits follow Informer / Autoformer / TSLib conventions.

---

## 9. Baselines (required for credibility)

| Family | Models |
|--------|--------|
| Naive / statistical | Last-value, ARIMA |
| Deep | LSTM, TCN |
| Linear | DLinear, NLinear |
| Transformer | Informer, Autoformer, FEDformer, PatchTST, iTransformer |
| Ablation control | **PlainKAN** (undecomposed) |

---

## 10. Metrics

**Forecasting:** MSE, MAE, RMSE, MAPE.

**Finance:** directional accuracy; illustrative Sharpe of a long/flat rule on $\mathrm{sign}(\hat{r})$ — labeled non-trading.

**Efficiency:** parameter count, inference latency (ms/batch), relative VRAM.

**Rigor:** multi-seed boxplots; Wilcoxon / average-rank critical difference across settings.

---

## 11. Claims (what to defend)

1. **Adaptive frequency gating** for KAN-based LTSF (vs fixed FFT split).
2. **Band-specific Chebyshev degree** matches capacity to frequency content.
3. **Cross-band attention fusion** beats naive concat/mean; gains are not explained by PlainKAN alone.

---

## 12. Implementation map

| Math object | Code |
|-------------|------|
| RevIN | `timekan/layers/revin.py` |
| Fixed / Adaptive decomp | `timekan/layers/decomp.py` |
| $\phi$, ChebyKAN, Temporal block | `timekan/layers/cheby_kan.py` |
| Attention / concat / mean fusion | `timekan/fusion/attention.py` |
| Full TimeKAN + PlainKAN | `timekan/models/timekan.py` |
| Train / metrics | `exp/train.py`, `metrics/` |
| Figures | `viz/`, `scripts/make_figures.py` |
| Numbers | [`results.md`](results.md) |

Default config example: `configs/timekan_etth1_h96.yaml` ($K=3$, adaptive decomp, attention fusion, Chebyshev basis, RevIN on).
