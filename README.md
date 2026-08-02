# TimeKAN: Frequency-Decomposed KAN for Long-Term Forecasting

RevIN → frequency decomposition (fixed / adaptive) → per-band Chebyshev M-KAN → cross-band attention fusion → denorm.

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py --datasets ETTh1
python exp/train.py --config configs/timekan_etth1_h96.yaml
python scripts/make_figures.py --results results/ --out figures/
```

Sweep helpers: `scripts/run_ett.sh`, `scripts/run_main_table.py`, `scripts/run_ablations.sh`, `scripts/run_finance.sh`, `scripts/run_stats_ablations.py`.

---

## Results

Full metric tables + all graph previews: **[`results.md`](results.md)**.

Artifacts live under `results/{dataset}/{model}/h{H}/seed{S}/[tag]/`:

| File | Contents |
|------|----------|
| `metrics.json` | MSE, MAE, RMSE, MAPE (+ finance: directional accuracy, illustrative Sharpe) |
| `timing.json` | params, ms/batch, epochs |
| `preds.npz` | `pred`, `true` tensors |
| `aux.npz` | TimeKAN bands / masks / attention (when available) |
| `history.json` | train loss & val MSE curves |
| `summary.csv` | flat table of all runs (feeds the figure factory) |

### Current smoke runs

| Scope | Coverage |
|-------|----------|
| Datasets | ETTh1, Weather, Finance |
| Models | TimeKAN, PlainKAN, DLinear, NLinear, Naive, LSTM, TCN, PatchTST, iTransformer, Informer, Autoformer, FEDformer |
| Horizons | 96 / 192 / 336 / 720 (ETT); 20 (Finance) |
| Ablations | K bands, basis, fixed vs adaptive decomp, fusion, RevIN, degree schedule |
| Seeds | 2021, 2022 (subset) |

> Smoke training used short epochs for pipeline validation — re-run with full `epochs` in configs for paper numbers.

Regenerate the aggregate table anytime:

```bash
python -c "from exp.train import aggregate_summary; aggregate_summary()"
```

---

## Graphs

All figures are produced by `python scripts/make_figures.py` into [`figures/`](figures/) (PNG + PDF @ 300dpi). Captions: [`paper/figure_captions.md`](paper/figure_captions.md).

### 1. Headline comparison

| Fig | File | What it shows |
|-----|------|----------------|
| 01 | [`01_grouped_bar_rmse.png`](figures/01_grouped_bar_rmse.png) | RMSE × model × dataset |
| 01b | [`01b_grouped_bar_mae.png`](figures/01b_grouped_bar_mae.png) | MAE companion |
| 02 | [`02_faceted_horizon_rmse.png`](figures/02_faceted_horizon_rmse.png) | RMSE faceted by horizon |
| 03 | [`03_radar_metrics.png`](figures/03_radar_metrics.png) | Multi-metric profile |
| 04 | [`04_winrate_heatmap.png`](figures/04_winrate_heatmap.png) | TimeKAN win-rate vs others |

![Grouped RMSE](figures/01_grouped_bar_rmse.png)

![Win-rate heatmap](figures/04_winrate_heatmap.png)

### 2. Forecast quality

| Fig | File | What it shows |
|-----|------|----------------|
| 05 | [`05_pred_vs_gt.png`](figures/05_pred_vs_gt.png) | Pred vs GT — calm & volatile windows |
| 06 | [`06_multi_horizon_overlay.png`](figures/06_multi_horizon_overlay.png) | Same window, multiple H |
| 07 | [`07_residual_qq_acf.png`](figures/07_residual_qq_acf.png) | Residual QQ + ACF |
| 08 | [`08_per_channel_error.png`](figures/08_per_channel_error.png) | Per-channel MAE strip |

![Pred vs GT](figures/05_pred_vs_gt.png)

### 3. Long-horizon robustness

| Fig | File | What it shows |
|-----|------|----------------|
| 09 | [`09_error_horizon_heatmap.png`](figures/09_error_horizon_heatmap.png) | RMSE: model × horizon |
| 10 | [`10_horizon_scaling.png`](figures/10_horizon_scaling.png) | RMSE vs H curves |
| 11 | [`11_degradation_ratio.png`](figures/11_degradation_ratio.png) | RMSE@720 / RMSE@96 |

![Horizon scaling](figures/10_horizon_scaling.png)

### 4. Frequency & interpretability (TimeKAN)

| Fig | File | What it shows |
|-----|------|----------------|
| 12 | [`12_decomp_stack.png`](figures/12_decomp_stack.png) | Original + K recovered bands |
| 13 | [`13_spectrum_masks.png`](figures/13_spectrum_masks.png) | DFT spectrum + band masks |
| 14 | [`14_attn_heatmap.png`](figures/14_attn_heatmap.png) | Cross-band attention α |
| 15 | [`15_attn_regime_montage.png`](figures/15_attn_regime_montage.png) | Calm vs shock attention |
| 16 | [`16_band_contribution.png`](figures/16_band_contribution.png) | Band mass over horizon |

![Decomposition stack](figures/12_decomp_stack.png)

![Attention heatmap](figures/14_attn_heatmap.png)

### 5. Ablations

| Fig | File | What it shows |
|-----|------|----------------|
| 17 | [`17_ablation_k_bands.png`](figures/17_ablation_k_bands.png) | Number of bands K |
| 18 | [`18_ablation_basis.png`](figures/18_ablation_basis.png) | Chebyshev vs B-spline vs MLP |
| 19 | [`19_ablation_decomp.png`](figures/19_ablation_decomp.png) | Fixed vs adaptive decomp |
| 20 | [`20_ablation_fusion.png`](figures/20_ablation_fusion.png) | Attention vs concat / mean |
| 21 | [`21_ablation_degree.png`](figures/21_ablation_degree.png) | Uniform vs band-specific degree |
| 22 | [`22_ablation_revin.png`](figures/22_ablation_revin.png) | RevIN on / off |

![Fusion ablation](figures/20_ablation_fusion.png)

### 6. Efficiency & rigor

| Fig | File | What it shows |
|-----|------|----------------|
| 23 | [`23_efficiency_scatter.png`](figures/23_efficiency_scatter.png) | Params vs RMSE (bubble ∝ latency) |
| 24 | [`24_latency_bar.png`](figures/24_latency_bar.png) | Inference ms/batch |
| 25 | [`25_critical_difference.png`](figures/25_critical_difference.png) | Average-rank CD + Wilcoxon |
| 26 | [`26_seed_boxplots.png`](figures/26_seed_boxplots.png) | Multi-seed RMSE |
| 27 | [`27_training_curves.png`](figures/27_training_curves.png) | Train / val curves |
| 28 | [`28_vram_heatmap.png`](figures/28_vram_heatmap.png) | RTX 4070 8GB VRAM guide |

![Efficiency scatter](figures/23_efficiency_scatter.png)

![Critical difference](figures/25_critical_difference.png)

### 7. Finance track

| Fig | File | What it shows |
|-----|------|----------------|
| 29 | [`29_finance_returns.png`](figures/29_finance_returns.png) | Log-return forecast + direction hit-rate |
| 30 | [`30_equity_curve.png`](figures/30_equity_curve.png) | Illustrative long/flat equity (not a trading claim) |
| 31 | [`31_vol_regime_error.png`](figures/31_vol_regime_error.png) | Error by volatility regime |

![Finance returns](figures/29_finance_returns.png)

---

## Layout

| Path | Role |
|------|------|
| [`FORMULATION.md`](FORMULATION.md) | Problem statement + math for TimeKAN (Stages A–D) |
| `timekan/` | Model: RevIN, FFT/adaptive decomp, ChebyKAN, attention fusion |
| `baselines/` | Naive, ARIMA, LSTM, TCN, DLinear, NLinear, PatchTST, iTransformer, PlainKAN, … |
| `exp/` | Train / test harness |
| `viz/` | Figure factory (31 plot types) |
| `configs/` | YAML experiment configs (+ `configs/ablations/`) |
| `data/` | Download + loaders |
| `metrics/` | MSE, MAE, RMSE, MAPE, directional accuracy, Sharpe |
| `results/` | Run artifacts + `summary.csv` |
| `figures/` | Published PNG/PDF graphs |
| `paper/` | Outline + figure captions |

## Datasets

ETT (h1/h2/m1/m2), Weather, Electricity, Finance (AAPL/TSLA/^GSPC log-returns via yfinance).

## Citation hooks

RevIN; efficient-kan–style Chebyshev layers; Informer / Autoformer / FEDformer / PatchTST / iTransformer / DLinear baselines.
