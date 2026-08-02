# Figure captions checklist (TimeKAN)

Use with `figures/*.png` produced by `scripts/make_figures.py`.

| ID | File | Caption stub |
|----|------|--------------|
| 1 | 01_grouped_bar_rmse.png | RMSE of TimeKAN vs baselines across datasets (best seed). |
| 1b | 01b_grouped_bar_mae.png | MAE companion to Fig. 1. |
| 2 | 02_faceted_horizon_rmse.png | RMSE by model, faceted by forecast horizon H∈{96,192,336,720}. |
| 3 | 03_radar_metrics.png | Multi-metric profile (inverted errors) for TimeKAN and top baselines. |
| 4 | 04_winrate_heatmap.png | Fraction of baselines beaten by TimeKAN per dataset×horizon. |
| 5 | 05_pred_vs_gt.png | Predicted vs ground truth on calm and volatile windows. |
| 6 | 06_multi_horizon_overlay.png | Same series forecast at multiple horizons. |
| 7 | 07_residual_qq_acf.png | Residual QQ plot and autocorrelation. |
| 8 | 08_per_channel_error.png | Per-channel MAE for high-dimensional datasets. |
| 9 | 09_error_horizon_heatmap.png | RMSE heatmap over models and horizons. |
| 10 | 10_horizon_scaling.png | RMSE vs horizon H (long-horizon robustness). |
| 11 | 11_degradation_ratio.png | RMSE@720 / RMSE@96 degradation ratio. |
| 12 | 12_decomp_stack.png | Original signal and K recovered frequency bands. |
| 13 | 13_spectrum_masks.png | DFT spectrum and fixed/learned band masks. |
| 14 | 14_attn_heatmap.png | Cross-band attention α_k over forecast steps. |
| 15 | 15_attn_regime_montage.png | Attention under calm vs shock regimes. |
| 16 | 16_band_contribution.png | Stacked band attention mass over the horizon. |
| 17 | 17_ablation_k_bands.png | Ablation on number of bands K. |
| 18 | 18_ablation_basis.png | Chebyshev vs B-spline vs MLP basis. |
| 19 | 19_ablation_decomp.png | Fixed vs adaptive frequency decomposition. |
| 20 | 20_ablation_fusion.png | Attention vs concat/mean fusion. |
| 21 | 21_ablation_degree.png | Uniform vs band-specific Chebyshev degree. |
| 22 | 22_ablation_revin.png | RevIN on/off. |
| 23 | 23_efficiency_scatter.png | Parameters vs RMSE (bubble ∝ latency). |
| 24 | 24_latency_bar.png | Inference latency per model. |
| 25 | 25_critical_difference.png | Average-rank CD diagram + Wilcoxon notes. |
| 26 | 26_seed_boxplots.png | Multi-seed RMSE variability. |
| 27 | 27_training_curves.png | Train/val curves for TimeKAN vs baselines. |
| 28 | 28_vram_heatmap.png | Relative VRAM guide for RTX 4070 8GB. |
| 29 | 29_finance_returns.png | Log-return forecasts and directional accuracy. |
| 30 | 30_equity_curve.png | Illustrative long/flat equity curve (not a trading claim). |
| 31 | 31_vol_regime_error.png | Forecast error conditioned on volatility regime. |
