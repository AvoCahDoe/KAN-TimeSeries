# TimeKAN — Paper Outline

## Title
TimeKAN: Frequency-Decomposed Kolmogorov–Arnold Networks for Long-Term Time Series Forecasting

## Abstract (draft)
Long-term forecasting requires modeling heterogeneous dynamics across frequency scales. TimeKAN decomposes a multivariate lookback window into K frequency bands (fixed FFT or adaptive soft masks), applies band-specific Chebyshev KAN blocks with capacity matched to frequency content, and fuses band representations with cross-band attention before RevIN denormalization. On ETT, Weather, Electricity, and a finance log-return track, TimeKAN is competitive with strong linear and Transformer baselines while remaining parameter-efficient. Ablations isolate gains from decomposition, adaptive gating, and attention fusion beyond a plain KAN.

## Sections
1. Introduction — LTSF, frequency mismatch hypothesis, contributions
2. Related work — Transformers (Informer/Autoformer/FEDformer/PatchTST/iTransformer), linear (DLinear), KAN, RevIN
3. Method — Stages A–D with equations
4. Experiments — datasets, baselines, metrics, efficiency
5. Analysis — interpretability (bands, attention), ablations, significance
6. Finance case study — returns, directional accuracy, illustrative Sharpe
7. Conclusion

## Novelty claims
1. Adaptive frequency gating for KAN-based LTSF
2. Band-specific Chebyshev degree schedule
3. Cross-band attention fusion (+ PlainKAN ablation)

## Reproducibility
Configs under `configs/`; figure factory `scripts/make_figures.py`; seeds {2021,2022,2023}.
