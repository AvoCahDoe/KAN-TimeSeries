# TimeKAN Results

Smoke-run numbers (short epochs for pipeline validation). Re-train with full configs for paper-ready tables.

**Source of truth:** `[results/summary.csv](results/summary.csv)`  
**Regenerate figures:** `python scripts/make_figures.py --results results/ --out figures/`

---

## Main forecasting table

Best seed per model / dataset / horizon. Metrics on the test split.

### ETTh1 — H = 96


| Model        | MSE        | MAE        | RMSE       | MAPE       | Params     | ms/batch  |
| ------------ | ---------- | ---------- | ---------- | ---------- | ---------- | --------- |
| PatchTST     | 0.3863     | 0.4009     | 0.6215     | 940.0      | 59,854     | 3.31      |
| DLinear      | 0.3884     | 0.4022     | 0.6232     | 866.6      | 18,624     | 1.15      |
| iTransformer | 0.3894     | 0.4089     | 0.6241     | 969.6      | 31,694     | 4.07      |
| Autoformer   | 0.3981     | 0.4074     | 0.6309     | 965.7      | 18,694     | 2.48      |
| FEDformer    | 0.3997     | 0.4075     | 0.6323     | 1028.9     | 12,462     | 3.41      |
| **TimeKAN**  | **0.4039** | **0.4135** | **0.6355** | **1063.8** | **77,320** | **17.32** |
| NLinear      | 0.4098     | 0.4165     | 0.6402     | 1002.9     | 9,312      | 0.48      |
| PlainKAN     | 0.4101     | 0.4223     | 0.6404     | 975.1      | 20,309     | 3.74      |
| TCN          | 0.4248     | 0.4296     | 0.6518     | 1167.4     | 2,087,758  | 7.62      |
| Informer     | 0.4399     | 0.4315     | 0.6632     | 1018.2     | 38,293     | 4.86      |
| LSTM         | 0.4402     | 0.4464     | 0.6635     | 1025.6     | 35,886     | 2.55      |
| Naive        | 1.2944     | 0.7132     | 1.1377     | 1663.4     | 0          | —         |




### ETTh1 — longer horizons (TimeKAN + selected baselines)


| Horizon | Model    | MSE    | MAE    | RMSE   |
| ------- | -------- | ------ | ------ | ------ |
| 192     | TimeKAN  | 0.4551 | 0.4558 | 0.6746 |
| 336     | TimeKAN  | 0.4885 | 0.4721 | 0.6989 |
| 720     | DLinear  | 0.5224 | 0.5168 | 0.7228 |
| 720     | PatchTST | 0.5719 | 0.5162 | 0.7563 |
| 720     | TimeKAN  | 0.6175 | 0.5534 | 0.7858 |




### Weather — H = 96


| Model       | MSE        | MAE        | RMSE       | MAPE     | Params     | ms/batch  |
| ----------- | ---------- | ---------- | ---------- | -------- | ---------- | --------- |
| **TimeKAN** | **0.0098** | **0.0568** | **0.0992** | **10.0** | **79,154** | **15.06** |
| DLinear     | 0.0796     | 0.1658     | 0.2821     | 25.0     | 18,624     | 0.32      |




### Finance — log-returns, H = 20


| Model       | MSE        | MAE        | RMSE       | Dir. acc. | Illustrative Sharpe | ms/batch  |
| ----------- | ---------- | ---------- | ---------- | --------- | ------------------- | --------- |
| **TimeKAN** | **0.8694** | **0.7316** | **0.9324** | **0.594** | **0.708**           | **14.45** |


Illustrative Sharpe = long/flat on sign of forecast — **not a trading claim**.

---



## Ablations (ETTh1, H = 96)


| Tag                                         | K   | Decomp   | Fusion    | Basis     | RMSE   | MAE    |
| ------------------------------------------- | --- | -------- | --------- | --------- | ------ | ------ |
| basis_mlp                                   | 3   | fixed    | attention | mlp       | 0.6295 | 0.4104 |
| decomp_fixed / fusion_attention / k_bands_3 | 3   | fixed    | attention | chebyshev | 0.6339 | 0.4173 |
| decomp_adaptive                             | 3   | adaptive | attention | chebyshev | 0.6355 | 0.4135 |
| degree_uniform                              | 3   | fixed    | attention | chebyshev | 0.6358 | 0.4232 |
| k_bands_2                                   | 2   | fixed    | attention | chebyshev | 0.6360 | 0.4158 |
| fusion_concat                               | 3   | fixed    | concat    | chebyshev | 0.6406 | 0.4220 |
| basis_bspline                               | 3   | fixed    | attention | bspline   | 0.6468 | 0.4205 |
| k_bands_1                                   | 1   | fixed    | attention | chebyshev | 0.6470 | 0.4283 |
| k_bands_4                                   | 4   | fixed    | attention | chebyshev | 0.6506 | 0.4208 |
| k_bands_5                                   | 5   | fixed    | attention | chebyshev | 0.6632 | 0.4417 |
| revin_off                                   | 3   | fixed    | attention | chebyshev | 0.7484 | 0.5414 |


**Takeaways (smoke):** RevIN matters a lot; K≈2–3 looks best; attention fusion slightly beats concat; PlainKAN (no decomp) sits behind the stronger linear/Transformer baselines at H=96.

---



## Graphs

All under `[figures/](figures/)` (PNG + PDF).

### Headline comparison


| Fig | Preview   | Description              |
| --- | --------- | ------------------------ |
| 01  | RMSE      | RMSE by model × dataset  |
| 01b | MAE       | MAE companion            |
| 02  | Faceted H | RMSE faceted by horizon  |
| 03  | Radar     | Multi-metric profile     |
| 04  | Win-rate  | TimeKAN win-rate heatmap |




### Forecast quality


| Fig | Preview    | Description              |
| --- | ---------- | ------------------------ |
| 05  | Pred vs GT | Calm vs volatile windows |
| 06  | Multi-H    | Multi-horizon overlay    |
| 07  | Residuals  | Residual QQ + ACF        |
| 08  | Channels   | Per-channel MAE          |




### Long-horizon robustness


| Fig | Preview     | Description             |
| --- | ----------- | ----------------------- |
| 09  | Heatmap     | Error × horizon heatmap |
| 10  | Scaling     | RMSE vs H curves        |
| 11  | Degradation | RMSE@720 / RMSE@96      |




### Frequency & interpretability


| Fig | Preview      | Description                |
| --- | ------------ | -------------------------- |
| 12  | Decomp       | Signal + K bands           |
| 13  | Spectrum     | DFT + band masks           |
| 14  | Attn         | Cross-band attention       |
| 15  | Regime       | Calm vs shock attention    |
| 16  | Contribution | Band contribution timeline |




### Ablations


| Fig | Preview | Description                |
| --- | ------- | -------------------------- |
| 17  | K       | Number of bands K          |
| 18  | Basis   | Chebyshev / B-spline / MLP |
| 19  | Decomp  | Fixed vs adaptive          |
| 20  | Fusion  | Attention vs concat        |
| 21  | Degree  | Degree schedule            |
| 22  | RevIN   | RevIN on/off               |




### Efficiency & rigor


| Fig | Preview    | Description                 |
| --- | ---------- | --------------------------- |
| 23  | Efficiency | Params vs RMSE              |
| 24  | Latency    | Inference latency           |
| 25  | CD         | Critical difference / ranks |
| 26  | Seeds      | Multi-seed boxplots         |
| 27  | Curves     | Training curves             |
| 28  | VRAM       | RTX 4070 VRAM guide         |




### Finance track


| Fig | Preview | Description                  |
| --- | ------- | ---------------------------- |
| 29  | Returns | Returns + direction hit-rate |
| 30  | Equity  | Illustrative equity curve    |
| 31  | Vol     | Error by vol regime          |


---



## How to refresh

```bash
# train more runs…
python scripts/run_main_table.py --datasets ETTh1 Weather --horizons 96 192 336 720 --epochs 10
python scripts/run_stats_ablations.py --mode both --epochs 8
python scripts/run_finance.sh

# rebuild table + all graphs
python -c "from exp.train import aggregate_summary; aggregate_summary()"
python scripts/make_figures.py
```

Artifact layout: `results/{dataset}/{model}/h{H}/seed{S}/[tag]/` → `metrics.json`, `preds.npz`, `timing.json`, `aux.npz`, `history.json`.