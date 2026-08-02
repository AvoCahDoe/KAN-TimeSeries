export type AblationRow = {
  tag: string
  k: number
  decomp: string
  fusion: string
  basis: string
  rmse: number
  mae: number
}

export const ablations: AblationRow[] = [
  { tag: 'basis_mlp', k: 3, decomp: 'fixed', fusion: 'attention', basis: 'mlp', rmse: 0.6295, mae: 0.4104 },
  { tag: 'k_bands_3 / fixed', k: 3, decomp: 'fixed', fusion: 'attention', basis: 'chebyshev', rmse: 0.6339, mae: 0.4173 },
  { tag: 'decomp_adaptive', k: 3, decomp: 'adaptive', fusion: 'attention', basis: 'chebyshev', rmse: 0.6355, mae: 0.4135 },
  { tag: 'degree_uniform', k: 3, decomp: 'fixed', fusion: 'attention', basis: 'chebyshev', rmse: 0.6358, mae: 0.4232 },
  { tag: 'k_bands_2', k: 2, decomp: 'fixed', fusion: 'attention', basis: 'chebyshev', rmse: 0.636, mae: 0.4158 },
  { tag: 'fusion_concat', k: 3, decomp: 'fixed', fusion: 'concat', basis: 'chebyshev', rmse: 0.6406, mae: 0.422 },
  { tag: 'basis_bspline', k: 3, decomp: 'fixed', fusion: 'attention', basis: 'bspline', rmse: 0.6468, mae: 0.4205 },
  { tag: 'k_bands_1', k: 1, decomp: 'fixed', fusion: 'attention', basis: 'chebyshev', rmse: 0.647, mae: 0.4283 },
  { tag: 'k_bands_4', k: 4, decomp: 'fixed', fusion: 'attention', basis: 'chebyshev', rmse: 0.6506, mae: 0.4208 },
  { tag: 'k_bands_5', k: 5, decomp: 'fixed', fusion: 'attention', basis: 'chebyshev', rmse: 0.6632, mae: 0.4417 },
  { tag: 'revin_off', k: 3, decomp: 'fixed', fusion: 'attention', basis: 'chebyshev', rmse: 0.7484, mae: 0.5414 },
]

export const kBandSweep = [
  { k: 1, rmse: 0.647 },
  { k: 2, rmse: 0.636 },
  { k: 3, rmse: 0.6339 },
  { k: 4, rmse: 0.6506 },
  { k: 5, rmse: 0.6632 },
]

export const financeExtras = {
  directionalAccuracy: 0.594,
  illustrativeSharpe: 0.708,
}

export const galleryFigures = [
  {
    src: '/figures/12_decomp_stack.png',
    title: 'Frequency decomposition',
    caption: 'Original series plus K recovered bands.',
  },
  {
    src: '/figures/13_spectrum_masks.png',
    title: 'Spectrum masks',
    caption: 'DFT spectrum with fixed / adaptive band masks.',
  },
  {
    src: '/figures/14_attn_heatmap.png',
    title: 'Cross-band attention',
    caption: 'Attention weights α across bands and forecast steps.',
  },
  {
    src: '/figures/15_attn_regime_montage.png',
    title: 'Calm vs shock attention',
    caption: 'How fusion shifts under different regimes.',
  },
  {
    src: '/figures/16_band_contribution.png',
    title: 'Band contribution',
    caption: 'Band mass over the forecast horizon.',
  },
  {
    src: '/figures/05_pred_vs_gt.png',
    title: 'Pred vs ground truth',
    caption: 'Calm and volatile windows.',
  },
  {
    src: '/figures/10_horizon_scaling.png',
    title: 'Horizon scaling',
    caption: 'RMSE vs forecast horizon H.',
  },
  {
    src: '/figures/17_ablation_k_bands.png',
    title: 'Ablation: K bands',
    caption: 'RMSE as the number of bands changes.',
  },
  {
    src: '/figures/22_ablation_revin.png',
    title: 'Ablation: RevIN',
    caption: 'RevIN on vs off — large effect in smoke runs.',
  },
  {
    src: '/figures/23_efficiency_scatter.png',
    title: 'Efficiency',
    caption: 'Params vs RMSE (bubble ∝ latency).',
  },
  {
    src: '/figures/29_finance_returns.png',
    title: 'Finance returns',
    caption: 'Log-return forecast and direction hit-rate.',
  },
  {
    src: '/figures/30_equity_curve.png',
    title: 'Illustrative equity',
    caption: 'Long/flat on forecast sign — not a trading claim.',
  },
] as const
