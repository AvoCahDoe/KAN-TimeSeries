export type DatasetId = 'ETTh1' | 'Weather' | 'Finance'

export type RunMetric = {
  id: string
  model: string
  dataset: DatasetId
  horizon: number
  mse: number
  mae: number
  rmse: number
  params: number
  msPerBatch: number | null
  isTimeKan: boolean
}

/** Best / reported snapshot from results.md (smoke runs). */
export const mainTable: RunMetric[] = [
  { id: 'patchtst_etth1_96', model: 'PatchTST', dataset: 'ETTh1', horizon: 96, mse: 0.3863, mae: 0.4009, rmse: 0.6215, params: 59854, msPerBatch: 3.31, isTimeKan: false },
  { id: 'dlinear_etth1_96', model: 'DLinear', dataset: 'ETTh1', horizon: 96, mse: 0.3884, mae: 0.4022, rmse: 0.6232, params: 18624, msPerBatch: 1.15, isTimeKan: false },
  { id: 'itransformer_etth1_96', model: 'iTransformer', dataset: 'ETTh1', horizon: 96, mse: 0.3894, mae: 0.4089, rmse: 0.6241, params: 31694, msPerBatch: 4.07, isTimeKan: false },
  { id: 'autoformer_etth1_96', model: 'Autoformer', dataset: 'ETTh1', horizon: 96, mse: 0.3981, mae: 0.4074, rmse: 0.6309, params: 18694, msPerBatch: 2.48, isTimeKan: false },
  { id: 'fedformer_etth1_96', model: 'FEDformer', dataset: 'ETTh1', horizon: 96, mse: 0.3997, mae: 0.4075, rmse: 0.6323, params: 12462, msPerBatch: 3.41, isTimeKan: false },
  { id: 'timekan_etth1_96', model: 'TimeKAN', dataset: 'ETTh1', horizon: 96, mse: 0.4039, mae: 0.4135, rmse: 0.6355, params: 77320, msPerBatch: 17.32, isTimeKan: true },
  { id: 'nlinear_etth1_96', model: 'NLinear', dataset: 'ETTh1', horizon: 96, mse: 0.4098, mae: 0.4165, rmse: 0.6402, params: 9312, msPerBatch: 0.48, isTimeKan: false },
  { id: 'plainkan_etth1_96', model: 'PlainKAN', dataset: 'ETTh1', horizon: 96, mse: 0.4101, mae: 0.4223, rmse: 0.6404, params: 20309, msPerBatch: 3.74, isTimeKan: false },
  { id: 'tcn_etth1_96', model: 'TCN', dataset: 'ETTh1', horizon: 96, mse: 0.4248, mae: 0.4296, rmse: 0.6518, params: 2087758, msPerBatch: 7.62, isTimeKan: false },
  { id: 'informer_etth1_96', model: 'Informer', dataset: 'ETTh1', horizon: 96, mse: 0.4399, mae: 0.4315, rmse: 0.6632, params: 38293, msPerBatch: 4.86, isTimeKan: false },
  { id: 'lstm_etth1_96', model: 'LSTM', dataset: 'ETTh1', horizon: 96, mse: 0.4402, mae: 0.4464, rmse: 0.6635, params: 35886, msPerBatch: 2.55, isTimeKan: false },
  { id: 'naive_etth1_96', model: 'Naive', dataset: 'ETTh1', horizon: 96, mse: 1.2944, mae: 0.7132, rmse: 1.1377, params: 0, msPerBatch: null, isTimeKan: false },
  { id: 'timekan_weather_96', model: 'TimeKAN', dataset: 'Weather', horizon: 96, mse: 0.0098, mae: 0.0568, rmse: 0.0992, params: 79154, msPerBatch: 15.06, isTimeKan: true },
  { id: 'dlinear_weather_96', model: 'DLinear', dataset: 'Weather', horizon: 96, mse: 0.0796, mae: 0.1658, rmse: 0.2821, params: 18624, msPerBatch: 0.32, isTimeKan: false },
  { id: 'timekan_finance_20', model: 'TimeKAN', dataset: 'Finance', horizon: 20, mse: 0.8694, mae: 0.7316, rmse: 0.9324, params: 92603, msPerBatch: 14.45, isTimeKan: true },
]

export const horizonRows = [
  { horizon: 96, model: 'TimeKAN', mse: 0.4039, mae: 0.4135, rmse: 0.6355 },
  { horizon: 192, model: 'TimeKAN', mse: 0.4551, mae: 0.4558, rmse: 0.6746 },
  { horizon: 336, model: 'TimeKAN', mse: 0.4885, mae: 0.4721, rmse: 0.6989 },
  { horizon: 720, model: 'TimeKAN', mse: 0.6175, mae: 0.5534, rmse: 0.7858 },
  { horizon: 720, model: 'DLinear', mse: 0.5224, mae: 0.5168, rmse: 0.7228 },
  { horizon: 720, model: 'PatchTST', mse: 0.5719, mae: 0.5162, rmse: 0.7563 },
]

export const datasets: { id: DatasetId; label: string; horizon: number }[] = [
  { id: 'ETTh1', label: 'ETTh1', horizon: 96 },
  { id: 'Weather', label: 'Weather', horizon: 96 },
  { id: 'Finance', label: 'Finance', horizon: 20 },
]

export const modelColor = (model: string): string => {
  if (model === 'TimeKAN') return '#0f7a72'
  if (model === 'PlainKAN') return '#1a9b90'
  if (model === 'DLinear' || model === 'NLinear') return '#4a5a70'
  if (model === 'PatchTST' || model === 'iTransformer') return '#3d5a80'
  if (model === 'Naive') return '#9aa3b2'
  return '#6b7c93'
}

export function metricsForDataset(dataset: DatasetId): RunMetric[] {
  return mainTable
    .filter((m) => m.dataset === dataset)
    .sort((a, b) => a.rmse - b.rmse)
}

export function formatParams(n: number): string {
  if (n <= 0) return '—'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`
  return String(n)
}
