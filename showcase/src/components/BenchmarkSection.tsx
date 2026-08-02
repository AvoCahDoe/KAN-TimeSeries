import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import {
  datasets,
  formatParams,
  horizonRows,
  metricsForDataset,
  modelColor,
  type DatasetId,
} from '../data/metrics'
import { financeExtras } from '../data/ablations'
import { chartTheme } from '../chartTheme'
import { Reveal } from './Reveal'
import styles from './BenchmarkSection.module.css'

const tick = { fill: chartTheme.tick, fontSize: 12 } as const
const tooltipStyle = chartTheme.tooltip

export function BenchmarkSection() {
  const [dataset, setDataset] = useState<DatasetId>('ETTh1')
  const rows = useMemo(() => metricsForDataset(dataset), [dataset])

  const barData = rows
    .filter((r) => r.model !== 'Naive' || dataset === 'ETTh1')
    .slice(0, dataset === 'ETTh1' ? 11 : undefined)
    .map((r) => ({
      name: r.model,
      rmse: Number(r.rmse.toFixed(4)),
      fill: modelColor(r.model),
    }))

  const scatterData = rows
    .filter((r) => r.params > 0)
    .map((r) => ({
      x: r.params,
      y: Number(r.rmse.toFixed(4)),
      name: r.model,
      fill: modelColor(r.model),
    }))

  const horizonChart = useMemo(() => {
    const byH = new Map<number, Record<string, number>>()
    for (const row of horizonRows) {
      const cur = byH.get(row.horizon) ?? { horizon: row.horizon }
      cur[row.model] = row.rmse
      byH.set(row.horizon, cur)
    }
    return [...byH.values()].sort((a, b) => (a.horizon as number) - (b.horizon as number))
  }, [])

  const top = rows[0]
  const timekan = rows.find((r) => r.isTimeKan)

  const highlights =
    dataset === 'ETTh1'
      ? [
          { value: top ? top.rmse.toFixed(4) : '—', label: `Best RMSE @ H=96 (${top?.model ?? '—'})` },
          { value: timekan ? timekan.rmse.toFixed(4) : '—', label: 'TimeKAN RMSE @ H=96' },
          { value: '0.7858', label: 'TimeKAN RMSE @ H=720' },
        ]
      : dataset === 'Weather'
        ? [
            { value: '0.0992', label: 'TimeKAN RMSE' },
            { value: '0.2821', label: 'DLinear RMSE' },
            { value: '~2.8×', label: 'RMSE gap vs DLinear' },
          ]
        : [
            { value: '0.9324', label: 'TimeKAN RMSE (H=20)' },
            { value: `${(financeExtras.directionalAccuracy * 100).toFixed(1)}%`, label: 'Directional accuracy' },
            { value: financeExtras.illustrativeSharpe.toFixed(3), label: 'Illustrative Sharpe (not a claim)' },
          ]

  return (
    <section className={`section ${styles.benchmarks}`} id="results">
      <div className="sectionInner">
        <Reveal>
          <p className="sectionEyebrow">Results</p>
          <h2 className="sectionTitle">RMSE, params, and horizon scaling</h2>
          <p className="sectionLead">
            Smoke-run snapshot from <code>results.md</code>. Lower RMSE is better. Switch datasets below.
          </p>
        </Reveal>

        <Reveal>
          <div className={styles.tabs} role="tablist" aria-label="Dataset">
            {datasets.map((d) => (
              <button
                key={d.id}
                type="button"
                role="tab"
                aria-selected={dataset === d.id}
                className={dataset === d.id ? styles.tabActive : styles.tab}
                onClick={() => setDataset(d.id)}
              >
                {d.label} (H={d.horizon})
              </button>
            ))}
          </div>

          <div className={styles.highlights}>
            {highlights.map((h) => (
              <div key={h.label} className={styles.stat}>
                <strong>{h.value}</strong>
                <span>{h.label}</span>
              </div>
            ))}
          </div>
        </Reveal>

        <Reveal>
          <div className={styles.charts}>
            <div className={styles.panel}>
              <h3 className={styles.panelTitle}>Test RMSE</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={barData} margin={{ top: 12, right: 12, left: 4, bottom: 48 }}>
                  <CartesianGrid stroke={chartTheme.grid} vertical={false} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="name"
                    tick={tick}
                    axisLine={false}
                    tickLine={false}
                    interval={0}
                    angle={dataset === 'ETTh1' ? -35 : 0}
                    textAnchor={dataset === 'ETTh1' ? 'end' : 'middle'}
                    height={dataset === 'ETTh1' ? 60 : 30}
                  />
                  <YAxis tick={tick} axisLine={false} tickLine={false} width={48} domain={['auto', 'auto']} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(value) => [String(value), 'RMSE']} cursor={{ fill: chartTheme.cursor }} />
                  <Bar dataKey="rmse" radius={[5, 5, 0, 0]} maxBarSize={52}>
                    {barData.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className={styles.panel}>
              <h3 className={styles.panelTitle}>Params vs RMSE</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 12, right: 16, left: 4, bottom: 12 }}>
                  <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="params"
                    tickFormatter={formatParams}
                    tick={tick}
                    axisLine={false}
                    tickLine={false}
                    scale="log"
                    domain={['auto', 'auto']}
                    label={{ value: 'Parameters', position: 'insideBottom', offset: -4, fill: chartTheme.tick, fontSize: 11 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="rmse"
                    tick={tick}
                    axisLine={false}
                    tickLine={false}
                    width={48}
                    domain={['auto', 'auto']}
                    label={{ value: 'RMSE', angle: -90, position: 'insideLeft', fill: chartTheme.tick, fontSize: 11 }}
                  />
                  <ZAxis range={[120, 120]} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    cursor={{ strokeDasharray: '3 3', stroke: '#9fd4cd' }}
                    formatter={(value, name) => {
                      if (name === 'params') return [formatParams(Number(value)), 'Params']
                      return [String(value), 'RMSE']
                    }}
                    labelFormatter={(_, payload) => (payload?.[0]?.payload as { name?: string })?.name ?? ''}
                  />
                  <Scatter data={scatterData}>
                    {scatterData.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Reveal>

        {dataset === 'ETTh1' && (
          <Reveal delay={0.05}>
            <div className={styles.curvesPanel}>
              <h3 className={styles.panelTitle}>ETTh1 — RMSE vs horizon</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={horizonChart} margin={{ top: 12, right: 20, left: 4, bottom: 8 }}>
                  <CartesianGrid stroke={chartTheme.grid} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="horizon"
                    tick={tick}
                    axisLine={false}
                    tickLine={false}
                    label={{ value: 'Horizon H', position: 'insideBottom', offset: -2, fill: chartTheme.tick, fontSize: 11 }}
                  />
                  <YAxis tick={tick} axisLine={false} tickLine={false} width={48} domain={['auto', 'auto']} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ color: chartTheme.tick, fontSize: 12 }} />
                  <Line type="monotone" dataKey="TimeKAN" stroke="#0f7a72" strokeWidth={2.4} dot={{ r: 3.5 }} connectNulls />
                  <Line type="monotone" dataKey="DLinear" stroke="#4a5a70" strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} connectNulls />
                  <Line type="monotone" dataKey="PatchTST" stroke="#3d5a80" strokeWidth={2} strokeDasharray="2 3" dot={{ r: 3 }} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className={styles.note}>
              At H=96, linear / PatchTST lead slightly; TimeKAN is competitive. Long horizons still degrade — smoke
              epochs, not final paper numbers.
            </p>
          </Reveal>
        )}
      </div>
    </section>
  )
}
