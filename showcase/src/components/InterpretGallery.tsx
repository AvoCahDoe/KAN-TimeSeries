import { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ablations, galleryFigures, kBandSweep } from '../data/ablations'
import { chartTheme } from '../chartTheme'
import { Reveal } from './Reveal'
import styles from './InterpretGallery.module.css'

const tick = { fill: chartTheme.tick, fontSize: 12 } as const
const tooltipStyle = chartTheme.tooltip

export function InterpretGallery() {
  const [active, setActive] = useState<(typeof galleryFigures)[number] | null>(null)

  useEffect(() => {
    if (!active) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setActive(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [active])

  const ablationBars = [...ablations]
    .sort((a, b) => a.rmse - b.rmse)
    .map((a) => ({
      name: a.tag.replace(' / fixed', ''),
      rmse: a.rmse,
      highlight: a.tag.includes('revin'),
    }))

  return (
    <section className={`section ${styles.interpret}`} id="interpretability">
      <div className="sectionInner">
        <Reveal>
          <p className="sectionEyebrow">Interpretability & ablations</p>
          <h2 className="sectionTitle">Bands, attention, and what matters</h2>
          <p className="sectionLead">
            Frequency views plus ETTh1 H=96 ablation RMSE. RevIN dominates; K≈2–3 looks best in these smoke runs.
          </p>
        </Reveal>

        <Reveal>
          <div className={styles.grid}>
            <div className={styles.panel}>
              <h3>Ablation RMSE (sorted)</h3>
              <p>ETTh1, H=96 — lower is better.</p>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={ablationBars} layout="vertical" margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
                  <CartesianGrid stroke={chartTheme.grid} horizontal={false} strokeDasharray="3 3" />
                  <XAxis type="number" tick={tick} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
                  <YAxis type="category" dataKey="name" width={118} tick={{ fill: chartTheme.tick, fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(value) => [String(value), 'RMSE']} />
                  <Bar dataKey="rmse" radius={[0, 5, 5, 0]} maxBarSize={18}>
                    {ablationBars.map((d) => (
                      <Cell key={d.name} fill={d.highlight ? '#b45309' : '#0f7a72'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className={styles.panel}>
              <h3>Number of bands K</h3>
              <p>Fixed decomp + attention + Chebyshev.</p>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={kBandSweep} margin={{ top: 12, right: 12, left: 4, bottom: 8 }}>
                  <CartesianGrid stroke={chartTheme.grid} vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="k" tick={tick} axisLine={false} tickLine={false} label={{ value: 'K', position: 'insideBottom', offset: -2, fill: chartTheme.tick, fontSize: 11 }} />
                  <YAxis tick={tick} axisLine={false} tickLine={false} width={48} domain={[0.62, 0.67]} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(value) => [String(value), 'RMSE']} />
                  <Bar dataKey="rmse" fill="#0f7a72" radius={[5, 5, 0, 0]} maxBarSize={48} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className={styles.callouts}>
            <div className={styles.callout}>
              <strong>0.748</strong>
              <span>RMSE with RevIN off — largest ablation hit</span>
            </div>
            <div className={styles.callout}>
              <strong>K ≈ 2–3</strong>
              <span>Best band counts in the smoke sweep</span>
            </div>
          </div>
        </Reveal>

        <Reveal delay={0.05}>
          <div className={styles.gallery}>
            {galleryFigures.map((fig) => (
              <button key={fig.src} type="button" className={styles.card} onClick={() => setActive(fig)}>
                <img src={fig.src} alt={fig.title} loading="lazy" />
                <div className={styles.cardBody}>
                  <strong>{fig.title}</strong>
                  <span>{fig.caption}</span>
                </div>
              </button>
            ))}
          </div>
        </Reveal>
      </div>

      {active && (
        <div
          className={styles.lightbox}
          role="dialog"
          aria-modal="true"
          aria-label={active.title}
          onClick={() => setActive(null)}
        >
          <div className={styles.lightboxInner} onClick={(e) => e.stopPropagation()}>
            <img src={active.src} alt={active.title} />
            <h3>{active.title}</h3>
            <p>{active.caption}</p>
            <button type="button" className={styles.close} onClick={() => setActive(null)}>
              Close
            </button>
          </div>
        </div>
      )}
    </section>
  )
}
