import { Reveal } from './Reveal'
import styles from './Introduction.module.css'

export function Introduction() {
  return (
    <section className={`section ${styles.intro}`} id="introduction">
      <div className="sectionInner">
        <Reveal>
          <p className="sectionEyebrow">Introduction</p>
          <h1 className={styles.title}>Frequency-decomposed KAN forecasting</h1>
          <p className={styles.subtitle}>
            Motivation, formulation, and measured smoke-run results for TimeKAN against linear and Transformer LTSF
            baselines.
          </p>
        </Reveal>

        <div className={styles.body}>
          <Reveal>
            <div className={styles.prose}>
              <p>
                Long-term forecasting mixes trend, seasonality, and noise in one window. A single shared backbone often
                underfits some bands and overfits others. Different frequency components also tend to need different
                capacity.
              </p>
              <p>
                <strong>TimeKAN</strong> follows a simple recipe: normalize with RevIN, split the lookback into{' '}
                <strong>K frequency bands</strong>, forecast each band with a Chebyshev M-KAN, then fuse with
                cross-band attention before denormalizing.
              </p>
              <p>
                This page reports ETTh1 / Weather / Finance smoke numbers from <code>results.md</code>, plus
                ablations and frequency-interpretability figures. Short epochs were used for pipeline validation —
                re-train with full configs for paper-ready tables.
              </p>
            </div>
          </Reveal>

          <Reveal delay={0.06}>
            <aside className={styles.aside}>
              <div className={styles.chip}>
                <strong>Pipeline</strong>
                <span>RevIN → freq. decomp → per-band M-KAN → attention fusion → denorm.</span>
              </div>
              <div className={styles.chip}>
                <strong>Baselines</strong>
                <span>DLinear, NLinear, PatchTST, iTransformer, Informer, Autoformer, FEDformer, PlainKAN, …</span>
              </div>
              <div className={styles.chip}>
                <strong>Tracks</strong>
                <span>ETTh1 (H=96–720), Weather (H=96), Finance log-returns (H=20).</span>
              </div>
            </aside>
          </Reveal>
        </div>
      </div>
    </section>
  )
}
