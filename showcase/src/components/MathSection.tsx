import { Reveal } from './Reveal'
import styles from './MathSection.module.css'

export function MathSection() {
  return (
    <section className={`section ${styles.math}`} id="math">
      <div className="sectionInner">
        <Reveal>
          <p className="sectionEyebrow">Formulation</p>
          <h2 className="sectionTitle">Problem and four-stage architecture</h2>
          <p className="sectionLead">
            Lookback X ∈ ℝ<sup>L×C</sup> → forecast Ŷ ∈ ℝ<sup>H×C</sup>. TimeKAN implements decompose → specialize →
            fuse.
          </p>
        </Reveal>

        <Reveal>
          <div className={styles.equations}>
            <div className={styles.block}>
              <h3>Forecasting objective</h3>
              <p className={styles.note}>Multivariate lookback of length L, C channels; predict the next H steps.</p>
              <div className={styles.eq} aria-label="Forecasting objective">
                X = {'{'}x₁…x_L{'}'} → Ŷ = {'{'}x̂<sub>L+1</sub>…x̂<sub>L+H</sub>{'}'}
              </div>
            </div>

            <div className={styles.grid}>
              <div className={styles.block}>
                <h3>A — RevIN</h3>
                <p className={styles.note}>Per-channel instance norm on the lookback; invert after the head.</p>
                <div className={styles.eq} aria-label="RevIN">
                  X̃<sub>t,c</sub> = γ<sub>c</sub> (X<sub>t,c</sub> − μ<sub>c</sub>) / σ<sub>c</sub> + β<sub>c</sub>
                </div>
              </div>

              <div className={styles.block}>
                <h3>B — Frequency bands</h3>
                <p className={styles.note}>Fixed hard masks or adaptive soft masks α from an MLP on the spectrum.</p>
                <div className={styles.eq} aria-label="Band reconstruction">
                  X<sup>(k)</sup> = ℱ⁻¹( M<sup>(k)</sup> ⊙ ℱ(X̃) )
                </div>
              </div>
            </div>

            <div className={styles.grid}>
              <div className={styles.block}>
                <h3>C — Chebyshev edge φ</h3>
                <p className={styles.note}>Per-band M-KAN with degree schedule (low freq → lower degree).</p>
                <div className={styles.eq} aria-label="Chebyshev edge">
                  φ(x) = Σ<sub>i=0…d</sub> c<sub>i</sub> T<sub>i</sub>(tanh(x))
                </div>
              </div>

              <div className={styles.block}>
                <h3>D — Cross-band fusion</h3>
                <p className={styles.note}>Attention over band embeddings Z<sup>(k)</sup> at each forecast step.</p>
                <div className={styles.eq} aria-label="Attention fusion">
                  α<sub>t</sub> = softmax( q<sub>t</sub> Kᵀ / √d ) → Ŷ<sub>t</sub> = Σ<sub>k</sub> α<sub>t,k</sub> Z<sub>t</sub>
                  <sup>(k)</sup>
                </div>
              </div>
            </div>

            <div className={styles.block}>
              <h3>Stages in one line</h3>
              <p className={styles.note}>
                X → RevIN → {'{'}X<sup>(1)</sup>…X<sup>(K)</sup>{'}'} → {'{'}Z<sup>(1)</sup>…Z<sup>(K)</sup>{'}'} →
                attention fuse → linear → RevIN⁻¹ → Ŷ. PlainKAN drops Stages B/D to test whether gains come from
                decomposition vs “KAN alone.”
              </p>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  )
}
