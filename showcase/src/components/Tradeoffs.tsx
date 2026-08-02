import { Reveal } from './Reveal'
import styles from './Tradeoffs.module.css'

const items = [
  {
    tag: 'Smoke',
    title: 'Numbers are pipeline validation',
    body: 'Short epochs produce the tables here. Full-config re-training is required before claiming paper-ready comparisons.',
  },
  {
    tag: 'Speed',
    title: 'TimeKAN is slower at inference',
    body: 'ETTh1 H=96 sits near ~17 ms/batch vs ~1–4 ms for linear / PatchTST-style baselines. Accuracy gains (when present) are not free.',
  },
  {
    tag: 'ETTh1',
    title: 'Not always first at H=96',
    body: 'PatchTST / DLinear / iTransformer edge TimeKAN slightly on ETTh1@96 in this snapshot. Weather shows a clearer TimeKAN win.',
  },
  {
    tag: 'Ablation',
    title: 'RevIN and K matter more than flashy fusion',
    body: 'Turning RevIN off collapses RMSE; K≈2–3 beats 1 or 5; attention fusion only slightly beats concat in these runs.',
  },
]

export function Tradeoffs() {
  return (
    <section className={`section ${styles.tradeoffs}`} id="tradeoffs">
      <div className="sectionInner">
        <Reveal>
          <p className="sectionEyebrow">Honest tradeoffs</p>
          <h2 className="sectionTitle">How to read these results</h2>
          <p className="sectionLead">
            Keep the caveats next to the wins — especially smoke training and latency.
          </p>
        </Reveal>

        <Reveal>
          <ul className={styles.list}>
            {items.map((item) => (
              <li key={item.title} className={styles.item}>
                <span className={styles.tag}>{item.tag}</span>
                <div>
                  <h3>{item.title}</h3>
                  <p>{item.body}</p>
                </div>
              </li>
            ))}
          </ul>
        </Reveal>
      </div>
    </section>
  )
}
