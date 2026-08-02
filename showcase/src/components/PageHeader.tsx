import { site } from '../site'
import styles from './PageHeader.module.css'

export function PageHeader() {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <a className={styles.brand} href="#top">
          Time<span>KAN</span>
        </a>
        <nav className={styles.nav} aria-label="Sections">
          <a href="#introduction">Intro</a>
          <a href="#math">Math</a>
          <a href="#results">Results</a>
          <a href="#interpretability">Interpret</a>
          <a href="#tradeoffs">Tradeoffs</a>
          <a href={site.githubUrl} target="_blank" rel="noreferrer">
            Code
          </a>
        </nav>
      </div>
    </header>
  )
}
