import { PageHeader } from './components/PageHeader'
import { Introduction } from './components/Introduction'
import { MathSection } from './components/MathSection'
import { BenchmarkSection } from './components/BenchmarkSection'
import { InterpretGallery } from './components/InterpretGallery'
import { Tradeoffs } from './components/Tradeoffs'

export default function App() {
  return (
    <div id="top">
      <PageHeader />
      <main>
        <Introduction />
        <MathSection />
        <BenchmarkSection />
        <InterpretGallery />
        <Tradeoffs />
      </main>
    </div>
  )
}
