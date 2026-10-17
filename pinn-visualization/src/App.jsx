import { useRef, useState } from 'react'
import ControlBar from './components/ControlBar'
import VisualizationPanel from './components/VisualizationPanel'
import StatsPanel from './components/StatsPanel'
import { INITIAL_CONDITION, MODEL_STATS } from './constants'
import './App.css'

const PAPER_URL =
  'https://repository.gatech.edu/entities/publication/3f8db9a5-25da-4dd9-b0e0-023bcaf23dd2'

const API_BASE = 'http://localhost:8000'

function App() {
  const [model, setModel] = useState('')
  const [colorScheme, setColorScheme] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const abortRef = useRef(null)

  const runPrediction = () => {
    if (!model || !colorScheme) return

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(true)

    fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        color_scheme: colorScheme,
        initial_condition: INITIAL_CONDITION,
      }),
      signal: controller.signal,
    })
      .then((r) => r.json())
      .then(setResult)
      .catch((err) => {
        if (err.name !== 'AbortError') console.error(err)
      })
      .finally(() => setLoading(false))
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Explore PINNs for Cardiac Modeling</h1>
      </header>

      <ControlBar
        model={model}
        onModelChange={setModel}
        colorScheme={colorScheme}
        onColorSchemeChange={setColorScheme}
        onReload={runPrediction}
        canReload={Boolean(model && colorScheme)}
        loading={loading}
      />

      <main className="viz-grid">
        <VisualizationPanel
          label="Network (model type)"
          src={result?.network_image}
          loading={loading}
        />
        <VisualizationPanel
          label="Ground Truth"
          src={result?.ground_truth_image}
          loading={loading}
        />
        <StatsPanel
          trainingPoints={MODEL_STATS[model]?.trainingPoints}
          accuracy={MODEL_STATS[model]?.accuracy}
        />
      </main>

      <footer className="app-footer">
        Check out my paper here!{' '}
        <a href={PAPER_URL} target="_blank" rel="noreferrer">
          {PAPER_URL}
        </a>
      </footer>
    </div>
  )
}

export default App
