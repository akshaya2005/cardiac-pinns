export default function StatsPanel({ trainingPoints, accuracy }) {
  return (
    <div className="stats-panel">
      <p className="stat-row">
        <span className="stat-label">Training points:</span>{' '}
        <span className="stat-value">{trainingPoints ?? '—'}</span>
      </p>
      <p className="stat-row">
        <span className="stat-label">Accuracy:</span>{' '}
        <span className="stat-value">{accuracy ?? '—'}</span>
      </p>
    </div>
  )
}
