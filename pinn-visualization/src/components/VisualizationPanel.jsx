export default function VisualizationPanel({ label, src, loading }) {
  return (
    <div className="viz-panel">
      {src ? (
        <img className="viz-box" src={src} alt={label} />
      ) : (
        <div className="viz-box">
          {loading && <span className="viz-box-status">Loading…</span>}
        </div>
      )}
      <p className="viz-label">{label}</p>
    </div>
  )
}
