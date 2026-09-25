import PlotCanvas from '../PlotCanvas'
import ActionChips from '../ActionChips'

/**
 * Step 6 — Validation
 */
export default function ValidationStep({ modelMetrics, plotHistory, actions, tips, onAction }) {
  function fmt(v) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
    return Number(v).toFixed(4)
  }

  return (
    <div className="animate-rise" style={{ display: 'flex', flexDirection: 'column', gap: '16px', minHeight: '100%' }}>
      {/* Dynamic tips + actions */}
      <ActionChips actions={actions} tips={tips} onAction={onAction} />

      {/* Parity / validation plot */}
      <PlotCanvas
        label="Predictions vs. Lab Measurements"
        plotHistory={plotHistory}
        emptyState={
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--color-mist)', padding: '30px', gap: '8px' }}>
            <p style={{ fontSize: '1.5rem' }}>📉</p>
            <p style={{ fontSize: '0.875rem', textAlign: 'center' }}>
              Click <strong style={{ color: 'var(--color-ember)' }}>"Generate a parity plot"</strong> above to visualize model performance
            </p>
          </div>
        }
      />

      {/* Metrics table */}
      {modelMetrics ? (
        <div className="panel" style={{ borderRadius: '14px', overflow: 'hidden' }}>
          <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--color-line)' }}>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: 0 }}>
              Model Metrics · {modelMetrics.algorithm || '—'}
            </p>
          </div>
          <table className="data-table">
            <thead>
              <tr><th>Metric</th><th>Value</th><th>Notes</th></tr>
            </thead>
            <tbody>
              <MetricRow label="R²" value={fmt(modelMetrics.r2_score)} note="Coefficient of determination" />
              <MetricRow label="RMSE" value={fmt(modelMetrics.rmse)} note="Root mean squared error" />
              <MetricRow label="Training samples" value={modelMetrics.n_train ?? '—'} note="" />
              <MetricRow label="Test samples" value={modelMetrics.n_test ?? '—'} note="" />
              {modelMetrics.target_column && (
                <MetricRow label="Target" value={modelMetrics.target_column} note="Soft-sensor output" />
              )}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: '20px', color: 'var(--color-mist)' }}>
          <p style={{ fontSize: '0.875rem' }}>Train a model in Step 5 to populate validation metrics.</p>
        </div>
      )}
    </div>
  )
}

function MetricRow({ label, value, note }) {
  return (
    <tr>
      <td style={{ color: 'var(--color-foam)', fontWeight: 500 }}>{label}</td>
      <td style={{ color: 'var(--color-signal)', fontFamily: 'var(--font-mono)' }}>{value}</td>
      <td style={{ color: 'var(--color-mist)', fontStyle: 'italic' }}>{note}</td>
    </tr>
  )
}
