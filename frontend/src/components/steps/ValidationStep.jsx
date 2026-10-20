import { ChartScatter } from 'lucide-react'
import PlotCanvas from '../PlotCanvas'
import { CV_LABELS, cvUnit, splitLabel } from '../../utils/splits'

/**
 * Step 6 — Validation
 */
export default function ValidationStep({ modelMetrics, plotHistory }) {
  const cv = modelMetrics?.cv
  const validation = modelMetrics?.validation
  function fmt(v) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
    return Number(v).toFixed(4)
  }

  return (
    <div className="animate-rise" style={{ display: 'flex', flexDirection: 'column', gap: '16px', minHeight: '100%' }}>
      {/* Parity / validation plot */}
      <PlotCanvas
        label="Predictions vs. Lab Measurements"
        plotHistory={plotHistory}
        emptyState={
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--color-mist)', padding: '30px', gap: '8px' }}>
            <ChartScatter size={24} aria-hidden="true" />
            <p style={{ fontSize: '0.875rem', textAlign: 'center' }}>
              Choose <strong style={{ color: 'var(--color-ember)' }}>Parity plot</strong> in the Tools panel above, or ask the Assistant, to visualize model performance
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
              <MetricRow label="Test R²" value={fmt(modelMetrics.r2_score)} note={modelMetrics.r2_score == null ? 'Not tested yet: test the model in Step 5' : 'Coefficient of determination on the held-out rows'} />
              <MetricRow label="Test RMSE" value={fmt(modelMetrics.rmse)} note={modelMetrics.r2_score == null ? '' : 'Root mean squared error on the held-out rows'} />
              {validation && (
                <>
                  <MetricRow label="Validation R²" value={fmt(validation.r2)} note="On the separate validation rows" />
                  <MetricRow label="Validation RMSE" value={fmt(validation.rmse)} note="On the separate validation rows" />
                </>
              )}
              <MetricRow label="Training samples" value={modelMetrics.n_train ?? '—'} note="" />
              {validation && <MetricRow label="Validation samples" value={validation.n_rows ?? '—'} note="" />}
              <MetricRow label="Test samples" value={modelMetrics.n_test ?? '—'} note="" />
              {modelMetrics.split_strategy && (
                <MetricRow
                  label="Train/test split"
                  value={splitLabel(modelMetrics.split_strategy)}
                  note={typeof modelMetrics.test_size === 'number' ? `${Math.round(modelMetrics.test_size * 100)}% held out for testing` : ''}
                />
              )}
              {cv && (
                <>
                  <MetricRow
                    label="Cross-validation"
                    value={CV_LABELS[cv.strategy] ?? cv.strategy}
                    note={`${cv.n_splits} ${cvUnit(cv.strategy)} over ${cv.n_rows} training rows`}
                  />
                  <MetricRow
                    label="CV R²"
                    value={fmt(cv.r2)}
                    note={cv.fold_r2_mean != null ? `Per fold ${fmt(cv.fold_r2_mean)} ± ${fmt(cv.fold_r2_std)}` : 'Pooled over every left-out row'}
                  />
                  <MetricRow
                    label="CV RMSE"
                    value={fmt(cv.rmse)}
                    note={cv.fold_rmse_mean != null ? `Per fold ${fmt(cv.fold_rmse_mean)} ± ${fmt(cv.fold_rmse_std)}` : 'Pooled over every left-out row'}
                  />
                </>
              )}
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
