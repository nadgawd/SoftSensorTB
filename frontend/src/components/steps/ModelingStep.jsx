import ActionChips from '../ActionChips'

/**
 * Step 5 — Modeling
 */
export default function ModelingStep({ modelMetrics, datasetVersionId, actions, tips, onAction }) {
  const hasDataset = Boolean(datasetVersionId)

  function formatMetric(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
    return Number(value).toFixed(4)
  }

  return (
    <div className="animate-rise" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Dynamic tips + clickable actions */}
      <ActionChips actions={actions} tips={tips} onAction={onAction} />

      {/* Algorithm display */}
      <div className="panel" style={{ borderRadius: '14px', padding: '16px 18px' }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: '0 0 10px' }}>
          Model Type
        </p>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          {['OLS', 'RIDGE', 'LASSO', 'PCR', 'PLS', 'KNN'].map((alg) => (
            <div
              key={alg}
              style={{
                padding: '7px 20px',
                borderRadius: '99px',
                border: `1px solid ${modelMetrics?.algorithm === alg ? 'var(--color-ember)' : 'var(--color-line-2)'}`,
                background: modelMetrics?.algorithm === alg ? 'rgba(212,165,53,0.12)' : 'transparent',
                color: modelMetrics?.algorithm === alg ? 'var(--color-ember)' : 'var(--color-mist)',
                fontSize: '0.875rem',
                fontWeight: 500,
              }}
            >
              {alg}
            </div>
          ))}
        </div>
        {!hasDataset && (
          <p style={{ margin: '10px 0 0', fontSize: '12px', color: 'var(--color-alert)' }}>
            ⚠ Upload a dataset first to enable model training.
          </p>
        )}
      </div>

      {/* Metrics scorecard */}
      {modelMetrics ? (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <MetricCard label="R²" value={formatMetric(modelMetrics.r2_score)} accent="signal" sub="Coefficient of determination" />
            <MetricCard label="RMSE" value={formatMetric(modelMetrics.rmse)} accent="ember" sub="Root mean squared error" />
            {modelMetrics.algorithm && (
              <MetricCard label="Algorithm" value={modelMetrics.algorithm} accent="mist" sub={`${modelMetrics.n_train ?? '?'} train · ${modelMetrics.n_test ?? '?'} test`} />
            )}
          </div>

          {/* Feature importances */}
          {modelMetrics.feature_importances && (
            <div className="panel" style={{ borderRadius: '14px', padding: '14px 18px' }}>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: '0 0 12px' }}>
                Feature Coefficients / Importances
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {Object.entries(modelMetrics.feature_importances)
                  .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                  .map(([name, value]) => {
                    const maxVal = Math.max(...Object.values(modelMetrics.feature_importances).map(Math.abs))
                    const pct = Math.abs(value / maxVal) * 100
                    return (
                      <div key={name} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-foam)', width: '120px', flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {name}
                        </span>
                        <div style={{ flex: 1, height: '6px', background: 'var(--color-line)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${pct}%`, height: '100%', background: 'var(--color-ember)', borderRadius: '3px', transition: 'width 600ms ease' }} />
                        </div>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-ember)', width: '60px', textAlign: 'right', flexShrink: 0 }}>
                          {formatMetric(value)}
                        </span>
                      </div>
                    )
                  })}
              </div>
            </div>
          )}

          {modelMetrics.model_id && (
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-mist)' }}>
              model_id · {modelMetrics.model_id}
            </p>
          )}
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: '30px 20px', color: 'var(--color-mist)' }}>
          <p style={{ fontSize: '1.5rem', marginBottom: '8px' }}>⚙️</p>
          <p style={{ fontSize: '0.875rem' }}>Click a suggested action above to train a soft-sensor model.</p>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value, accent, sub }) {
  const colors = {
    signal: { bg: 'rgba(46,196,182,0.08)', border: 'rgba(46,196,182,0.25)', text: 'var(--color-signal)' },
    ember:  { bg: 'rgba(212,165,53,0.08)',  border: 'rgba(212,165,53,0.25)',  text: 'var(--color-ember)' },
    mist:   { bg: 'rgba(138,164,184,0.08)', border: 'rgba(138,164,184,0.2)',  text: 'var(--color-mist)' },
  }
  const c = colors[accent] || colors.mist

  return (
    <div style={{ flex: '1 1 140px', padding: '14px 18px', borderRadius: '14px', background: c.bg, border: `1px solid ${c.border}` }}>
      <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: c.text, margin: '0 0 6px', opacity: 0.8 }}>
        {label}
      </p>
      <p style={{ fontFamily: 'var(--font-display)', fontSize: '1.6rem', fontWeight: 700, color: c.text, margin: 0, lineHeight: 1 }}>
        {value}
      </p>
      {sub && <p style={{ fontSize: '11px', color: 'var(--color-mist)', margin: '6px 0 0' }}>{sub}</p>}
    </div>
  )
}
