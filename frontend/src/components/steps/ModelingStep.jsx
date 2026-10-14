import { CV_LABELS, cvUnit, splitLabel } from '../../utils/splits'

const ALGORITHM_LABELS = { OLS: 'LINEAR (OLS)' }

/**
 * Step 5 — Modeling
 */
export default function ModelingStep({ modelMetrics, datasetVersionId }) {
  function formatMetric(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
    return Number(value).toFixed(4)
  }

  const train = modelMetrics?.train

  return (
    <div className="animate-rise" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {modelMetrics ? (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: 0 }}>
            Results · {ALGORITHM_LABELS[modelMetrics.algorithm] ?? modelMetrics.algorithm}
            {modelMetrics.target_column ? ` → ${modelMetrics.target_column}` : ''}
          </p>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {train && (
              <MetricCard label="Train R²" value={formatMetric(train.r2)} accent="mist" sub={`On the ${train.n_rows} training rows`} />
            )}
            <MetricCard label="Test R²" value={formatMetric(modelMetrics.r2_score)} accent="signal" sub={`On the ${modelMetrics.n_test ?? '?'} held-out test rows`} />
            {train && (
              <MetricCard label="Train RMSE" value={formatMetric(train.rmse)} accent="mist" sub="Fit error on the training rows" />
            )}
            <MetricCard label="Test RMSE" value={formatMetric(modelMetrics.rmse)} accent="ember" sub="Root mean squared error" />
          </div>
          {train && Number.isFinite(train.r2) && Number.isFinite(modelMetrics.r2_score) && train.r2 - modelMetrics.r2_score > 0.1 && (
            <p style={{ margin: 0, fontSize: '12px', color: 'var(--color-alert)' }}>
              ⚠ Train R² is {formatMetric(train.r2 - modelMetrics.r2_score)} higher than test R²: the model may be overfitting.
              Try fewer components, a larger alpha, more neighbours or fewer features.
            </p>
          )}
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <MetricCard
              label="Split"
              value={modelMetrics.split_strategy ? splitLabel(modelMetrics.split_strategy) : '—'}
              accent="mist"
              sub={`${modelMetrics.n_train ?? '?'} train${modelMetrics.validation ? ` · ${modelMetrics.validation.n_rows} validation` : ''} · ${modelMetrics.n_test ?? '?'} test`}
            />
            {modelMetrics.validation && (
              <>
                <MetricCard label="Validation R²" value={formatMetric(modelMetrics.validation.r2)} accent="signal" sub="On the separate validation rows" />
                <MetricCard label="Validation RMSE" value={formatMetric(modelMetrics.validation.rmse)} accent="ember" sub="On the separate validation rows" />
              </>
            )}
          </div>

          {modelMetrics.cv && (
            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
              <MetricCard
                label="CV R²"
                value={formatMetric(modelMetrics.cv.r2)}
                accent="signal"
                sub={modelMetrics.cv.fold_r2_mean != null
                  ? `Folds ${formatMetric(modelMetrics.cv.fold_r2_mean)} ± ${formatMetric(modelMetrics.cv.fold_r2_std)}`
                  : 'Pooled over every left-out row'}
              />
              <MetricCard
                label="CV RMSE"
                value={formatMetric(modelMetrics.cv.rmse)}
                accent="ember"
                sub={modelMetrics.cv.fold_rmse_mean != null
                  ? `Folds ${formatMetric(modelMetrics.cv.fold_rmse_mean)} ± ${formatMetric(modelMetrics.cv.fold_rmse_std)}`
                  : 'Pooled over every left-out row'}
              />
              <MetricCard
                label="Cross-validation"
                value={CV_LABELS[modelMetrics.cv.strategy] ?? modelMetrics.cv.strategy}
                accent="mist"
                sub={`${modelMetrics.cv.n_splits} ${cvUnit(modelMetrics.cv.strategy)} · ${modelMetrics.cv.n_rows} training rows scored`}
              />
            </div>
          )}

          {modelMetrics.hyperparameters && Object.keys(modelMetrics.hyperparameters).length > 0 && (
            <div className="panel" style={{ borderRadius: '14px', padding: '12px 18px' }}>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: '0 0 8px' }}>
                Settings Used
              </p>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {Object.entries(modelMetrics.hyperparameters).map(([name, value]) => (
                  <span key={name} className="badge-mist" style={{ padding: '3px 9px', borderRadius: '99px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
                    {name} = {String(value)}
                  </span>
                ))}
              </div>
            </div>
          )}

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
          <p style={{ fontSize: '0.875rem' }}>
            {datasetVersionId
              ? 'Choose a model and its parameters above, set the train/test split, then press Train & test. Or ask the Assistant.'
              : 'Upload a dataset first to train and test a model.'}
          </p>
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
