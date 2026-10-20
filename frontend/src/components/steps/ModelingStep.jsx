import { useEffect, useState } from 'react'
import { errorDetail, runManualTool } from '../../api/client'
import { CV_LABELS, cvUnit, splitLabel } from '../../utils/splits'

const ALGORITHM_LABELS = { OLS: 'LINEAR (OLS)' }
const SUBSETS = [
  { id: 'train', label: 'Training' },
  { id: 'validation', label: 'Validation' },
  { id: 'test', label: 'Test' },
]

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(4)
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toPrecision(4).replace(/\.?0+$/, '')
}

/** The split as stored with the model, or rebuilt from the row counts of an older model. */
function splitParts(m) {
  if (Array.isArray(m.split) && m.split.length) return m.split
  return [
    { set: 'train', rows: m.n_train },
    ...(m.n_validation ? [{ set: 'validation', rows: m.n_validation }] : []),
    { set: 'test', rows: m.n_test },
  ].filter((p) => Number.isFinite(p.rows))
}

/**
 * Step 5 — Modeling: the split, the training results and, once run, the test results.
 */
export default function ModelingStep({ modelMetrics, datasetVersionId, onToolResult, busy }) {
  if (!modelMetrics) {
    return (
      <div style={{ textAlign: 'center', padding: '30px 20px', color: 'var(--color-mist)' }}>
        <p style={{ fontSize: '1.5rem', marginBottom: '8px' }}>⚙️</p>
        <p style={{ fontSize: '0.875rem' }}>
          {datasetVersionId
            ? 'Pick a model, its parameters and the split above, then press Split & train. Test the model afterwards on the held-out rows. Or ask the Assistant.'
            : 'Upload a dataset first to train and test a model.'}
        </p>
      </div>
    )
  }

  const m = modelMetrics
  const parts = splitParts(m)
  return (
    <div className="animate-rise" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      <div className="model-head">
        <p className="model-kicker">
          {ALGORITHM_LABELS[m.algorithm] ?? m.algorithm}
          {m.target_column ? ` → ${m.target_column}` : ''}
          {m.model_id ? ` · model ${m.model_id.slice(0, 8)}` : ''}
        </p>
        <ol className="model-stages" aria-label="Progress">
          <li className="done">Split</li>
          <li className="done">Train</li>
          <li className={m.tested ? 'done' : 'todo'}>Test</li>
        </ol>
      </div>

      <SplitPanel metrics={m} parts={parts} datasetVersionId={datasetVersionId} />
      <TrainingPanel metrics={m} />
      <TestingPanel metrics={m} datasetVersionId={datasetVersionId} onToolResult={onToolResult} busy={busy} />

      {m.feature_importances && <CoefficientsPanel values={m.feature_importances} />}
    </div>
  )
}

function SplitPanel({ metrics, parts, datasetVersionId }) {
  const [subset, setSubset] = useState(null)
  const [table, setTable] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const total = parts.reduce((sum, p) => sum + (p.rows || 0), 0) || 1
  const hasStats = parts.some((p) => p.target_mean !== undefined)

  useEffect(() => {
    setSubset(null)
    setTable(null)
    setError(null)
  }, [metrics.model_id])

  useEffect(() => {
    if (!subset || !datasetVersionId || !metrics.model_id) return undefined
    let active = true
    setLoading(true)
    setError(null)
    runManualTool('view_data_split', datasetVersionId, { model_id: metrics.model_id, subset, n_rows: 50 })
      .then((res) => { if (active) setTable({ title: res.table_title, rows: res.table_preview || [] }) })
      .catch((err) => { if (active) setError(errorDetail(err, 'Could not load the rows.')) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [subset, datasetVersionId, metrics.model_id])

  return (
    <section className="panel model-panel" aria-label="Data split">
      <p className="model-panel-title">
        Data split · {metrics.split_strategy ? splitLabel(metrics.split_strategy) : '—'} · {total} complete rows
      </p>
      <div className="split-bar" role="img" aria-label={parts.map((p) => `${p.set} ${p.rows} rows`).join(', ')}>
        {parts.map((p) => {
          const meta = SUBSETS.find((s) => s.id === p.set)
          return (
            <div key={p.set} className={`split-seg split-${p.set}`} style={{ flexGrow: p.rows || 0 }}>
              {meta?.label} {p.rows} · {Math.round((100 * (p.rows || 0)) / total)}%
            </div>
          )
        })}
      </div>

      {hasStats && (
        <table className="data-table split-table">
          <thead>
            <tr>
              <th>Set</th><th>Rows</th><th>Share</th>
              <th>Target mean</th><th>Target std</th><th>Target range</th><th>First – last row</th>
            </tr>
          </thead>
          <tbody>
            {parts.map((p) => (
              <tr key={p.set}>
                <td><span className={`split-dot split-${p.set}`} />{SUBSETS.find((s) => s.id === p.set)?.label ?? p.set}</td>
                <td>{p.rows}</td>
                <td>{Math.round(100 * (p.share ?? p.rows / total))}%</td>
                <td>{formatNumber(p.target_mean)}</td>
                <td>{formatNumber(p.target_std)}</td>
                <td>{formatNumber(p.target_min)} – {formatNumber(p.target_max)}</td>
                <td>{p.first_row != null ? `${p.first_row} – ${p.last_row}` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div className="split-tabs" role="tablist" aria-label="Show the rows of">
        <span className="manual-hint">Show the rows:</span>
        {SUBSETS.filter((s) => parts.some((p) => p.set === s.id)).map((s) => (
          <button
            key={s.id}
            type="button"
            role="tab"
            aria-selected={subset === s.id}
            className={`split-tab${subset === s.id ? ' active' : ''}`}
            onClick={() => setSubset((cur) => (cur === s.id ? null : s.id))}
          >
            <span className={`split-dot split-${s.id}`} />{s.label}
          </button>
        ))}
      </div>
      {subset && (
        <div className="split-rows">
          {loading && <p className="manual-hint">Loading rows…</p>}
          {error && <p className="manual-error">{error}</p>}
          {!loading && !error && table && (
            <>
              <p className="manual-hint">{table.title}</p>
              <div className="manual-table-wrap">
                <table className="data-table">
                  <thead><tr>{Object.keys(table.rows[0] || {}).map((c) => <th key={c}>{c}</th>)}</tr></thead>
                  <tbody>
                    {table.rows.map((row, i) => (
                      <tr key={i}>{Object.keys(table.rows[0]).map((c) => <td key={c}>{typeof row[c] === 'number' ? formatNumber(row[c]) : String(row[c] ?? '—')}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  )
}

function TrainingPanel({ metrics: m }) {
  const train = m.train
  return (
    <section className="panel model-panel" aria-label="Training">
      <p className="model-panel-title">1 · Training · fitted on the {m.n_train ?? '?'} training rows</p>
      <div className="model-cards">
        {train ? (
          <>
            <MetricCard label="Train R²" value={formatMetric(train.r2)} accent="signal" sub={`On the ${train.n_rows} training rows`} />
            <MetricCard label="Train RMSE" value={formatMetric(train.rmse)} accent="ember" sub="Fit error on the training rows" />
          </>
        ) : (
          <MetricCard label="Train R²" value="—" accent="mist" sub="Not recorded for this model" />
        )}
        {m.validation && (
          <>
            <MetricCard label="Validation R²" value={formatMetric(m.validation.r2)} accent="signal" sub={`On the ${m.validation.n_rows} validation rows`} />
            <MetricCard label="Validation RMSE" value={formatMetric(m.validation.rmse)} accent="ember" sub="On the validation rows" />
          </>
        )}
      </div>
      {m.cv && (
        <div className="model-cards">
          <MetricCard
            label="CV R²"
            value={formatMetric(m.cv.r2)}
            accent="signal"
            sub={m.cv.fold_r2_mean != null ? `Folds ${formatMetric(m.cv.fold_r2_mean)} ± ${formatMetric(m.cv.fold_r2_std)}` : 'Pooled over every left-out row'}
          />
          <MetricCard
            label="CV RMSE"
            value={formatMetric(m.cv.rmse)}
            accent="ember"
            sub={m.cv.fold_rmse_mean != null ? `Folds ${formatMetric(m.cv.fold_rmse_mean)} ± ${formatMetric(m.cv.fold_rmse_std)}` : 'Pooled over every left-out row'}
          />
          <MetricCard
            label="Cross-validation"
            value={CV_LABELS[m.cv.strategy] ?? m.cv.strategy}
            accent="mist"
            sub={`${m.cv.n_splits} ${cvUnit(m.cv.strategy)} · ${m.cv.n_rows} training rows scored`}
          />
        </div>
      )}
      {m.hyperparameters && Object.keys(m.hyperparameters).length > 0 && (
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {Object.entries(m.hyperparameters).map(([name, value]) => (
            <span key={name} className="badge-mist model-setting">{name} = {String(value)}</span>
          ))}
        </div>
      )}
    </section>
  )
}

function TestingPanel({ metrics: m, datasetVersionId, onToolResult, busy }) {
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const test = m.test || (m.tested ? { r2: m.r2_score, rmse: m.rmse, n_rows: m.n_test } : null)
  const train = m.train

  async function runTest() {
    setRunning(true)
    setError(null)
    try {
      const args = { model_id: m.model_id }
      const result = await runManualTool('evaluate_on_test_set', datasetVersionId, args)
      await onToolResult?.(result, { tool: 'evaluate_on_test_set', label: 'Test the model', args })
    } catch (err) {
      setError(errorDetail(err, 'The test failed.'))
    } finally {
      setRunning(false)
    }
  }

  return (
    <section className="panel model-panel" aria-label="Testing">
      <p className="model-panel-title">2 · Testing · {m.n_test ?? '?'} held-out test rows</p>
      {test ? (
        <>
          <div className="model-cards">
            <MetricCard label="Test R²" value={formatMetric(test.r2)} accent="signal" sub={`On the ${test.n_rows ?? m.n_test} held-out test rows`} />
            <MetricCard label="Test RMSE" value={formatMetric(test.rmse)} accent="ember" sub="Root mean squared error" />
            {test.mae != null && <MetricCard label="Test MAE" value={formatMetric(test.mae)} accent="mist" sub="Mean absolute error" />}
            {test.bias != null && <MetricCard label="Bias" value={formatMetric(test.bias)} accent="mist" sub="Mean of predicted − actual" />}
          </div>
          {train && Number.isFinite(train.r2) && Number.isFinite(test.r2) && train.r2 - test.r2 > 0.1 && (
            <p className="model-warning">
              ⚠ Train R² is {formatMetric(train.r2 - test.r2)} higher than test R²: the model may be overfitting.
              Try fewer components, a larger alpha, more neighbours or fewer features.
            </p>
          )}
        </>
      ) : (
        <div className="model-untested">
          <p>
            Not tested yet. The {m.n_test ?? ''} test rows were set aside at the split and the model has not seen them.
          </p>
          <button type="button" className="btn-signal" onClick={runTest} disabled={running || busy || !datasetVersionId}>
            {running ? 'Testing…' : 'Test the model'}
          </button>
          {error && <p className="manual-error">{error}</p>}
        </div>
      )}
    </section>
  )
}

function CoefficientsPanel({ values }) {
  const maxVal = Math.max(...Object.values(values).map(Math.abs)) || 1
  return (
    <section className="panel model-panel" aria-label="Feature coefficients">
      <p className="model-panel-title">Feature coefficients / importances</p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {Object.entries(values)
          .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
          .map(([name, value]) => (
            <div key={name} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-foam)', width: '120px', flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {name}
              </span>
              <div style={{ flex: 1, height: '6px', background: 'var(--color-line)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ width: `${Math.abs(value / maxVal) * 100}%`, height: '100%', background: 'var(--color-ember)', borderRadius: '3px', transition: 'width 600ms ease' }} />
              </div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-ember)', width: '60px', textAlign: 'right', flexShrink: 0 }}>
                {formatMetric(value)}
              </span>
            </div>
          ))}
      </div>
    </section>
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
