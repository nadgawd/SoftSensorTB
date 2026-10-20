import { useCallback, useEffect, useState } from 'react'
import { ChartScatter, CircleCheck, FlaskConical } from 'lucide-react'
import { errorDetail, fetchVersionModels } from '../../api/client'
import Elapsed from '../Elapsed'

const MODEL_SETTINGS = ['n_components', 'alpha', 'n_neighbors', 'knn_weights', 'pls_scale', 'max_iter', 'fit_intercept']

function fmt(v) {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : '—'
}

function settings(m) {
  const h = m.hyperparameters || {}
  return MODEL_SETTINGS.filter((k) => h[k] !== undefined && h[k] !== null).map((k) => `${k} ${h[k]}`)
}

/** The highest value of `get` over the models, to mark the best score in each column. */
function best(models, get) {
  const values = models.map(get).filter((v) => typeof v === 'number' && Number.isFinite(v))
  return values.length > 1 ? Math.max(...values) : null
}

function Score({ r2, rmse, isBest }) {
  return (
    <td className={`compare-score${isBest ? ' best' : ''}`}>
      <span>{fmt(r2)}</span>
      <span className="compare-rmse">{fmt(rmse)}</span>
    </td>
  )
}

/** Model step › Compare models: every model trained on this data, side by side. */
export default function CompareModelsTab({ datasetVersionId, modelMetrics, busy, onRunTool, onActivate }) {
  const [models, setModels] = useState(null)
  const [error, setError] = useState(null)
  const [working, setWorking] = useState(null)

  const load = useCallback(() => {
    if (!datasetVersionId) return
    fetchVersionModels(datasetVersionId)
      .then((list) => { setModels(list); setError(null) })
      .catch((err) => setError(errorDetail(err, 'Could not load the models.')))
  }, [datasetVersionId])

  useEffect(load, [load, modelMetrics?.model_id, modelMetrics?.r2_score])

  async function act(key, fn) {
    setWorking(key)
    setError(null)
    try {
      await fn()
      load()
    } catch (err) {
      setError(errorDetail(err, 'That did not work.'))
    } finally {
      setWorking(null)
    }
  }

  if (error && !models) return <p className="manual-error">{error}</p>
  if (!models) return <p className="manual-hint">Loading the models…</p>
  if (!models.length) {
    return <p className="manual-hint" style={{ textAlign: 'center', padding: '20px' }}>No models yet. Train one on the Train &amp; test tab.</p>
  }

  const activeId = modelMetrics?.model_id
  const top = {
    train: best(models, (m) => m.train?.r2),
    validation: best(models, (m) => m.validation?.r2),
    cv: best(models, (m) => m.cv?.r2),
    test: best(models, (m) => m.r2_score),
  }
  const hasValidation = models.some((m) => m.validation)
  const hasCv = models.some((m) => m.cv)

  return (
    <section className="panel model-panel" aria-label="Compare models">
      <p className="model-panel-title">
        {models.length} model{models.length === 1 ? '' : 's'} on this data · R² above, RMSE below · the best score in each column is highlighted
      </p>
      <div className="manual-table-wrap">
        <table className="data-table compare-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Settings</th>
              <th>Train</th>
              {hasValidation && <th>Validation</th>}
              {hasCv && <th>CV</th>}
              <th>Test</th>
              <th aria-label="Actions" />
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const active = m.model_id === activeId
              const disabled = busy || Boolean(working)
              return (
                <tr key={m.model_id} className={active ? 'compare-active' : undefined}>
                  <td>
                    <div className="compare-model">
                      <strong>{m.algorithm}</strong> → {m.target_column}
                    </div>
                    <div className="compare-meta">
                      {String(m.model_id).slice(0, 8)} · {m.feature_columns?.length ?? '?'} inputs
                      {m.trained_at && <> · {new Date(m.trained_at).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })}</>}
                    </div>
                  </td>
                  <td className="compare-settings">{settings(m).join(' · ') || '—'}</td>
                  <Score r2={m.train?.r2} rmse={m.train?.rmse} isBest={m.train?.r2 === top.train} />
                  {hasValidation && <Score r2={m.validation?.r2} rmse={m.validation?.rmse} isBest={m.validation?.r2 === top.validation} />}
                  {hasCv && <Score r2={m.cv?.r2} rmse={m.cv?.rmse} isBest={m.cv?.r2 === top.cv} />}
                  {m.r2_score == null ? (
                    <td>
                      <button type="button" className="compare-btn" disabled={disabled}
                        onClick={() => act(`test-${m.model_id}`, () => onRunTool('evaluate_on_test_set', { model_id: m.model_id }))}>
                        <FlaskConical size={12} aria-hidden="true" />
                        {working === `test-${m.model_id}` ? <>Testing… <Elapsed /></> : 'Test'}
                      </button>
                    </td>
                  ) : (
                    <Score r2={m.r2_score} rmse={m.rmse} isBest={m.r2_score === top.test} />
                  )}
                  <td className="compare-actions">
                    {active ? (
                      <span className="compare-active-badge"><CircleCheck size={12} aria-hidden="true" /> Active</span>
                    ) : (
                      <button type="button" className="compare-btn" disabled={disabled}
                        title="Show this model on Train & test and use it in the diagnostic and export tools"
                        onClick={() => act(`use-${m.model_id}`, () => onActivate(m.model_id))}>
                        {working === `use-${m.model_id}` ? 'Loading…' : 'Make active'}
                      </button>
                    )}
                    <button type="button" className="compare-btn" disabled={disabled}
                      title="Parity plot of this model on its test rows (opens Diagnostics)"
                      onClick={() => act(`parity-${m.model_id}`, () => onRunTool('generate_parity_plot', { model_id: m.model_id }))}>
                      <ChartScatter size={12} aria-hidden="true" />
                      {working === `parity-${m.model_id}` ? <>Plotting… <Elapsed /></> : 'Parity'}
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {error && <p className="manual-error">{error}</p>}
      <p className="manual-hint">
        Parity plots of several models stay side by side in the Diagnostics tab’s plot history, so you can flip between them.
      </p>
    </section>
  )
}
