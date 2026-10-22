import { useEffect, useState } from 'react'
import { CircleAlert, CircleCheck, CircleMinus, Rocket } from 'lucide-react'
import { createDeployment, errorDetail, fetchDeploymentPlan, fetchVersionModels } from '../../api/client'
import Elapsed from '../Elapsed'

/** How each lineage step runs on live rows. */
const MODE_LABELS = {
  exact: ['Runs as is', 'ok'],
  fitted: ['Uses values learnt in training', 'ok'],
  window: ['Uses recent rows', 'ok'],
  approx: ['Approximated live', 'warn'],
  training_only: ['Training data only', 'muted'],
  upstream: ['Before the live feed', 'muted'],
  not_needed: ['Not needed', 'muted'],
  no_change: ['Changed nothing', 'muted'],
  unsupported: ['Cannot run live', 'alert'],
}

const CHECKS = [
  ['training_rows', 'Every step on the training data reproduces the model’s inputs'],
  ['live_columns', 'Only the live columns, only the kept steps, give the same inputs'],
  ['streaming', 'Rows sent a few at a time, with the history buffer, give the same inputs'],
]

function dtypeLabel(dtype) {
  if (!dtype) return '?'
  if (/^(float|int|uint|Float|Int|UInt)/.test(dtype)) return 'number'
  if (dtype.startsWith('datetime')) return 'time'
  if (dtype === 'bool' || dtype === 'boolean') return 'true / false'
  return 'text'
}

function modelLabel(m) {
  const score = m.r2_score != null ? `test R² ${Number(m.r2_score).toFixed(3)}`
    : m.cv?.r2 != null ? `CV R² ${Number(m.cv.r2).toFixed(3)}` : 'not tested'
  return `${m.algorithm} → ${m.target_column} · ${m.feature_columns?.length ?? 0} inputs · ${score} · ${m.model_id.slice(0, 8)}`
}

function CheckIcon({ ok }) {
  if (ok === true) return <CircleCheck size={14} className="deploy-ok" aria-label="passed" />
  if (ok === false) return <CircleAlert size={14} className="deploy-alert" aria-label="failed" />
  return <CircleMinus size={14} className="deploy-muted" aria-label="skipped" />
}

/** Pick a model, see exactly how it will run on live rows, and deploy it. */
export default function NewDeployment({ datasetVersionId, activeModelId, onCreated, atLimit }) {
  const [models, setModels] = useState(null)
  const [modelId, setModelId] = useState('')
  const [plan, setPlan] = useState(null)
  const [planError, setPlanError] = useState(null)
  const [loadingPlan, setLoadingPlan] = useState(false)
  const [name, setName] = useState('')
  const [gain, setGain] = useState(0.3)
  const [deploying, setDeploying] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!datasetVersionId) return undefined
    let cancelled = false
    fetchVersionModels(datasetVersionId)
      .then((list) => {
        if (cancelled) return
        setModels(list)
        setModelId((current) => (list.some((m) => m.model_id === current) ? current
          : list.find((m) => m.model_id === activeModelId)?.model_id || list[0]?.model_id || ''))
      })
      .catch((err) => { if (!cancelled) { setModels([]); setPlanError(errorDetail(err, 'Could not load the models.')) } })
    return () => { cancelled = true }
  }, [datasetVersionId, activeModelId])

  useEffect(() => {
    if (!datasetVersionId || !modelId) return undefined
    let cancelled = false
    setPlan(null)
    setPlanError(null)
    setLoadingPlan(true)
    fetchDeploymentPlan(datasetVersionId, modelId)
      .then((p) => { if (!cancelled) setPlan(p) })
      .catch((err) => { if (!cancelled) setPlanError(errorDetail(err, 'Could not check this model.')) })
      .finally(() => { if (!cancelled) setLoadingPlan(false) })
    return () => { cancelled = true }
  }, [datasetVersionId, modelId])

  async function deploy() {
    setDeploying(true)
    setError(null)
    try {
      const created = await createDeployment({ datasetVersionId, modelId, name: name.trim(), biasGain: gain })
      onCreated(created)
    } catch (err) {
      setError(errorDetail(err, 'Could not deploy this model.'))
    } finally {
      setDeploying(false)
    }
  }

  if (!datasetVersionId) {
    return <section className="panel model-panel"><p className="manual-hint">Upload a dataset and train a model to deploy it.</p></section>
  }
  if (models && !models.length) {
    return (
      <section className="panel model-panel">
        <p className="model-panel-title">New deployment</p>
        <p className="manual-hint">No model trained on this dataset yet. Train one in the Model step, then come back here to deploy it.</p>
      </section>
    )
  }

  const keptSteps = plan?.steps.filter((s) => s.kept) || []
  const otherSteps = plan?.steps.filter((s) => !s.kept) || []
  const inputs = plan?.inputs || []

  return (
    <section className="panel model-panel" aria-label="New deployment">
      <p className="model-panel-title">New deployment</p>
      <label className="manual-label" htmlFor="deploy-model">Model</label>
      <select id="deploy-model" className="manual-input" value={modelId} onChange={(e) => setModelId(e.target.value)} disabled={!models}>
        {!models && <option>Loading the models…</option>}
        {models?.map((m) => <option key={m.model_id} value={m.model_id}>{modelLabel(m)}</option>)}
      </select>

      {loadingPlan && <p className="manual-hint">Rebuilding every step and checking it against the stored data… <Elapsed /></p>}
      {planError && <p className="manual-error">{planError}</p>}

      {plan && (
        <>
          {plan.ready ? (
            <p className="deploy-banner ok"><CircleCheck size={15} aria-hidden="true" />
              Ready: on live rows this model gets exactly the inputs it was trained on.
            </p>
          ) : (
            <div className="deploy-banner alert">
              <p><CircleAlert size={15} aria-hidden="true" /> This model cannot be deployed yet:</p>
              <ul>{plan.blocking.map((b) => <li key={b}>{b}</li>)}</ul>
              <p className="manual-hint">Train a model on inputs that do not depend on these steps, or redo the steps with live-friendly options (for example a rolling window instead of an expanding one).</p>
            </div>
          )}

          <div className="deploy-grid">
            <div>
              <p className="deploy-subtitle">The live feed sends, for each time stamp</p>
              <table className="data-table deploy-table">
                <thead><tr><th>Column</th><th>Type</th></tr></thead>
                <tbody>
                  {inputs.map((i) => (
                    <tr key={i.name}><td><code>{i.name}</code>{i.is_time && <span className="deploy-tag">time</span>}</td><td>{dtypeLabel(i.dtype)}</td></tr>
                  ))}
                </tbody>
              </table>
              <p className="manual-hint">
                {plan.cut.index > 0
                  ? <>In the form of the data after step {plan.cut.index} ({plan.cut.title}): the plant system must already do that step. </>
                  : <>As in the uploaded file. </>}
                {plan.buffer_rows > 0
                  ? <>The deployment keeps the last {plan.buffer_rows} rows for lags and windows, so the first predictions wait for that history.</>
                  : <>No history is needed: each row is predicted on its own.</>}
              </p>
            </div>
            <div>
              <p className="deploy-subtitle">Checks</p>
              <ul className="deploy-checks">
                {CHECKS.map(([key, label]) => {
                  const c = plan.checks?.[key]
                  return (
                    <li key={key}>
                      <CheckIcon ok={c?.ok} />
                      <span>{label}{c?.rows ? ` (${c.rows} rows)` : ''}{c?.detail ? <span className="manual-hint"> — {c.detail}</span> : null}</span>
                    </li>
                  )
                })}
              </ul>
              {plan.warnings.length > 0 && (
                <>
                  <p className="deploy-subtitle">Worth knowing</p>
                  <ul className="deploy-warnings">{plan.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
                </>
              )}
            </div>
          </div>

          <div>
            <p className="deploy-subtitle">Steps run on every live row ({keptSteps.length})</p>
            <StepTable steps={keptSteps} />
            {otherSteps.length > 0 && (
              <details className="deploy-details">
                <summary>Steps not run live ({otherSteps.length})</summary>
                <StepTable steps={otherSteps} />
              </details>
            )}
          </div>

          <div className="deploy-form">
            <label className="manual-label" htmlFor="deploy-name">Name</label>
            <input id="deploy-name" className="manual-input" placeholder={`${plan.model.algorithm} → ${plan.model.target_column}`}
              value={name} maxLength={200} onChange={(e) => setName(e.target.value)} />
            <label className="manual-label" htmlFor="deploy-gain">
              Bias gain <span className="manual-hint">how strongly each lab result corrects the prediction (0 = never, 1 = fully)</span>
            </label>
            <input id="deploy-gain" className="manual-input" type="number" min={0} max={1} step={0.05} value={gain}
              onChange={(e) => setGain(Math.min(1, Math.max(0, Number(e.target.value) || 0)))} />
            <div className="manual-actions">
              <button type="button" className="btn-signal" disabled={!plan.ready || deploying || atLimit} onClick={deploy}>
                <Rocket size={14} aria-hidden="true" /> {deploying ? <>Deploying… <Elapsed /></> : 'Deploy'}
              </button>
              {atLimit && <span className="manual-hint">You have reached the deployment limit; delete one first.</span>}
            </div>
            {error && <p className="manual-error">{error}</p>}
          </div>
        </>
      )}
    </section>
  )
}

function StepTable({ steps }) {
  if (!steps.length) return <p className="manual-hint">None.</p>
  return (
    <div className="manual-table-wrap">
      <table className="data-table deploy-table">
        <thead><tr><th>#</th><th>Step</th><th>Live</th></tr></thead>
        <tbody>
          {steps.map((s) => {
            const [label, tone] = MODE_LABELS[s.mode] || [s.mode, 'muted']
            const note = [s.note, s.detail].filter(Boolean).join(' ')
            return (
              <tr key={s.index}>
                <td>{s.index}</td>
                <td className="deploy-wrap">
                  <strong>{s.title}</strong>
                  {s.summary && <div className="deploy-meta">{s.summary}</div>}
                  {note && <div className="deploy-note">{note}</div>}
                </td>
                <td>
                  <span className="deploy-live">
                    <span className={`deploy-chip ${tone}`}>{label}</span>
                    {s.verified != null && <CheckIcon ok={s.verified} />}
                  </span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
