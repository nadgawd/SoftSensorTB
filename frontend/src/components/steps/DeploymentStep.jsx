import { useCallback, useEffect, useState } from 'react'
import { KeyRound, Plus, Radio } from 'lucide-react'
import { errorDetail, fetchDeployments } from '../../api/client'
import NewDeployment from '../deployment/NewDeployment'
import MonitorPanel, { StatusChip } from '../deployment/MonitorPanel'
import ConnectPanel from '../deployment/ConnectPanel'
import SimulatePanel from '../deployment/SimulatePanel'
import SettingsPanel from '../deployment/SettingsPanel'

const TABS = [
  { id: 'monitor', label: 'Monitor' },
  { id: 'connect', label: 'Connect' },
  { id: 'simulate', label: 'Simulate' },
  { id: 'settings', label: 'Settings' },
]

function KeyReveal({ apiKey, onDone }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="deploy-key" role="alert">
      <p><KeyRound size={15} aria-hidden="true" /> <strong>API key — copy it now, it is not shown again.</strong></p>
      <code>{apiKey}</code>
      <div className="manual-actions">
        <button type="button" className="btn-signal" onClick={() => {
          navigator.clipboard?.writeText(apiKey).then(() => setCopied(true))
        }}>{copied ? 'Copied' : 'Copy key'}</button>
        <button type="button" className="btn-ghost" onClick={onDone}>I have stored it</button>
      </div>
      <p className="manual-hint">
        Give it to the plant connector as the <code>SOFTSENSOR_API_KEY</code> environment variable. Anyone with the key can send data to
        this deployment and read its predictions; if it leaks, make a new one under Connect.
      </p>
    </div>
  )
}

/**
 * Deployment step: put a trained model on live data. A plant-side connector streams the inputs in,
 * every training step is replayed on them, and each prediction comes back checked and bias-corrected.
 */
export default function DeploymentStep({ datasetVersionId, modelMetrics }) {
  const [list, setList] = useState(null)
  const [limit, setLimit] = useState(0)
  const [selected, setSelected] = useState(null)
  const [tab, setTab] = useState('monitor')
  const [newKey, setNewKey] = useState(null)
  const [error, setError] = useState(null)
  const [tick, setTick] = useState(0)

  const load = useCallback(async (select) => {
    try {
      const data = await fetchDeployments()
      if (!Array.isArray(data?.deployments)) throw new Error('The server did not return the deployment list.')
      setList(data.deployments)
      setLimit(data.max_deployments || 0)
      setError(null)
      setSelected((current) => {
        const wanted = select !== undefined ? select : current
        if (wanted === 'new') return 'new'
        if (wanted && data.deployments.some((d) => d.id === wanted)) return wanted
        return data.deployments[0]?.id || 'new'
      })
    } catch (err) {
      setError(errorDetail(err, 'Could not load the deployments.'))
      setList((current) => current || [])
    }
  }, [])

  useEffect(() => {
    load()
    const timer = setInterval(() => load(), 15_000)
    return () => clearInterval(timer)
  }, [load])

  const deployment = list?.find((d) => d.id === selected)
  const atLimit = limit > 0 && (list?.length || 0) >= limit

  function replace(updated) {
    setList((current) => current.map((d) => (d.id === updated.id ? { ...d, ...updated } : d)))
  }

  return (
    <div className="deploy-step animate-rise">
      <section className="panel model-panel" aria-label="Deployments">
        <div className="model-head">
          <p className="model-panel-title">
            Live deployments{list ? ` · ${list.length}${limit ? ` of ${limit}` : ''}` : ''}
          </p>
          <button type="button" className="compare-btn" onClick={() => setSelected('new')} disabled={selected === 'new'}>
            <Plus size={12} aria-hidden="true" /> New deployment
          </button>
        </div>
        {!list && <p className="manual-hint">Loading…</p>}
        {list && !list.length && (
          <p className="manual-hint">
            None yet. Deploying a model lets the plant send live readings and get the soft-sensor prediction back, checked against the
            conditions it was trained on and corrected with lab results.
          </p>
        )}
        {list && list.length > 0 && (
          <ul className="deploy-list">
            {list.map((d) => (
              <li key={d.id}>
                <button type="button" className={`deploy-item${d.id === selected ? ' active' : ''}`} onClick={() => { setSelected(d.id); setTab('monitor') }}>
                  <Radio size={14} aria-hidden="true" className={d.status === 'active' ? 'deploy-ok' : 'deploy-muted'} />
                  <span className="deploy-item-name">{d.name}</span>
                  <span className="deploy-meta">
                    {d.model?.algorithm} → {d.model?.target_column} · {d.dataset_name}
                    {' · '}{d.last_received ? `last data ${new Date(d.last_received).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })}` : 'no data yet'}
                  </span>
                  <StatusChip status={d.status} />
                </button>
              </li>
            ))}
          </ul>
        )}
        {error && <p className="manual-error">{error}</p>}
      </section>

      {newKey && <KeyReveal apiKey={newKey} onDone={() => setNewKey(null)} />}

      {selected === 'new' && (
        <NewDeployment
          datasetVersionId={datasetVersionId}
          activeModelId={modelMetrics?.model_id}
          atLimit={atLimit}
          onCreated={({ deployment: created, api_key: key }) => {
            setNewKey(key)
            setTab('connect')
            load(created.id)
          }}
        />
      )}

      {deployment && (
        <section className="panel model-panel" aria-label={deployment.name}>
          <div className="model-head">
            <p className="model-panel-title">{deployment.name}</p>
            <span className="deploy-meta">
              {deployment.model?.algorithm} → {deployment.model?.target_column} · model {String(deployment.model_id).slice(0, 8)} ·
              {' '}{(deployment.inputs || []).length} live inputs
            </span>
          </div>
          <div className="model-tabs" role="tablist" aria-label="Deployment">
            {TABS.map((t) => (
              <button key={t.id} type="button" role="tab" aria-selected={tab === t.id}
                className={`model-tab${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
                {t.label}
              </button>
            ))}
          </div>
          {tab === 'monitor' && <MonitorPanel key={deployment.id} deploymentId={deployment.id} refreshKey={tick} />}
          {tab === 'connect' && <ConnectPanel key={deployment.id} deployment={deployment} onKey={setNewKey} />}
          {/* Stays mounted on the other tabs so a running simulation keeps feeding the Monitor tab. */}
          <div hidden={tab !== 'simulate'}>
            <SimulatePanel key={deployment.id} deployment={deployment} onTick={() => setTick((t) => t + 1)} />
          </div>
          {tab === 'settings' && (
            <SettingsPanel key={deployment.id} deployment={deployment} onChanged={replace} onDeleted={() => load(null)} />
          )}
        </section>
      )}
    </div>
  )
}
