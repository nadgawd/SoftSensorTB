import { useState } from 'react'
import { Pause, Play, Trash2 } from 'lucide-react'
import { deleteDeployment, errorDetail, updateDeployment } from '../../api/client'

/** Rename, pause or resume, tune the bias correction, or delete a deployment. */
export default function SettingsPanel({ deployment, onChanged, onDeleted }) {
  const [name, setName] = useState(deployment.name)
  const [gain, setGain] = useState(deployment.bias_gain)
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)

  async function act(key, fn) {
    setBusy(key)
    setError(null)
    try {
      await fn()
    } catch (err) {
      setError(errorDetail(err, 'That did not work.'))
    } finally {
      setBusy(null)
    }
  }

  const paused = deployment.status === 'paused'

  return (
    <div className="deploy-settings">
      <div className="deploy-form">
        <label className="manual-label" htmlFor="dep-name">Name</label>
        <input id="dep-name" className="manual-input" value={name} maxLength={200} onChange={(e) => setName(e.target.value)} />
        <label className="manual-label" htmlFor="dep-gain">
          Bias gain <span className="manual-hint">0 = ignore lab results, 1 = jump to the latest error</span>
        </label>
        <input id="dep-gain" className="manual-input" type="number" min={0} max={1} step={0.05} value={gain}
          onChange={(e) => setGain(Math.min(1, Math.max(0, Number(e.target.value) || 0)))} />
        <div className="manual-actions">
          <button type="button" className="btn-signal" disabled={Boolean(busy) || !name.trim()}
            onClick={() => act('save', async () => onChanged(await updateDeployment(deployment.id, { name: name.trim(), bias_gain: gain })))}>
            {busy === 'save' ? 'Saving…' : 'Save'}
          </button>
          <button type="button" className="btn-ghost" disabled={Boolean(busy)}
            title="Start again from no correction, for example after the analyser or the lab method changed"
            onClick={() => act('reset', async () => onChanged(await updateDeployment(deployment.id, { reset_bias: true })))}>
            Reset bias to 0
          </button>
        </div>
      </div>

      <div className="export-row">
        <div>
          <p className="export-name">{paused ? 'Paused' : 'Active'}</p>
          <p className="export-desc">
            {paused
              ? 'Rows sent now are refused (HTTP 409), so a connector keeps them and sends them once you resume.'
              : 'Pausing refuses new rows (a connector keeps them and retries), for example during a planned shutdown or maintenance.'}
          </p>
        </div>
        <div className="export-actions">
          <button type="button" className="btn-ghost export-btn" disabled={Boolean(busy)}
            onClick={() => act('status', async () => onChanged(await updateDeployment(deployment.id, { status: paused ? 'active' : 'paused' })))}>
            {paused ? <><Play size={13} aria-hidden="true" /> Resume</> : <><Pause size={13} aria-hidden="true" /> Pause</>}
          </button>
        </div>
      </div>

      <div className="export-row">
        <div>
          <p className="export-name">Delete</p>
          <p className="export-desc">Removes the deployment, its key and its stored predictions. The model and the dataset stay.</p>
        </div>
        <div className="export-actions">
          <button type="button" className="btn-ghost export-btn deploy-danger" disabled={Boolean(busy)}
            onClick={() => {
              if (!window.confirm(`Delete “${deployment.name}”? Connectors using its key will stop working.`)) return
              act('delete', async () => { await deleteDeployment(deployment.id); onDeleted() })
            }}>
            <Trash2 size={13} aria-hidden="true" /> {busy === 'delete' ? 'Deleting…' : 'Delete'}
          </button>
        </div>
      </div>
      {error && <p className="manual-error">{error}</p>}
    </div>
  )
}
