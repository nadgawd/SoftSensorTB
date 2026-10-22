import { useState } from 'react'
import { Download, KeyRound } from 'lucide-react'
import { downloadExport, errorDetail, liveApiBase, rotateDeploymentKey } from '../../api/client'

function CopyBlock({ text, label }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="deploy-code">
      <div className="deploy-code-head">
        <span>{label}</span>
        <button type="button" className="compare-btn" onClick={() => {
          navigator.clipboard?.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })
        }}>{copied ? 'Copied' : 'Copy'}</button>
      </div>
      <pre>{text}</pre>
    </div>
  )
}

function exampleRow(deployment) {
  const row = {}
  for (const i of deployment.inputs || []) {
    if (i.is_time) row[i.name] = '2024-05-01T08:00:00Z'
    else if (i.stats?.numeric) row[i.name] = Number((i.stats.typical ?? ((i.stats.low ?? 0) + (i.stats.high ?? 0)) / 2).toPrecision(4))
    else row[i.name] = i.stats?.categories?.[0] ?? 'value'
  }
  return row
}

/** How a plant system sends live data: the endpoint, the key, examples, and the ready-made connector. */
export default function ConnectPanel({ deployment, onKey }) {
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)
  const base = `${liveApiBase()}/live/${deployment.id}`
  const row = exampleRow(deployment)
  const target = deployment.model?.target_column || 'target'

  const curl = `curl -X POST ${base}/predict \\
  -H "X-API-Key: $SOFTSENSOR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify({ rows: [row] })}'`

  const python = `import os, requests

URL = "${base}"
HEADERS = {"X-API-Key": os.environ["SOFTSENSOR_API_KEY"]}

# One object per time stamp; send several at once if you have them (up to 1000).
rows = [${JSON.stringify(row)}]
for p in requests.post(f"{URL}/predict", json={"rows": rows}, headers=HEADERS, timeout=30).json()["predictions"]:
    print(p["time"], p["status"], p["corrected"], p["reasons"])

# When the lab result for a sample arrives (hours later), send it with the sample time:
requests.post(f"{URL}/lab", json={"samples": [{"time": "2024-05-01T08:00:00Z", "value": 12.3}]},
              headers=HEADERS, timeout=30)`

  async function rotate() {
    if (!window.confirm('Create a new API key? The current key stops working immediately, so every connector must be updated.')) return
    setBusy('rotate')
    setError(null)
    try {
      const { api_key: key } = await rotateDeploymentKey(deployment.id)
      onKey(key)
    } catch (err) {
      setError(errorDetail(err, 'Could not rotate the key.'))
    } finally {
      setBusy(null)
    }
  }

  async function downloadConnector() {
    setBusy('connector')
    setError(null)
    try {
      await downloadExport(`/deployments/${deployment.id}/connector`, { fallbackName: 'softsensor_connector.zip' })
    } catch (err) {
      setError(errorDetail(err, 'Download failed.'))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="deploy-connect">
      <div className="deploy-flow" aria-label="Data flow">
        <span>Sensors</span><span>PLC / DCS</span><span>Historian · OPC UA · MQTT</span>
        <span className="deploy-flow-hl">Connector (plant side)</span><span className="deploy-flow-hl">This deployment</span>
        <span>Prediction → back to a tag</span>
      </div>
      <p className="manual-hint">
        A small program next to the plant reads the inputs every cycle (typically every minute) and sends them here over HTTPS.
        Only outbound connections are made, so no firewall port has to be opened toward the plant. The prediction comes back
        in the response and can be written back to a soft-sensor tag the operators can trend.
      </p>

      <div className="export-row">
        <div>
          <p className="export-name">Ready-made connector</p>
          <p className="export-desc">
            A Python script with a config filled in for this deployment. It reads from a CSV export, an OPC UA server, an MQTT broker
            or the PI Web API, keeps rows while the network is down, sends lab results, and writes the prediction back.
            Set the key as <code>SOFTSENSOR_API_KEY</code>, adjust <code>config.json</code>, then run{' '}
            <code>python softsensor_connector.py --check</code> and start it.
          </p>
        </div>
        <div className="export-actions">
          <button type="button" className="btn-ghost export-btn" disabled={busy === 'connector'} onClick={downloadConnector}>
            <Download size={13} aria-hidden="true" /> {busy === 'connector' ? 'Preparing…' : 'Download'}
          </button>
        </div>
      </div>

      <div className="export-row">
        <div>
          <p className="export-name">API key</p>
          <p className="export-desc">
            <code>{deployment.api_key_prefix}…</code> — shown in full only when it was created. Keep it on the plant server
            (an environment variable), never in code. If it leaks, make a new one: the old one stops working at once.
          </p>
        </div>
        <div className="export-actions">
          <button type="button" className="btn-ghost export-btn" disabled={busy === 'rotate'} onClick={rotate}>
            <KeyRound size={13} aria-hidden="true" /> {busy === 'rotate' ? 'Creating…' : 'New key'}
          </button>
        </div>
      </div>

      <div>
        <p className="deploy-subtitle">Endpoints</p>
        <ul className="deploy-endpoints">
          <li><code>POST {base}/predict</code><span>Readings in, predictions out: <code>{'{"rows": [{…}, …]}'}</code>, at most 1000 rows.</span></li>
          <li><code>POST {base}/lab</code><span>Lab results of {target}: <code>{'{"samples": [{"time", "value"}]}'}</code>; they correct the bias.</span></li>
          <li><code>GET {base}</code><span>Checks the key and lists the inputs to send.</span></li>
        </ul>
        <p className="manual-hint">
          Send {(deployment.inputs || []).map((i) => i.name).join(', ')} for each time stamp, oldest first, in the units of the
          training data. Rows not newer than the last one are ignored, so resending after a reconnect is safe. At most 120 requests
          per minute.
        </p>
      </div>

      <CopyBlock label="curl" text={curl} />
      <CopyBlock label="Python" text={python} />
      {error && <p className="manual-error">{error}</p>}
    </div>
  )
}
