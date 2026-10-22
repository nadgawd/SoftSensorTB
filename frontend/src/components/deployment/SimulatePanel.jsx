import { useEffect, useRef, useState } from 'react'
import { Play, Square } from 'lucide-react'
import { errorDetail, fetchSimulationRows, sendLabResults, sendLiveRows } from '../../api/client'

const PAGE = 500

/**
 * Replays the deployment's own data as a live feed, through the same endpoint a plant connector uses,
 * with each lab value sent a while after its sample (as a lab would).
 */
export default function SimulatePanel({ deployment, onTick }) {
  const [total, setTotal] = useState(null)
  const [start, setStart] = useState(0)
  const [batch, setBatch] = useState(5)
  const [intervalS, setIntervalS] = useState(1)
  const [labDelay, setLabDelay] = useState(30)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(null)
  const [error, setError] = useState(null)
  const stopRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    fetchSimulationRows(deployment.id, 0, 1)
      .then((d) => {
        if (cancelled) return
        setTotal(d.total)
        setStart(Math.floor(d.total * 0.7))
      })
      .catch((err) => { if (!cancelled) setError(errorDetail(err, 'Could not load the data to replay.')) })
    return () => { cancelled = true; stopRef.current = true }
  }, [deployment.id])

  async function run() {
    stopRef.current = false
    setRunning(true)
    setError(null)
    const pending = []
    const tally = { sent: 0, ok: 0, warning: 0, suppressed: 0, ignored: 0, labs: 0 }
    let page = { start: -1, rows: [], labs: [] }
    let position = start
    try {
      while (!stopRef.current && position < total) {
        if (position >= page.start + page.rows.length || position < page.start) {
          const d = await fetchSimulationRows(deployment.id, position, PAGE)
          page = { start: d.start, rows: d.rows, labs: d.labs }
          if (!d.rows.length) break
        }
        const offset = position - page.start
        const rows = page.rows.slice(offset, offset + batch)
        const labValues = page.labs.slice(offset, offset + batch)
        const predictions = await sendLiveRows(deployment.id, rows)
        predictions.forEach((p, i) => {
          tally[p.status] = (tally[p.status] || 0) + 1
          if (p.status !== 'ignored' && labValues[i] != null) {
            pending.push({ time: p.time, value: labValues[i], due: tally.sent + i + labDelay })
          }
        })
        tally.sent += rows.length
        position += rows.length
        const due = pending.filter((l) => l.due <= tally.sent)
        if (due.length) {
          await sendLabResults(deployment.id, due.map(({ time, value }) => ({ time, value })))
          tally.labs += due.length
          pending.splice(0, pending.length, ...pending.filter((l) => l.due > tally.sent))
        }
        setProgress({ ...tally, position, last: predictions[predictions.length - 1] })
        onTick()
        await new Promise((resolve) => setTimeout(resolve, intervalS * 1000))
      }
      if (!stopRef.current && pending.length) {
        await sendLabResults(deployment.id, pending.map(({ time, value }) => ({ time, value })))
        tally.labs += pending.length
        setProgress((p) => ({ ...p, labs: tally.labs }))
        onTick()
      }
    } catch (err) {
      setError(errorDetail(err, 'The simulation stopped.'))
    } finally {
      setRunning(false)
    }
  }

  const target = deployment.model?.target_column || 'the target'

  return (
    <div className="deploy-sim">
      <p className="manual-hint">
        Replays this dataset as if it were arriving from the plant: rows are sent to the live endpoint a few at a time and the
        lab values of {target} follow some rows later. Watch the Monitor tab. Simulated rows are stored like real ones, and rows
        not newer than the last reading are ignored, so replay later rows (the last 30% by default, roughly the test period).
      </p>
      <div className="deploy-sim-form">
        <label className="manual-label" htmlFor="sim-start">Start at row <span className="manual-hint">of {total ?? '…'}</span></label>
        <input id="sim-start" className="manual-input" type="number" min={0} max={Math.max(0, (total || 1) - 1)} value={start}
          disabled={running} onChange={(e) => setStart(Math.max(0, Math.floor(Number(e.target.value) || 0)))} />
        <label className="manual-label" htmlFor="sim-batch">Rows per send</label>
        <input id="sim-batch" className="manual-input" type="number" min={1} max={200} value={batch}
          disabled={running} onChange={(e) => setBatch(Math.min(200, Math.max(1, Math.floor(Number(e.target.value) || 1))))} />
        <label className="manual-label" htmlFor="sim-interval">Seconds between sends</label>
        <input id="sim-interval" className="manual-input" type="number" min={1} max={60} step={0.5} value={intervalS}
          disabled={running} onChange={(e) => setIntervalS(Math.min(60, Math.max(1, Number(e.target.value) || 1)))} />
        <label className="manual-label" htmlFor="sim-lab">Lab result arrives after <span className="manual-hint">rows</span></label>
        <input id="sim-lab" className="manual-input" type="number" min={0} max={1000} value={labDelay}
          disabled={running} onChange={(e) => setLabDelay(Math.min(1000, Math.max(0, Math.floor(Number(e.target.value) || 0))))} />
      </div>
      <div className="manual-actions">
        {running ? (
          <button type="button" className="btn-ghost" onClick={() => { stopRef.current = true }}>
            <Square size={13} aria-hidden="true" /> Stop
          </button>
        ) : (
          <button type="button" className="btn-signal" disabled={!total || deployment.status !== 'active'} onClick={run}>
            <Play size={13} aria-hidden="true" /> Start live feed
          </button>
        )}
        {deployment.status !== 'active' && <span className="manual-hint">The deployment is paused.</span>}
      </div>
      {progress && (
        <p className="manual-hint deploy-mono">
          Row {progress.position} of {total} · sent {progress.sent} · ok {progress.ok} · warning {progress.warning} ·
          suppressed {progress.suppressed} · ignored {progress.ignored} · lab results {progress.labs}
          {progress.last?.reasons?.length ? ` · last: ${progress.last.reasons[0]}` : ''}
        </p>
      )}
      {error && <p className="manual-error">{error}</p>}
    </div>
  )
}
