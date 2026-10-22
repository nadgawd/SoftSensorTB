import { useCallback, useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { errorDetail, fetchDeploymentMonitor } from '../../api/client'
import { useTheme } from '../../hooks/useTheme'

const POLL_MS = 5000
// Same margin as the server's range check (backend/deployment/monitor.py RANGE_MARGIN).
const RANGE_MARGIN = 0.05

function fmt(v, digits = 4) {
  return typeof v === 'number' && Number.isFinite(v) ? Number(v.toPrecision(digits)).toString() : '—'
}

function ago(iso) {
  if (!iso) return 'never'
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000)
  if (s < 90) return `${Math.round(s)} s ago`
  if (s < 5400) return `${Math.round(s / 60)} min ago`
  if (s < 172800) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} days ago`
}

function when(iso) {
  return iso ? new Date(iso).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'medium' }) : '—'
}

function plotColors(theme) {
  const css = getComputedStyle(document.documentElement)
  const get = (name, fallback) => css.getPropertyValue(name).trim() || fallback
  return {
    font: get('--plot-font', theme === 'light' ? '#1a2433' : '#e7f2f4'),
    bg: get('--plot-bg', theme === 'light' ? '#ffffff' : 'rgba(11,24,30,0.55)'),
    grid: get('--plot-grid', 'rgba(36,53,69,0.6)'),
    signal: get('--color-signal', '#2ec4b6'),
    ember: get('--color-ember', '#d4a535'),
    success: get('--color-success', '#3cb371'),
    mist: get('--color-mist', '#8aa4b8'),
    alert: get('--color-alert', '#e4572e'),
  }
}

export function StatusChip({ status }) {
  return <span className={`deploy-chip status-${status}`}>{status}</span>
}

/** Live view of one deployment: what came in, what it predicted, how the lab results compare. */
export default function MonitorPanel({ deploymentId, refreshKey }) {
  const { theme } = useTheme()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(() => {
    fetchDeploymentMonitor(deploymentId, 500)
      .then((d) => { setData(d); setError(null) })
      .catch((err) => setError(errorDetail(err, 'Could not load the live data.')))
  }, [deploymentId])

  useEffect(() => {
    load()
    const timer = setInterval(() => { if (document.visibilityState === 'visible') load() }, POLL_MS)
    return () => clearInterval(timer)
  }, [load, refreshKey])

  if (error && !data) return <p className="manual-error">{error}</p>
  if (!data) return <p className="manual-hint">Loading the live data…</p>

  const { deployment: dep, records, labs, accuracy } = data
  const counts = dep.counts || {}
  const target = dep.model?.target_column || 'target'
  const colors = plotColors(theme)
  const latest = records[records.length - 1]
  const hasBias = records.some((r) => r.bias)
  const times = records.map((r) => r.time)

  const traces = [
    {
      x: times, y: records.map((r) => r.corrected), name: hasBias ? 'Prediction (bias-corrected)' : 'Prediction',
      type: 'scatter', mode: 'lines', line: { color: colors.signal, width: 2 }, connectgaps: false,
    },
    hasBias && {
      x: times, y: records.map((r) => r.prediction), name: 'Model output', type: 'scatter', mode: 'lines',
      line: { color: colors.mist, width: 1, dash: 'dot' }, connectgaps: false,
    },
    {
      x: records.filter((r) => r.status === 'warning').map((r) => r.time),
      y: records.filter((r) => r.status === 'warning').map((r) => r.corrected),
      name: 'Warning', type: 'scatter', mode: 'markers', marker: { color: colors.ember, size: 6 },
      text: records.filter((r) => r.status === 'warning').map((r) => r.reasons.join('<br>')), hoverinfo: 'x+y+text',
    },
    labs.length > 0 && {
      x: labs.map((l) => l.time), y: labs.map((l) => l.value), name: 'Lab result', type: 'scatter', mode: 'markers',
      marker: { color: colors.success, size: 9, symbol: 'diamond', line: { width: 1, color: colors.font } },
    },
  ].filter(Boolean)

  const health = (dep.inputs || []).filter((i) => !i.is_time)

  return (
    <div className="deploy-monitor">
      <div className="deploy-kpis">
        <div className="deploy-kpi">
          <span className="deploy-kpi-label">Last reading</span>
          <span className="deploy-kpi-value">{dep.last_time ? when(dep.last_time) : 'none yet'}</span>
          <span className="deploy-kpi-sub">received {ago(dep.last_received)}</span>
        </div>
        <div className="deploy-kpi">
          <span className="deploy-kpi-label">Latest {target}</span>
          <span className="deploy-kpi-value">{latest ? fmt(latest.corrected) : '—'}</span>
          <span className="deploy-kpi-sub">{latest ? <StatusChip status={latest.status} /> : 'no prediction yet'}</span>
        </div>
        <div className="deploy-kpi">
          <span className="deploy-kpi-label">Rows</span>
          <span className="deploy-kpi-value">{(counts.ok || 0) + (counts.warning || 0)} predicted</span>
          <span className="deploy-kpi-sub">{counts.warning || 0} with warnings · {counts.suppressed || 0} suppressed · {counts.ignored || 0} repeats</span>
        </div>
        <div className="deploy-kpi">
          <span className="deploy-kpi-label">Bias</span>
          <span className="deploy-kpi-value">{fmt(dep.bias)}</span>
          <span className="deploy-kpi-sub">gain {dep.bias_gain} · from {labs.filter((l) => l.corrected != null).length} lab results</span>
        </div>
        <div className={`deploy-kpi${accuracy.alarm ? ' alarm' : ''}`}>
          <span className="deploy-kpi-label">Lab error (RMSE)</span>
          <span className="deploy-kpi-value">{fmt(accuracy.rmse)}</span>
          <span className="deploy-kpi-sub">test RMSE {fmt(accuracy.test_rmse)} · last {accuracy.n} labs</span>
        </div>
      </div>
      {accuracy.alarm && <p className="deploy-banner alert">{accuracy.message}</p>}
      {dep.status === 'paused' && <p className="deploy-banner warn">Paused: rows sent now are refused until you resume it in Settings.</p>}

      {records.length ? (
        <div className="deploy-chart">
          <Plot
            data={traces}
            layout={{
              autosize: true, height: 320, margin: { t: 16, r: 16, b: 40, l: 56 },
              paper_bgcolor: 'transparent', plot_bgcolor: colors.bg,
              font: { family: 'Sora, sans-serif', color: colors.font, size: 11 },
              xaxis: { gridcolor: colors.grid, linecolor: colors.grid },
              yaxis: { gridcolor: colors.grid, linecolor: colors.grid, title: { text: target } },
              legend: { orientation: 'h', y: 1.12 }, hovermode: 'closest',
            }}
            config={{ displaylogo: false, responsive: true }}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </div>
      ) : (
        <p className="manual-hint deploy-empty">
          No rows received yet. Connect the plant (Connect tab) or replay this dataset as a live feed (Simulate tab).
        </p>
      )}

      {latest && health.length > 0 && (
        <div>
          <p className="deploy-subtitle">Latest readings against the training range</p>
          <div className="manual-table-wrap">
            <table className="data-table deploy-table">
              <thead><tr><th>Input</th><th>Latest</th><th>Training range</th></tr></thead>
              <tbody>
                {health.map((i) => {
                  const v = latest.inputs?.[i.name]
                  const st = i.stats
                  const margin = st?.numeric ? RANGE_MARGIN * ((st.high ?? 0) - (st.low ?? 0)) : 0
                  const out = st?.numeric && typeof v === 'number' && (v < st.low - margin || v > st.high + margin)
                  return (
                    <tr key={i.name} className={v == null ? 'deploy-row-missing' : out ? 'deploy-row-out' : undefined}>
                      <td><code>{i.name}</code></td>
                      <td>{v == null ? 'missing' : typeof v === 'number' ? fmt(v, 6) : String(v)}</td>
                      <td>{st?.numeric ? `${fmt(st.low)} … ${fmt(st.high)}` : st?.categories?.length ? st.categories.slice(0, 8).join(', ') : '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {records.length > 0 && (
        <div>
          <p className="deploy-subtitle">Latest predictions</p>
          <div className="manual-table-wrap deploy-records">
            <table className="data-table deploy-table">
              <thead><tr><th>Time</th><th>Status</th><th>{target}</th><th>T²</th><th>SPE</th><th>Why</th></tr></thead>
              <tbody>
                {records.slice(-40).reverse().map((r) => (
                  <tr key={r.id}>
                    <td className="deploy-mono">{when(r.time)}</td>
                    <td><StatusChip status={r.status} /></td>
                    <td className="deploy-mono">{fmt(r.corrected)}</td>
                    <td className="deploy-mono">{fmt(r.t2, 3)}</td>
                    <td className="deploy-mono">{fmt(r.spe, 3)}</td>
                    <td className="deploy-note">{r.reasons.join(' ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="manual-hint">
            T² measures how far the inputs are from the training operating point; SPE measures how much they break the
            relationships seen in training. Either above its training limit gives a warning.
          </p>
        </div>
      )}
      {error && <p className="manual-error">{error}</p>}
    </div>
  )
}
