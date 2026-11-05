import { useEffect, useMemo, useRef, useState } from 'react'
import Plot from 'react-plotly.js'
import ToolForm from './ToolForm'
import Elapsed from '../Elapsed'
import { errorDetail, fetchSeries, previewDataSplit, runManualTool } from '../../api/client'
import { useDatasetSchema } from '../../hooks/useManualTools'
import { useTheme } from '../../hooks/useTheme'
import { buildArguments, formValues } from '../../lib/manualForm'
import { formatCount, parseServerTime } from '../../utils/format'
import { SPLIT_SETS, splitCountsText, splitLabel } from '../../utils/splits'

const TOOL = 'split_data'

const METHODS = [
  { id: 'random', label: 'Random', note: 'The rows are shuffled, then each set gets its share. For data without a time order.' },
  {
    id: 'stratified',
    label: 'Stratified',
    note: 'Like random, but every set gets the same mix of low and high target values (or of the groups of a column).',
  },
  {
    id: 'sequential',
    label: 'Chronological',
    note: 'The last rows (in time order) are the test set and the ones just before them the validation set, so the model is tested on later data than it learned from.',
  },
  { id: 'group', label: 'Group', note: 'Whole groups (batches, campaigns, units) go to one set, so no group is in two sets.' },
  {
    id: 'custom',
    label: 'Custom',
    note: 'Choose the rows of each set yourself: drag across the plot, or type row numbers or times.',
  },
]

const PAINT_SETS = [...SPLIT_SETS, { id: 'none', code: 0, label: 'Not used', color: '--color-mist' }]
const SET_OF_CODE = Object.fromEntries(PAINT_SETS.map((s) => [s.code, s]))
const CODE_OF_SET = Object.fromEntries(PAINT_SETS.map((s) => [s.id, s.code]))
const PLOT_ORDER = [1, 2, 3, 0]
const CUSTOM_FIELDS = new Set(['target_column', 'time_column'])
const PANEL_ONLY = new Set(['split_strategy', 'ranges', 'rest'])

function plotColors(theme) {
  const css = getComputedStyle(document.documentElement)
  const get = (name, fallback) => css.getPropertyValue(name).trim() || fallback
  return {
    font: get('--plot-font', theme === 'light' ? '#1a2433' : '#e7f2f4'),
    bg: get('--plot-bg', theme === 'light' ? '#ffffff' : 'rgba(11,24,30,0.55)'),
    grid: get('--plot-grid', 'rgba(36,53,69,0.6)'),
    sets: Object.fromEntries(PAINT_SETS.map((s) => [s.code, get(s.color, '#8aa4b8')])),
  }
}

/** "2024-01-02 03:04:05.678" (or with a "T") in milliseconds, read as UTC like the plot's date axis; NaN when it is no time. */
function timeMs(value) {
  const m = /^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?)?$/.exec(String(value ?? '').trim())
  if (!m) return NaN
  const ms = m[7] ? Number(m[7].slice(0, 3).padEnd(3, '0')) : 0
  return Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]), Number(m[4] || 0), Number(m[5] || 0), Number(m[6] || 0), ms)
}

/** A time from the plot ("2024-01-02 03:04:05.6789") as the split stores it, to the millisecond. */
function plotTime(value) {
  return String(value).replace('T', ' ').replace(/(\.\d{3})\d+$/, '$1')
}

function blank(value) {
  return value === null || value === undefined || String(value).trim() === ''
}

/** Whether range `r` covers the point at row `row` with axis value `x` (`kind` of the time column's values). */
function covers(r, row, x, kind) {
  let value
  let lo
  let hi
  if (r.by === 'index') {
    value = row
    lo = blank(r.start) ? null : Number(r.start)
    hi = blank(r.end) ? null : Number(r.end)
  } else if (kind === 'time') {
    value = timeMs(x)
    lo = blank(r.start) ? null : timeMs(r.start)
    hi = blank(r.end) ? null : timeMs(r.end)
  } else if (kind === 'number') {
    value = x
    lo = blank(r.start) ? null : Number(r.start)
    hi = blank(r.end) ? null : Number(r.end)
  } else {
    return false
  }
  if (lo !== null && hi !== null && hi < lo) [lo, hi] = [hi, lo]
  return (lo === null || value >= lo) && (hi === null || value <= hi)
}

/** The set code of every plotted point under the draft ranges: later ranges win, then `rest`; rows without a target get none. */
function draftCodes(series, ranges, rest) {
  const restCode = CODE_OF_SET[rest] ?? 0
  return series.row.map((row, i) => {
    if (series.has_target && !series.has_target[i]) return 0
    let code = null
    for (const r of ranges) {
      if (covers(r, row, series.x[i], series.x_kind)) code = CODE_OF_SET[r.set] ?? 0
    }
    return code ?? restCode
  })
}

function rangeText(r) {
  const what = r.by === 'index' ? 'rows' : ''
  const start = blank(r.start) ? 'the start' : String(r.start)
  const end = blank(r.end) ? 'the end' : String(r.end)
  return `${what} ${start} – ${end}`.trim()
}

function isCancel(err) {
  return err?.name === 'CanceledError' || err?.code === 'ERR_CANCELED'
}

function SplitBar({ counts }) {
  const parts = SPLIT_SETS.map((s) => ({ ...s, rows: counts?.[s.id] || 0 })).filter((p) => p.rows > 0)
  const total = parts.reduce((sum, p) => sum + p.rows, 0)
  if (!total) return null
  return (
    <div className="split-bar" role="img" aria-label={parts.map((p) => `${p.label} ${p.rows} rows`).join(', ')}>
      {parts.map((p) => (
        <div key={p.id} className={`split-seg split-${p.id}`} style={{ flexGrow: p.rows }}>
          {p.label} {formatCount(p.rows)} · {Math.round((100 * p.rows) / total)}%
        </div>
      ))}
    </div>
  )
}

/** The rows of each set, from the server: what the split would make (preview) or made (saved). */
function CountsLine({ counts, extra }) {
  if (!counts) return null
  return (
    <p className="split-counts">
      {PAINT_SETS.map((s) => {
        const n = s.id === 'none' ? counts.unused : counts[s.id]
        if (s.id === 'validation' && !n) return null
        if (s.id === 'none' && !n) return null
        return (
          <span key={s.id} className="split-count">
            <span className={`split-dot split-${s.id}`} />
            {s.label} <strong>{formatCount(n || 0)}</strong>
          </span>
        )
      })}
      {extra}
    </p>
  )
}

/**
 * Step 1 of modeling: split the rows into train / validation / test and save the split.
 * The fixed plot shows a column against time (or the row number) coloured by set; with the
 * custom method, dragging across it paints the rows of the set chosen in "Paint as".
 */
export default function SplitPanel({
  datasetVersionId,
  targetVariable,
  selectedFeatures,
  timeColumn,
  modelKey,
  tools,
  edits,
  editsFrom,
  onEdit,
  onReset,
  split,
  assistantBusy,
  onBusyChange,
  onResult,
}) {
  const { theme } = useTheme()
  const colors = useMemo(() => plotColors(theme), [theme])
  const { schema, error: schemaError, loading: schemaLoading } = useDatasetSchema(datasetVersionId, modelKey)
  const tool = useMemo(() => (tools || []).find((t) => t.name === TOOL), [tools])
  const app = useMemo(() => ({ targetVariable, selectedFeatures, timeColumn }), [targetVariable, selectedFeatures, timeColumn])
  const values = useMemo(() => (tool ? formValues(tool, edits[TOOL], schema, app) : {}), [tool, edits, schema, app])

  const [errors, setErrors] = useState({})
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState(null)
  const [variable, setVariable] = useState('')
  const [axis, setAxis] = useState('time')
  const [paint, setPaint] = useState('train')
  const [typed, setTyped] = useState({ by: '', from: '', to: '' })
  const [typedError, setTypedError] = useState(null)
  const [seriesState, setSeriesState] = useState({ data: null, error: null, note: null, loading: false })
  const [preview, setPreview] = useState({ data: null, error: null, loading: false })
  const [selectionRevision, setSelectionRevision] = useState(0)
  const selecting = useRef(null)

  const method = values.split_strategy || 'random'
  const custom = method === 'custom'
  const ranges = useMemo(() => (Array.isArray(values.ranges) ? values.ranges : []), [values.ranges])
  const rest = values.rest || ''
  const target = values.target_column || ''
  const timeCol = values.time_column || ''
  const saved = split?.split || null
  const savedFits = Boolean(saved && saved.target_column === target)

  const numeric = useMemo(() => (schema?.columns || []).filter((c) => c.numeric).map((c) => c.name), [schema])
  const variableGroups = useMemo(() => {
    const features = (selectedFeatures || []).filter((c) => c !== target && numeric.includes(c))
    const other = numeric.filter((c) => c !== target && c !== timeCol && !features.includes(c))
    return [
      { label: 'Target', columns: numeric.includes(target) ? [target] : [] },
      { label: 'Features', columns: features },
      { label: 'Other columns', columns: other },
    ].filter((g) => g.columns.length)
  }, [numeric, selectedFeatures, target, timeCol])
  const plotColumn = numeric.includes(variable) ? variable : (variableGroups[0]?.columns[0] || '')
  const byTime = Boolean(timeCol) && axis === 'time'

  // The column, always with the time column when there is one so ranges by time can colour the points.
  const seriesKey = JSON.stringify({
    vid: datasetVersionId, column: plotColumn, time: timeCol, target, split: !custom && savedFits ? saved.split_id : null,
  })
  useEffect(() => {
    const request = JSON.parse(seriesKey)
    if (!request.vid || !request.column) {
      setSeriesState({ data: null, error: null, note: null, loading: false })
      return undefined
    }
    const controller = new AbortController()
    const params = { column: request.column, timeColumn: request.time, target: request.target, splitId: request.split }
    setSeriesState((prev) => ({ ...prev, loading: true, error: null }))
    fetchSeries(request.vid, params, { signal: controller.signal })
      .then((data) => ({ data, note: null }))
      .catch((err) => {
        if (isCancel(err) || !request.time || err?.response?.status !== 400) throw err
        return fetchSeries(request.vid, { ...params, timeColumn: null }, { signal: controller.signal })
          .then((data) => ({ data, note: `Plotted against the row number: ${errorDetail(err)}` }))
      })
      .then(({ data, note }) => setSeriesState({ data, error: null, note, loading: false }))
      .catch((err) => {
        if (!isCancel(err)) setSeriesState({ data: null, error: errorDetail(err, 'Could not load the column.'), note: null, loading: false })
      })
    return () => controller.abort()
  }, [seriesKey])

  const built = useMemo(() => (tool ? buildArguments(tool, values) : { args: {}, errors: {} }), [tool, values])
  const previewKey = datasetVersionId && target && !Object.keys(built.errors).length && (!custom || ranges.length)
    ? JSON.stringify(built.args)
    : ''
  useEffect(() => {
    if (!previewKey) {
      setPreview({ data: null, error: null, loading: false })
      return undefined
    }
    const controller = new AbortController()
    setPreview((prev) => ({ ...prev, loading: true }))
    const timer = setTimeout(() => {
      previewDataSplit(datasetVersionId, JSON.parse(previewKey), { signal: controller.signal })
        .then((data) => setPreview({ data, error: null, loading: false }))
        .catch((err) => {
          if (!isCancel(err)) setPreview({ data: null, error: errorDetail(err, 'Could not preview the split.'), loading: false })
        })
    }, 300)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [datasetVersionId, previewKey])

  const series = seriesState.data
  const timeAxis = byTime && series && series.x_kind !== 'index'
  const codes = useMemo(() => {
    if (!series) return null
    if (custom) return draftCodes(series, ranges, rest)
    return series.set || null
  }, [series, custom, ranges, rest])

  const traces = useMemo(() => {
    if (!series) return []
    const xs = timeAxis ? series.x : series.row
    const hover = `${timeAxis ? '%{x}<br>row %{customdata}' : 'row %{x}'}<br>${series.column} %{y:.6g}<extra>%{fullData.name}</extra>`
    const base = { type: 'scattergl', mode: 'markers', hovertemplate: hover }
    if (!codes) {
      return [{ ...base, x: xs, y: series.y, customdata: series.row, name: series.column, marker: { size: 4, color: colors.sets[1] } }]
    }
    return PLOT_ORDER.map((code) => {
      const idx = []
      codes.forEach((c, i) => { if (c === code) idx.push(i) })
      return {
        ...base,
        x: idx.map((i) => xs[i]),
        y: idx.map((i) => series.y[i]),
        customdata: idx.map((i) => series.row[i]),
        name: SET_OF_CODE[code].label,
        marker: { size: code === 0 ? 3 : 4, color: colors.sets[code], opacity: code === 0 ? 0.45 : 0.9 },
      }
    }).filter((t) => t.x.length)
  }, [series, codes, timeAxis, colors.sets])

  // Shaded bands for the drawn ranges, on whichever axis is shown.
  const shapes = useMemo(() => {
    if (!custom || !series || !series.row.length) return []
    const first = timeAxis ? series.x.reduce((a, b) => (series.x_kind === 'time' ? (timeMs(b) < timeMs(a) ? b : a) : Math.min(a, b))) : null
    const last = timeAxis ? series.x.reduce((a, b) => (series.x_kind === 'time' ? (timeMs(b) > timeMs(a) ? b : a) : Math.max(a, b))) : null
    return ranges.map((r) => {
      let x0
      let x1
      if (timeAxis && r.by === 'time') {
        x0 = blank(r.start) ? first : r.start
        x1 = blank(r.end) ? last : r.end
      } else if (!timeAxis && r.by === 'index') {
        x0 = (blank(r.start) ? 0 : Number(r.start)) - 0.5
        x1 = (blank(r.end) ? series.n_rows - 1 : Number(r.end)) + 0.5
      } else {
        const hit = series.row.map((row, i) => (covers(r, row, series.x[i], series.x_kind) ? i : -1)).filter((i) => i >= 0)
        if (!hit.length) return null
        const at = hit.map((i) => (timeAxis ? series.x[i] : series.row[i]))
        const order = timeAxis && series.x_kind === 'time' ? at.map(timeMs) : at
        x0 = at[order.indexOf(Math.min(...order))]
        x1 = at[order.indexOf(Math.max(...order))]
      }
      const color = colors.sets[CODE_OF_SET[r.set] ?? 0]
      return {
        type: 'rect', xref: 'x', yref: 'paper', x0, x1, y0: 0, y1: 1, layer: 'below',
        fillcolor: color, opacity: 0.14, line: { width: 1, color, dash: 'dot' },
      }
    }).filter(Boolean)
  }, [custom, series, ranges, timeAxis, colors.sets])

  if (!tool) return null

  const counts = preview.data?.counts || null
  const uncovered = custom ? preview.data?.uncovered ?? null : null
  const painted = new Set(ranges.map((r) => r.set))
  const missing = SPLIT_SETS.filter((s) => !painted.has(s.id))
  const busy = running || assistantBusy
  const edited = Boolean(edits[TOOL] && Object.keys(edits[TOOL]).length)
  const methodNote = METHODS.find((m) => m.id === method)?.note
  const typedBy = typed.by === 'time' && !timeCol ? 'index' : typed.by || (byTime ? 'time' : 'index')
  const formTool = {
    ...tool,
    fields: tool.fields.filter((f) => !PANEL_ONLY.has(f.name) && (!custom || CUSTOM_FIELDS.has(f.name))),
  }

  function setField(name, value) {
    onEdit(TOOL, name, value)
    if (errors[name]) setErrors((prev) => ({ ...prev, [name]: undefined }))
  }

  function addRange(range) {
    setField('ranges', [...ranges, range])
  }

  function onSelected(event) {
    const span = event?.range?.x || selecting.current
    selecting.current = null
    setSelectionRevision((n) => n + 1)
    if (!custom || !span || span.length < 2) return
    let [a, b] = span
    if (timeAxis) {
      if (series.x_kind === 'time') {
        if (timeMs(b) < timeMs(a)) [a, b] = [b, a]
        addRange({ set: paint, by: 'time', start: plotTime(a), end: plotTime(b) })
      } else {
        addRange({ set: paint, by: 'time', start: Math.min(a, b), end: Math.max(a, b) })
      }
      return
    }
    const start = Math.max(0, Math.ceil(Math.min(a, b)))
    const end = Math.min(series.n_rows - 1, Math.floor(Math.max(a, b)))
    if (end >= start) addRange({ set: paint, by: 'index', start, end })
  }

  function addTyped() {
    const by = typedBy
    const from = typed.from.trim()
    const to = typed.to.trim()
    const check = (text, what) => {
      if (!text) return null
      if (by === 'index') {
        const n = Number(text)
        if (!Number.isInteger(n) || n < 0) throw new Error(`The ${what} must be a row number (0 or more).`)
        return n
      }
      if (series?.x_kind === 'number') {
        const n = Number(text)
        if (!Number.isFinite(n)) throw new Error(`The ${what} must be a number of ${timeCol}.`)
        return n
      }
      if (Number.isNaN(timeMs(text))) throw new Error(`The ${what} must be a time like 2024-01-31 or 2024-01-31 06:00.`)
      return text.replace('T', ' ')
    }
    try {
      addRange({ set: paint, by, start: check(from, 'start'), end: check(to, 'end') })
      setTyped({ by: typed.by, from: '', to: '' })
      setTypedError(null)
    } catch (err) {
      setTypedError(err.message)
    }
  }

  function reset() {
    onReset(TOOL)
    setErrors({})
    setStatus(null)
  }

  async function save() {
    const { args, errors: found } = built
    setErrors(found)
    if (Object.keys(found).length) {
      setStatus({ ok: false, text: 'Fix the highlighted fields first.' })
      return
    }
    if (custom && !ranges.length) {
      setStatus({ ok: false, text: 'Drag across the plot or type a range first.' })
      return
    }
    setRunning(true)
    onBusyChange?.(true)
    setStatus(null)
    try {
      const result = await runManualTool(TOOL, datasetVersionId, args)
      await onResult(result, { tool: TOOL, label: tool.label, args })
      setStatus({ ok: true, text: result.message || 'Split saved.' })
    } catch (err) {
      setStatus({ ok: false, text: errorDetail(err, 'The split failed.') })
    } finally {
      setRunning(false)
      onBusyChange?.(false)
    }
  }

  const savedAt = parseServerTime(saved?.created_at)

  return (
    <section className="panel manual-panel split-panel" aria-label="Split the data">
      <div className="manual-head">
        <p className="manual-title">{tool.label}</p>
        {editsFrom[TOOL] === 'assistant' && <span className="badge badge-ember manual-source">Set by the assistant’s last run</span>}
      </div>
      <p className="manual-tool-desc">
        Decide which rows the model learns from (train), which tune it (validation, optional) and which test it at the end.
        Training below uses the saved split, so every model is tested on the same rows.
      </p>

      <div className="split-current">
        {saved ? (
          <>
            <p className="manual-hint">
              <strong>Saved split</strong> · {splitLabel(saved.strategy)} for {saved.target_column} · {splitCountsText(saved.counts)}
              {saved.counts?.unused ? ` · ${formatCount(saved.counts.unused)} not used` : ''}
              {savedAt ? ` · ${savedAt.toLocaleString()}` : ''} · split {saved.split_id.slice(0, 8)}
            </p>
            <SplitBar counts={saved.counts} />
          </>
        ) : split?.loading ? (
          <p className="manual-hint">Loading the saved split…</p>
        ) : split?.outdated ? (
          <p className="manual-error">The last split no longer fits this data: {split.outdatedReason}. Split again.</p>
        ) : (
          <p className="manual-hint">No split saved yet for this data.</p>
        )}
        {split?.error && <p className="manual-error">{split.error}</p>}
      </div>

      <div className="split-methods" role="radiogroup" aria-label="How to split">
        {METHODS.map((m) => (
          <button
            key={m.id}
            type="button"
            role="radio"
            aria-checked={method === m.id}
            className={`split-tab${method === m.id ? ' active' : ''}`}
            onClick={() => setField('split_strategy', m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>
      {methodNote && <p className="manual-hint split-method-note">{methodNote}</p>}

      {schemaError && <p className="manual-error">Could not load the dataset columns: {schemaError}</p>}
      {!schema && schemaLoading ? (
        <p className="manual-hint">Loading columns…</p>
      ) : (
        <ToolForm tool={formTool} values={values} onChange={setField} schema={schema} errors={errors} />
      )}

      <div className="split-plot-bar">
        <label className="manual-label" htmlFor="split-plot-column">Plot</label>
        <select
          id="split-plot-column"
          className="input-base manual-input"
          value={plotColumn}
          onChange={(e) => setVariable(e.target.value)}
          disabled={!variableGroups.length}
        >
          {!variableGroups.length && <option value="">No numeric columns</option>}
          {variableGroups.map((g) => (
            <optgroup key={g.label} label={g.label}>
              {g.columns.map((c) => <option key={c} value={c}>{c}</option>)}
            </optgroup>
          ))}
        </select>
        <span className="manual-label">against</span>
        <div className="split-axis" role="radiogroup" aria-label="Plot against">
          <button
            type="button"
            role="radio"
            aria-checked={byTime}
            className={`split-tab${byTime ? ' active' : ''}`}
            disabled={!timeCol}
            title={timeCol ? undefined : 'Choose a time column above to plot against time'}
            onClick={() => setAxis('time')}
          >
            {timeCol || 'Time'}
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={!byTime}
            className={`split-tab${!byTime ? ' active' : ''}`}
            onClick={() => setAxis('index')}
          >
            Row number
          </button>
        </div>
        {custom && (
          <span className="split-paint-group">
            <label className="manual-label" htmlFor="split-paint">Paint as</label>
            <select id="split-paint" className={`input-base manual-input split-paint split-paint-${paint}`} value={paint} onChange={(e) => setPaint(e.target.value)}>
              {PAINT_SETS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
            </select>
          </span>
        )}
      </div>

      <div className="split-plot">
        {seriesState.error ? (
          <p className="manual-error">{seriesState.error}</p>
        ) : !series ? (
          <p className="manual-hint">{seriesState.loading ? 'Loading the plot…' : 'Choose a column to plot.'}</p>
        ) : (
          <Plot
            // Fresh trace objects drop the `selectedpoints` Plotly stores on them after a drag.
            data={traces.map((t) => ({ ...t }))}
            layout={{
              autosize: true,
              height: 320,
              margin: { t: 12, r: 16, b: 44, l: 60 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: colors.bg,
              font: { family: 'Sora, sans-serif', color: colors.font, size: 11 },
              xaxis: {
                gridcolor: colors.grid,
                linecolor: colors.grid,
                type: timeAxis && series.x_kind === 'time' ? 'date' : 'linear',
                title: { text: timeAxis ? series.time_column : 'Row number' },
              },
              yaxis: { gridcolor: colors.grid, linecolor: colors.grid, title: { text: series.column } },
              legend: { orientation: 'h', y: 1.1 },
              hovermode: 'closest',
              dragmode: custom ? 'select' : 'zoom',
              selectdirection: 'h',
              selections: [],
              selectionrevision: selectionRevision,
              shapes,
              uirevision: `${plotColumn}|${timeAxis ? 'time' : 'row'}|${custom}`,
            }}
            config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'autoScale2d'] }}
            useResizeHandler
            style={{ width: '100%' }}
            onSelecting={(e) => { selecting.current = e?.range?.x || selecting.current }}
            onSelected={onSelected}
          />
        )}
        {series && (
          <p className="manual-hint split-plot-note">
            {custom
              ? `Drag across the plot to put those rows in ${PAINT_SETS.find((s) => s.id === paint).label}. Use the zoom tool in the plot’s toolbar to look closer, then switch back to Box select.`
              : savedFits
                ? 'Coloured by the saved split. Choose Custom to pick the rows yourself.'
                : 'Choose Custom to pick the rows of each set by dragging across the plot.'}
            {series.sampled && ` Showing ${formatCount(series.row.length)} of ${formatCount(series.n_values)} points (each stretch keeps its lowest and highest value).`}
            {seriesState.note && ` ${seriesState.note}`}
          </p>
        )}
      </div>

      {custom && (
        <div className="split-custom">
          <div className="split-typed">
            <span className="manual-label">Or type a range</span>
            <select
              className="input-base manual-input"
              aria-label="Range by"
              value={typedBy}
              onChange={(e) => setTyped({ ...typed, by: e.target.value })}
            >
              <option value="index">Row numbers</option>
              <option value="time" disabled={!timeCol}>{timeCol ? `Times of ${timeCol}` : 'Times (choose a time column)'}</option>
            </select>
            <input
              className="input-base manual-input"
              aria-label="From"
              placeholder={typedBy === 'index' ? 'From row (blank: first)' : 'From (blank: start)'}
              value={typed.from}
              onChange={(e) => setTyped({ ...typed, from: e.target.value })}
            />
            <input
              className="input-base manual-input"
              aria-label="To"
              placeholder={typedBy === 'index' ? 'To row (blank: last)' : 'To (blank: end)'}
              value={typed.to}
              onChange={(e) => setTyped({ ...typed, to: e.target.value })}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addTyped() } }}
            />
            <button type="button" className="btn-ghost" onClick={addTyped}>
              Add as {PAINT_SETS.find((s) => s.id === paint).label}
            </button>
          </div>
          {typedError && <p className="manual-error">{typedError}</p>}
          {typedBy === 'time' && series?.x_kind !== 'number' && (
            <p className="manual-hint">Times like 2024-01-31 or 2024-01-31 06:30; both ends are included.</p>
          )}

          {ranges.length > 0 ? (
            <ol className="split-ranges">
              {ranges.map((r, i) => (
                <li key={i} className="split-range">
                  <span className={`split-dot split-${r.set}`} />
                  <strong>{PAINT_SETS.find((s) => s.id === r.set)?.label ?? r.set}</strong>
                  <span>{r.by === 'time' ? `${timeCol || 'time'} ` : ''}{rangeText(r)}</span>
                  <button
                    type="button"
                    className="manual-link"
                    aria-label={`Remove range ${i + 1}`}
                    onClick={() => setField('ranges', ranges.filter((_, j) => j !== i))}
                  >
                    ✕
                  </button>
                </li>
              ))}
            </ol>
          ) : (
            <p className="manual-hint">No ranges yet.</p>
          )}
          {ranges.length > 1 && <p className="manual-hint">Where ranges overlap, the later one wins.</p>}
          {ranges.length > 0 && (
            <button type="button" className="manual-link" onClick={() => { setField('ranges', []); setField('rest', '') }}>
              Clear all ranges
            </button>
          )}

          {ranges.length > 0 && uncovered !== null && (uncovered > 0 || rest) && (
            <div className="split-rest">
              <span>
                {uncovered > 0
                  ? `${formatCount(uncovered)} row${uncovered === 1 ? '' : 's'} with a target ${uncovered === 1 ? 'is' : 'are'} in no range.`
                  : 'Every row is in a range.'}
              </span>
              <label className="manual-label" htmlFor="split-rest">Rows in no range</label>
              <select id="split-rest" className="input-base manual-input" value={rest} onChange={(e) => setField('rest', e.target.value)}>
                <option value="">Leave them out</option>
                {SPLIT_SETS.map((s) => <option key={s.id} value={s.id}>Put them in {s.label}</option>)}
              </select>
              {uncovered > 0 && missing.filter((s) => s.id !== rest).map((s) => (
                <button
                  key={s.id}
                  type="button"
                  className={missing.length === 1 ? 'btn-signal' : 'btn-ghost'}
                  onClick={() => setField('rest', s.id)}
                >
                  Use the remaining {formatCount(uncovered)} rows as {s.label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="split-preview" aria-live="polite">
        {preview.error ? (
          <p className="manual-error">{preview.error}</p>
        ) : counts ? (
          <>
            <p className="manual-label">This split would give{preview.loading ? ' (updating…)' : ''}</p>
            <CountsLine
              counts={counts}
              extra={preview.data.no_target ? <span className="manual-hint">({formatCount(preview.data.no_target)} rows have no target and go to no set)</span> : null}
            />
            <SplitBar counts={counts} />
            {preview.data.problem && <p className="manual-error">{preview.data.problem}</p>}
          </>
        ) : preview.loading ? (
          <p className="manual-hint">Working out the sets…</p>
        ) : null}
      </div>

      <div className="manual-actions">
        <button
          type="button"
          className="btn-signal"
          onClick={save}
          disabled={busy || !schema || !datasetVersionId || Boolean(preview.data?.problem) || (custom && !ranges.length)}
        >
          {running ? <>Saving… <Elapsed /></> : saved ? 'Save as the new split' : 'Save the split'}
        </button>
        <button type="button" className="btn-ghost" onClick={reset} disabled={!edited || running}>Reset to defaults</button>
        {!datasetVersionId ? (
          <span className="manual-hint">Upload a dataset in Step 1 first.</span>
        ) : assistantBusy ? (
          <span className="manual-hint">The assistant is working on this dataset; wait for it to finish.</span>
        ) : (
          <span className="manual-hint">Saving replaces the split training uses; models trained before keep their own rows.</span>
        )}
      </div>
      {status && <p className={status.ok ? 'manual-status' : 'manual-status error'} role="status">{status.text}</p>}
    </section>
  )
}
