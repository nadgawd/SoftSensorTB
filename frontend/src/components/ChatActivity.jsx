import { useState } from 'react'

const TOOL_LABELS = {
  run_duckdb_query: 'SQL query',
  get_column_statistics: 'Column statistics',
  remove_missing_data: 'Remove nulls',
  remove_outliers: 'Remove outliers',
  normalize_data: 'Normalize',
  create_lag_features: 'Lag features',
  rolling_aggregate: 'Rolling aggregate',
  feature_engineer: 'Feature engineering',
  execute_formula: 'Formula column',
  transform_features: 'Transform features',
  encode_categorical: 'Encode categorical',
  drop_columns: 'Drop columns',
  rename_columns: 'Rename columns',
  drop_collinear_features: 'Drop collinear',
  select_features_rfe: 'RFE selection',
  reduce_dimensions: 'Reduce dimensions',
  balance_data: 'Balance data',
  detect_anomalies: 'Detect anomalies',
  generate_custom_plot: 'Plot',
  generate_parity_plot: 'Parity plot',
  generate_residuals_plot: 'Residuals plot',
  generate_importance_plot: 'Importance plot',
  generate_coefficients_plot: 'Coefficients plot',
  train_soft_sensor: 'Train model',
  list_trained_models: 'List models',
  set_model_features: 'Set features',
}

const toolLabel = (name) => TOOL_LABELS[name] || name

function formatDuration(ms) {
  if (ms == null) return ''
  if (ms < 1000) return `${ms} ms`
  const s = ms / 1000
  return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

/** Group tool steps by name, keeping first-use order, for the collapsed summary. */
function toolCounts(tools) {
  const counts = new Map()
  for (const t of tools) {
    const c = counts.get(t.name) || { name: t.name, n: 0, failed: 0 }
    c.n += 1
    if (t.ok === false) c.failed += 1
    counts.set(t.name, c)
  }
  return [...counts.values()]
}

function ToolChip({ name, n, failed }) {
  const allFailed = failed > 0 && failed === n
  return (
    <span
      className={`badge ${allFailed ? 'badge-alert' : 'badge-mist'}`}
      style={{ fontSize: '10px' }}
      title={failed ? `${failed} of ${n} call(s) failed and were retried or skipped` : undefined}
    >
      ⚙ {toolLabel(name)}{n > 1 ? ` ×${n}` : ''}
    </span>
  )
}

function ToolStep({ step }) {
  const running = step.ok === undefined
  const failed = step.ok === false
  return (
    <div className={`activity-tool${failed ? ' failed' : ''}`}>
      {running ? (
        <span className="spinner" aria-label="running" />
      ) : failed ? (
        <span className="activity-icon-err">✕</span>
      ) : (
        <span className="activity-icon-ok">✓</span>
      )}
      <span className="name">{toolLabel(step.name)}</span>
      <span className="ms">{running ? '' : formatDuration(step.ms)}</span>
      {step.args && <span className="args" title={step.args}>{step.args}</span>}
      {step.summary && <span className="summary">{step.summary}</span>}
    </div>
  )
}

function ReasoningStep({ step, live }) {
  return (
    <div className="activity-reasoning">
      <div className="label">{live ? 'Thinking…' : 'Thought'}</div>
      <div className="text">{step.text}</div>
    </div>
  )
}

function Steps({ steps, live = false }) {
  return steps.map((step, i) => {
    if (step.type === 'reasoning') {
      return <ReasoningStep key={i} step={step} live={live && i === steps.length - 1} />
    }
    if (step.type === 'thought') {
      return <div key={i} className="activity-thought">{step.text}</div>
    }
    return <ToolStep key={step.id || i} step={step} />
  })
}

/** Live log shown while the assistant works: thoughts, tool calls and the current phase. */
export function LiveActivity({ steps, status, elapsedMs, reasoningActive }) {
  return (
    <div className="activity animate-fade-in">
      <div className="activity-steps" style={{ borderTop: 'none' }}>
        <Steps steps={steps} live={reasoningActive} />
        <div className="activity-status">
          <span className="dot-pulse" style={{ display: 'inline-flex' }}>
            <span /><span /><span />
          </span>
          <span>{status || 'Working'}…</span>
          <span style={{ marginLeft: 'auto', fontVariantNumeric: 'tabular-nums' }}>
            {formatDuration(Math.floor(elapsedMs / 1000) * 1000)}
          </span>
        </div>
      </div>
    </div>
  )
}

/** Collapsed summary of a finished turn; expands to the full step log. */
export function ActivitySummary({ steps, durationMs }) {
  const [open, setOpen] = useState(false)
  const tools = steps.filter((s) => s.type === 'tool')
  const reasoned = steps.some((s) => s.type === 'reasoning')
  if (tools.length === 0 && !reasoned && !steps.some((s) => s.type === 'thought')) return null

  const ran = tools.length ? `ran ${tools.length} tool${tools.length === 1 ? '' : 's'}` : ''
  const label = reasoned
    ? `Thought${ran ? ` and ${ran}` : ''}`
    : ran ? ran[0].toUpperCase() + ran.slice(1) : 'Thought'
  return (
    <div className="activity">
      <button
        type="button"
        className="activity-head"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className={`chev${open ? ' open' : ''}`}>▶</span>
        <span>
          {label}
          {durationMs ? ` in ${formatDuration(durationMs)}` : ''}
        </span>
      </button>
      {!open && tools.length > 0 && (
        <div className="activity-chips">
          {toolCounts(tools).map((c) => <ToolChip key={c.name} {...c} />)}
        </div>
      )}
      {open && (
        <div className="activity-steps">
          <Steps steps={steps} />
        </div>
      )}
    </div>
  )
}

/** Chips for messages saved before activity logs existed. */
export function LegacyToolChips({ names }) {
  if (!names?.length) return null
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', paddingLeft: '4px' }}>
      {toolCounts(names.map((name) => ({ name }))).map((c) => <ToolChip key={c.name} {...c} />)}
    </div>
  )
}
