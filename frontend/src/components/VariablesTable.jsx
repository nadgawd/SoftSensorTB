import { useMemo, useState } from 'react'
import { formatCount } from '../utils/format'

const ROLES = [
  { id: 'time', label: 'Time', hint: 'The timestamp of each row: used for gaps, resampling and time-ordered splits' },
  { id: 'target', label: 'Target', hint: 'The lab value the soft sensor predicts (y)' },
  { id: 'feature', label: 'Feature', hint: 'A model input (x)' },
  { id: 'ignore', label: 'Ignore', hint: 'Kept in the data but not used by the model' },
]

function formatNumber(v) {
  if (v === null || v === undefined) return '—'
  const a = Math.abs(v)
  return a !== 0 && (a >= 1e5 || a < 1e-3) ? v.toExponential(2) : Number(v.toPrecision(4)).toString()
}

/**
 * One place to give every column its role. The roles are the defaults of every tool form:
 * the target, the features (none chosen = every numeric column except the target and time) and the time column.
 */
export default function VariablesTable({
  profile, previewColumns, timeColumn, targetVariable, selectedFeatures,
  onTimeColumnChange, onTargetSelect, onFeaturesChange,
}) {
  const [filter, setFilter] = useState('')
  const columns = useMemo(() => {
    if (profile?.data?.columns) return profile.data.columns
    return (previewColumns || []).map((name) => ({ name, kind: null }))
  }, [profile, previewColumns])

  const numeric = useMemo(() => new Set(columns.filter((c) => c.kind === 'numeric').map((c) => c.name)), [columns])
  const explicit = (selectedFeatures || []).length > 0
  const implicitFeatures = useMemo(
    () => columns.map((c) => c.name).filter((n) => numeric.has(n) && n !== targetVariable && n !== timeColumn),
    [columns, numeric, targetVariable, timeColumn],
  )
  const features = explicit ? selectedFeatures.filter((f) => f !== targetVariable && f !== timeColumn) : implicitFeatures

  function roleOf(name) {
    if (name === timeColumn) return 'time'
    if (name === targetVariable) return 'target'
    return features.includes(name) ? 'feature' : 'ignore'
  }

  function setRole(name, role) {
    const current = roleOf(name)
    if (role === current) return
    if (current === 'target') onTargetSelect(null)
    const base = features.filter((f) => f !== name)
    if (role === 'time') onTimeColumnChange(name)
    if (role === 'target') onTargetSelect(name)
    if (role === 'feature') onFeaturesChange([...base, name])
    else if (role === 'ignore' || (explicit && current === 'feature')) onFeaturesChange(base)
  }

  const shown = filter
    ? columns.filter((c) => c.name.toLowerCase().includes(filter.toLowerCase()))
    : columns

  return (
    <section className="panel vars-panel" aria-label="Variables">
      <div className="vars-head">
        <div>
          <p className="model-panel-title">Variables</p>
          <p className="vars-summary">
            {targetVariable ? <>Target <strong>{targetVariable}</strong></> : <span className="quality-warn">No target yet</span>}
            {' · '}{features.length} feature{features.length === 1 ? '' : 's'}{!explicit && features.length > 0 ? ' (every numeric tag)' : ''}
            {' · '}{timeColumn ? <>time <strong>{timeColumn}</strong></> : 'no time column'}
          </p>
        </div>
        <div className="vars-actions">
          {columns.length > 12 && (
            <input
              className="manual-input vars-filter"
              placeholder="Find a tag…"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              aria-label="Find a tag"
            />
          )}
          {explicit && (
            <button type="button" className="btn-ghost vars-reset" onClick={() => onFeaturesChange([])}
              title="Use every numeric tag except the target and time as a feature">
              All numeric as features
            </button>
          )}
        </div>
      </div>
      <div className="manual-table-wrap vars-table-wrap">
        <table className="data-table">
          <thead>
            <tr><th>Tag</th><th>Type</th><th>Missing</th><th>Range</th><th>Role</th></tr>
          </thead>
          <tbody>
            {shown.map((c) => {
              const role = roleOf(c.name)
              return (
                <tr key={c.name} className={`vars-row role-${role}`}>
                  <td className="quality-name">{c.name}</td>
                  <td>{c.kind || '…'}</td>
                  <td>{c.missing ? `${formatCount(c.missing)} (${(c.missing_share * 100).toFixed(1)}%)` : c.kind ? '—' : '…'}</td>
                  <td>{c.min != null ? `${formatNumber(c.min)} – ${formatNumber(c.max)}` : '—'}</td>
                  <td>
                    <div className="vars-roles" role="radiogroup" aria-label={`Role of ${c.name}`}>
                      {ROLES.map((r) => {
                        const disabled = Boolean(c.kind) && (
                          (r.id === 'feature' && c.kind !== 'numeric' && c.kind !== 'boolean')
                          || (r.id === 'target' && c.kind === 'time')
                          || (r.id === 'time' && c.kind !== 'time' && c.kind !== 'text'))
                        const locked = role === 'time' && r.id !== 'time'
                        return (
                          <button
                            key={r.id}
                            type="button"
                            role="radio"
                            aria-checked={role === r.id}
                            disabled={disabled || locked}
                            title={locked ? 'Pick another column as Time to change it' : disabled ? `Not available for a ${c.kind} column` : r.hint}
                            className={`vars-role${role === r.id ? ` active ${r.id}` : ''}`}
                            onClick={() => setRole(c.name, r.id)}
                          >
                            {r.label}
                          </button>
                        )
                      })}
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
