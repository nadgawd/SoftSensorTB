import { useMemo, useState } from 'react'
import { ROLES, roleAllowed, useColumnRoles } from '../hooks/useColumnRoles'
import { formatCount } from '../utils/format'

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
  const { features, explicit, roleOf, setRole } = useColumnRoles({
    columns, timeColumn, targetVariable, selectedFeatures, onTimeColumnChange, onTargetSelect, onFeaturesChange,
  })

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
                    <RolePicker name={c.name} kind={c.kind} role={role} onChange={(r) => setRole(c.name, r)} />
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

export function RolePicker({ name, kind, role, onChange }) {
  return (
    <div className="vars-roles" role="radiogroup" aria-label={`Role of ${name}`}>
      {ROLES.map((r) => {
        const disabled = !roleAllowed(r.id, kind)
        const locked = role === 'time' && r.id !== 'time'
        return (
          <button
            key={r.id}
            type="button"
            role="radio"
            aria-checked={role === r.id}
            disabled={disabled || locked}
            title={locked ? 'Pick another column as Time to change it' : disabled ? `Not available for a ${kind} column` : r.hint}
            className={`vars-role${role === r.id ? ` active ${r.id}` : ''}`}
            onClick={() => onChange(r.id)}
          >
            {r.label}
          </button>
        )
      })}
    </div>
  )
}
