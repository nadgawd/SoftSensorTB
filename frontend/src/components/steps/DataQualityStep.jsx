import { useMemo } from 'react'
import ActionChips from '../ActionChips'

/**
 * Step 2 — Data Quality
 */
export default function DataQualityStep({ tablePreview, fileName, actions, tips, onAction }) {
  const stats = useMemo(() => {
    if (!tablePreview?.length) return null
    const columns = Object.keys(tablePreview[0])
    const numericCols = []
    const textCols = []

    for (const col of columns) {
      const vals = tablePreview.map((r) => r[col]).filter((v) => v !== null && v !== undefined)
      const numeric = vals.every((v) => typeof v === 'number' || (!isNaN(Number(v)) && v !== ''))
      if (numeric) numericCols.push(col)
      else textCols.push(col)
    }

    const nullCounts = {}
    for (const col of columns) {
      nullCounts[col] = tablePreview.filter((r) => r[col] === null || r[col] === undefined || r[col] === '').length
    }

    const usable = columns.filter((c) => nullCounts[c] < tablePreview.length * 0.3)
    const issues = columns.filter((c) => nullCounts[c] >= tablePreview.length * 0.1 && nullCounts[c] < tablePreview.length * 0.3)
    const unusable = columns.filter((c) => nullCounts[c] >= tablePreview.length * 0.3)

    return { columns, numericCols, textCols, nullCounts, usable, issues, unusable }
  }, [tablePreview])

  if (!tablePreview?.length) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--color-mist)' }}>
          <p style={{ fontSize: '1.5rem', marginBottom: '10px' }}>🔍</p>
          <p style={{ fontSize: '0.875rem' }}>Upload a dataset in Step 1 to see data quality assessment here.</p>
        </div>
        <ActionChips actions={actions} tips={tips} onAction={onAction} />
      </div>
    )
  }

  return (
    <div className="animate-rise" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Dynamic tips */}
      <ActionChips actions={actions} tips={tips} onAction={onAction} />

      {/* Quality summary badges */}
      <div className="panel" style={{ borderRadius: '14px', padding: '16px 20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: 0 }}>
            Quality Assessment — {stats?.columns.length} tags
          </p>
          <span className="badge badge-mist">{fileName || 'dataset'}</span>
        </div>
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          <QualityBadge count={stats?.usable.length} label="Usable" color="var(--color-success)" />
          <QualityBadge count={stats?.issues.length} label="Have issues" color="var(--color-warning)" />
          <QualityBadge count={stats?.unusable.length} label="Unusable" color="var(--color-alert)" />
        </div>
      </div>

      {/* Column breakdown */}
      <div className="panel" style={{ borderRadius: '14px', overflow: 'hidden' }}>
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--color-line)' }}>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: 0 }}>
            Column Quality Report
          </p>
        </div>
        <div style={{ overflowY: 'auto', maxHeight: '340px' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Tag name</th>
                <th>Type</th>
                <th>Null %</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {stats?.columns.map((col) => {
                const nullPct = ((stats.nullCounts[col] / tablePreview.length) * 100).toFixed(1)
                const isNumeric = stats.numericCols.includes(col)
                const status = stats.usable.includes(col)
                  ? { label: 'Usable', cls: 'badge-success' }
                  : stats.issues.includes(col)
                    ? { label: 'Issues', cls: 'badge-ember' }
                    : { label: 'Unusable', cls: 'badge-alert' }
                return (
                  <tr key={col}>
                    <td style={{ color: 'var(--color-foam)', fontWeight: 500 }}>{col}</td>
                    <td><span className="badge badge-mist" style={{ fontSize: '10px' }}>{isNumeric ? 'Numeric' : 'Text'}</span></td>
                    <td style={{ color: Number(nullPct) > 20 ? 'var(--color-alert)' : 'inherit' }}>{nullPct}%</td>
                    <td><span className={`badge ${status.cls}`} style={{ fontSize: '10px' }}>{status.label}</span></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function QualityBadge({ count, label, color }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: color, flexShrink: 0 }} />
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1.1rem', fontWeight: 600, color }}>{count ?? 0}</span>
      <span style={{ fontSize: '0.8125rem', color: 'var(--color-mist)' }}>{label}</span>
    </div>
  )
}
