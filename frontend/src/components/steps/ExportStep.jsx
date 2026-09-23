import ActionChips from '../ActionChips'

/**
 * Step 7 — Export
 */
export default function ExportStep({ fileName, modelMetrics, tablePreview, datasetVersionId, actions, tips, onAction }) {
  const hasModel = Boolean(modelMetrics)
  const hasData = Boolean(tablePreview?.length)

  function downloadSummary() {
    const summary = {
      file: fileName,
      dataset_version_id: datasetVersionId,
      rows: tablePreview?.length,
      model: modelMetrics,
      exported_at: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(summary, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `soft-sensor-export-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="animate-rise" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Dynamic tips & actions */}
      <ActionChips actions={actions} tips={tips} onAction={onAction} />

      {/* Export summary card */}
      <div className="panel" style={{ borderRadius: '14px', padding: '18px 22px' }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: '0 0 14px' }}>
          Project Summary
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '18px' }}>
          <SummarySection label="Data" items={[
            { key: 'Source file', val: fileName || '—' },
            { key: 'Rows processed', val: tablePreview?.length ? `${tablePreview.length} (preview)` : '—' },
            { key: 'Active version', val: datasetVersionId ? datasetVersionId.slice(0, 8) + '…' : '—' },
          ]} />
          <SummarySection label="Model" items={[
            { key: 'Algorithm', val: modelMetrics?.algorithm || '—' },
            { key: 'R²', val: modelMetrics?.r2_score != null ? Number(modelMetrics.r2_score).toFixed(4) : '—' },
            { key: 'RMSE', val: modelMetrics?.rmse != null ? Number(modelMetrics.rmse).toFixed(4) : '—' },
          ]} />
        </div>

        {/* Checklist */}
        <div style={{ borderTop: '1px solid var(--color-line)', paddingTop: '14px', display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
          <CheckItem done={Boolean(fileName)} label="Dataset uploaded" />
          <CheckItem done={hasData} label="Data preview available" />
          <CheckItem done={hasModel} label="Soft sensor trained" />
          <CheckItem done={hasModel && modelMetrics?.r2_score > 0.5} label="R² > 0.5 (recommended)" />
        </div>

        {/* Export options + button */}
        <div style={{ borderTop: '1px solid var(--color-line)', paddingTop: '14px' }}>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '14px' }}>
            <ExportOption label="Decision log" format="JSON" checked={true} />
            <ExportOption label="Model coefficients" format="JSON" checked={hasModel} />
            <ExportOption label="Prepared dataset" format="Parquet" checked={hasData} />
          </div>
          <button
            type="button"
            className="btn-primary"
            onClick={downloadSummary}
            disabled={!hasData && !hasModel}
            style={{ padding: '10px 22px', fontSize: '0.9rem' }}
          >
            Export summary JSON →
          </button>
        </div>
      </div>
    </div>
  )
}

function SummarySection({ label, items }) {
  return (
    <div>
      <p style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-mist)', marginBottom: '8px', fontWeight: 500 }}>
        {label}
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {items.map(({ key, val }) => (
          <div key={key} style={{ display: 'flex', gap: '8px', fontSize: '0.8125rem' }}>
            <span style={{ color: 'var(--color-mist)', minWidth: '120px', flexShrink: 0 }}>{key}</span>
            <span style={{ color: 'var(--color-foam)', fontFamily: 'var(--font-mono)', fontSize: '12px', wordBreak: 'break-all' }}>{val}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function ExportOption({ label, format, checked }) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'default', opacity: checked ? 1 : 0.4 }}>
      <input type="checkbox" checked={checked} readOnly style={{ accentColor: 'var(--color-signal)' }} />
      <span style={{ fontSize: '0.8125rem', color: 'var(--color-foam)' }}>{label}</span>
      <span style={{ fontSize: '11px', color: 'var(--color-mist)', fontFamily: 'var(--font-mono)' }}>{format}</span>
    </label>
  )
}

function CheckItem({ done, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.875rem' }}>
      <span style={{
        width: '18px', height: '18px', borderRadius: '50%',
        background: done ? 'var(--color-success)' : 'transparent',
        border: `2px solid ${done ? 'var(--color-success)' : 'var(--color-line-2)'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0, fontSize: '10px', color: '#fff',
      }}>
        {done ? '✓' : ''}
      </span>
      <span style={{ color: done ? 'var(--color-foam)' : 'var(--color-mist)' }}>{label}</span>
    </div>
  )
}
