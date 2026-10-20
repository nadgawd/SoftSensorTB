/**
 * Step 3 — Preprocessing
 */
export default function PreprocessingStep({ tablePreview, datasetVersionId }) {
  const hasData = tablePreview?.length > 0

  if (!hasData) {
    return (
      <div style={{ textAlign: 'center', padding: '30px 20px', color: 'var(--color-mist)' }}>
        <p style={{ fontSize: '1.5rem', marginBottom: '8px' }}>🔧</p>
        <p style={{ fontSize: '0.875rem' }}>Upload a dataset in Step 1 to enable preprocessing tools.</p>
      </div>
    )
  }

  return (
    <div className="animate-rise panel" style={{ borderRadius: '14px', padding: '12px 18px' }}>
      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap', alignItems: 'center' }}>
        <StatItem label="Rows shown" value={tablePreview.length} />
        <StatItem label="Columns" value={Object.keys(tablePreview[0]).length} />
        <StatItem label="Status" value={datasetVersionId ? '✓ Active version' : '—'} valueColor="var(--color-signal)" />
        <p className="manual-hint" style={{ margin: 0, flex: '1 1 220px' }}>
          Check the effect of each step in the Data preview and Graph tabs at the bottom.
        </p>
      </div>
    </div>
  )
}

function StatItem({ label, value, valueColor }) {
  return (
    <div>
      <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: '0 0 2px' }}>
        {label}
      </p>
      <p style={{ fontSize: '1.1rem', fontWeight: 600, color: valueColor || 'var(--color-foam)', margin: 0, fontFamily: 'var(--font-mono)' }}>
        {value}
      </p>
    </div>
  )
}
