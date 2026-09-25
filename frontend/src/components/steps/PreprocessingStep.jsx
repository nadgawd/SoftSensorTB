import PlotCanvas from '../PlotCanvas'
import ActionChips from '../ActionChips'

/**
 * Step 3 — Preprocessing
 */
export default function PreprocessingStep({ tablePreview, plotHistory, datasetVersionId, actions, tips, onAction }) {
  const hasData = tablePreview?.length > 0

  return (
    <div className="animate-rise" style={{ display: 'flex', flexDirection: 'column', gap: '16px', minHeight: '100%' }}>
      {/* Dynamic tips + clickable actions */}
      <ActionChips actions={actions} tips={tips} onAction={onAction} />

      {/* Status bar */}
      {hasData && (
        <div className="panel" style={{ borderRadius: '14px', padding: '12px 18px' }}>
          <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
            <StatItem label="Rows" value={tablePreview.length} />
            <StatItem label="Columns" value={Object.keys(tablePreview[0]).length} />
            <StatItem
              label="Status"
              value={datasetVersionId ? '✓ Active version' : '—'}
              valueColor="var(--color-signal)"
            />
          </div>
        </div>
      )}

      {/* Plot area */}
      <PlotCanvas
        label="Visualization Canvas"
        plotHistory={plotHistory}
        emptyState={
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--color-mist)', gap: '8px', padding: '30px' }}>
            <p style={{ fontSize: '1.5rem' }}>📈</p>
            <p style={{ fontSize: '0.875rem', textAlign: 'center', color: 'var(--color-mist)' }}>
              Click a suggested action above to generate a chart
            </p>
          </div>
        }
      />

      {!hasData && (
        <div style={{ textAlign: 'center', padding: '30px 20px', color: 'var(--color-mist)' }}>
          <p style={{ fontSize: '1.5rem', marginBottom: '8px' }}>🔧</p>
          <p style={{ fontSize: '0.875rem' }}>Upload a dataset in Step 1 to enable preprocessing tools.</p>
        </div>
      )}
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
