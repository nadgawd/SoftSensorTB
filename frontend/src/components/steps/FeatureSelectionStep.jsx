import ActionChips from '../ActionChips'

/**
 * Step 4 — Feature Selection
 */
export default function FeatureSelectionStep({
  tablePreview, targetVariable, selectedFeatures, onFeaturesChange,
  actions, tips, onAction,
}) {
  const hasData = tablePreview?.length > 0
  const columns = hasData ? Object.keys(tablePreview[0]) : []

  function toggleFeature(col) {
    if (col === targetVariable) return
    const current = selectedFeatures || []
    if (current.includes(col)) {
      onFeaturesChange?.(current.filter((c) => c !== col))
    } else {
      onFeaturesChange?.([...current, col])
    }
  }

  function selectAll() {
    onFeaturesChange?.(columns.filter((c) => c !== targetVariable))
  }
  function clearAll() { onFeaturesChange?.([]) }

  if (!hasData) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--color-mist)' }}>
          <p style={{ fontSize: '1.5rem', marginBottom: '10px' }}>🎯</p>
          <p style={{ fontSize: '0.875rem' }}>Upload a dataset to select feature variables.</p>
        </div>
        <ActionChips actions={actions} tips={tips} onAction={onAction} />
      </div>
    )
  }

  const features = selectedFeatures || []

  return (
    <div className="animate-rise" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Dynamic tips & actions */}
      <ActionChips actions={actions} tips={tips} onAction={onAction} />

      {/* Header row */}
      <div className="panel" style={{ borderRadius: '14px', padding: '14px 18px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <div>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: 0 }}>
              Feature Variables (X)
            </p>
            <p style={{ fontSize: '0.9375rem', fontWeight: 600, color: 'var(--color-foam)', margin: '4px 0 0' }}>
              {features.length} selected of {columns.length - (targetVariable ? 1 : 0)} available
            </p>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button type="button" onClick={selectAll} className="btn-ghost" style={{ padding: '5px 10px', fontSize: '12px' }}>Select all</button>
            <button type="button" onClick={clearAll} className="btn-ghost" style={{ padding: '5px 10px', fontSize: '12px' }}>Clear</button>
          </div>
        </div>
        {targetVariable && (
          <p style={{ fontSize: '12px', color: 'var(--color-mist)', margin: 0 }}>
            Target: <span className="badge badge-signal">{targetVariable}</span> (excluded from features)
          </p>
        )}
      </div>

      {/* Column selection grid */}
      <div className="panel" style={{ borderRadius: '14px', padding: '14px', maxHeight: '380px', overflowY: 'auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '8px' }}>
          {columns.map((col) => {
            const isTarget = col === targetVariable
            const isSelected = features.includes(col)
            return (
              <button
                key={col}
                type="button"
                disabled={isTarget}
                onClick={() => toggleFeature(col)}
                style={{
                  padding: '10px 14px',
                  borderRadius: '10px',
                  textAlign: 'left',
                  border: `1px solid ${isTarget ? 'rgba(46,196,182,0.4)' : isSelected ? 'rgba(212,165,53,0.5)' : 'var(--color-line)'}`,
                  background: isTarget ? 'rgba(46,196,182,0.08)' : isSelected ? 'rgba(212,165,53,0.08)' : 'var(--feature-card-bg)',
                  cursor: isTarget ? 'default' : 'pointer',
                  transition: 'all 150ms ease',
                }}
              >
                <p style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', margin: 0, color: isTarget ? 'var(--color-signal)' : isSelected ? 'var(--color-ember)' : 'var(--color-foam)', wordBreak: 'break-all' }}>
                  {col}
                </p>
                <p style={{ fontSize: '10px', margin: '3px 0 0', color: 'var(--color-mist)' }}>
                  {isTarget ? 'Target (y)' : isSelected ? '✓ Feature (x)' : 'Click to add'}
                </p>
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
