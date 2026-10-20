/** A short name for the action that produced a version; `labels` maps tool names to their form labels. */
export function actionLabel(action, labels) {
  if (!action || action === 'initial_upload') return 'Upload'
  if (action.startsWith('trained_model_for_')) return 'Train model'
  const label = labels?.get(action)
  if (label) return label.replace(/^\d+ · /, '').replace(/\s*\(.*\)$/, '')
  return action.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase())
}

/** "−70 rows · +2 columns · 1 changed", or null when nothing visible changed. */
export function describeDiff(diff) {
  if (!diff || diff.rows_before == null) return null
  const parts = []
  const dRows = diff.rows_after - diff.rows_before
  if (dRows) parts.push(`${dRows > 0 ? '+' : '−'}${Math.abs(dRows).toLocaleString()} rows`)
  if (diff.added.length) parts.push(`+${diff.added.length} column${diff.added.length > 1 ? 's' : ''}`)
  if (diff.removed.length) parts.push(`−${diff.removed.length} column${diff.removed.length > 1 ? 's' : ''}`)
  if (diff.changed.length) parts.push(`${diff.changed.length} column${diff.changed.length > 1 ? 's' : ''} changed`)
  return parts.length ? parts.join(' · ') : null
}
