import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronRight, History, Redo2, Undo2 } from 'lucide-react'
import { errorDetail, fetchDatasetHistory, rollbackVersion } from '../api/client'
import { parseServerTime } from '../utils/format'
import { isTyping, shortcutLabel } from '../utils/keys'
import { actionLabel } from '../utils/versions'

/**
 * The steps from the upload to the active version. Click one to go back to it;
 * Undo (Ctrl+Z) goes to the previous version and Redo (Ctrl+Shift+Z / Ctrl+Y) forward again.
 */
export default function HistoryStrip({
  versionHistory, datasetVersionId, tools, onRollback, onHistoryRefresh,
  showAll, onToggleAll, disabled,
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const scrollRef = useRef(null)
  const labels = useMemo(() => new Map((tools || []).map((t) => [t.name, t.label])), [tools])

  const { path, parentId, redoId } = useMemo(() => {
    const byId = new Map((versionHistory || []).map((v) => [v.id, v]))
    const out = []
    let node = byId.get(datasetVersionId)
    while (node && out.length < 500) {
      out.unshift(node)
      node = node.parent_version_id ? byId.get(node.parent_version_id) : null
    }
    const children = (versionHistory || [])
      .filter((v) => v.parent_version_id === datasetVersionId)
      .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))
    return { path: out, parentId: out.length > 1 ? out[out.length - 2].id : null, redoId: children[0]?.id || null }
  }, [versionHistory, datasetVersionId])

  const goTo = useCallback(async (versionId) => {
    if (!versionId || versionId === datasetVersionId || busy || disabled) return
    setBusy(true)
    setError(null)
    try {
      const result = await rollbackVersion(versionId)
      onRollback?.(result)
      if (result.dataset_id) onHistoryRefresh?.(await fetchDatasetHistory(result.dataset_id))
    } catch (err) {
      setError(errorDetail(err, 'Could not switch versions.'))
    } finally {
      setBusy(false)
    }
  }, [datasetVersionId, busy, disabled, onRollback, onHistoryRefresh])

  useEffect(() => {
    function onKey(e) {
      if (!(e.ctrlKey || e.metaKey) || e.altKey || isTyping(e.target)) return
      const key = e.key.toLowerCase()
      if (key === 'z' && !e.shiftKey && parentId) {
        e.preventDefault()
        goTo(parentId)
      } else if (((key === 'z' && e.shiftKey) || key === 'y') && redoId) {
        e.preventDefault()
        goTo(redoId)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [parentId, redoId, goTo])

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollLeft = el.scrollWidth
  }, [path.length])

  if (!path.length) return null

  return (
    <div className="history-strip" aria-label="Version history">
      <button type="button" className="history-btn" onClick={() => goTo(parentId)} disabled={!parentId || busy || disabled}
        title={`Undo: go back to the previous version (${shortcutLabel('Z')})`}>
        <Undo2 size={13} aria-hidden="true" /> Undo
      </button>
      <button type="button" className="history-btn history-icon" onClick={() => goTo(redoId)} disabled={!redoId || busy || disabled}
        title={`Redo: go forward again (${shortcutLabel('Shift+Z')})`} aria-label="Redo">
        <Redo2 size={13} aria-hidden="true" />
      </button>
      <ol className="history-path" ref={scrollRef}>
        {path.map((v, i) => {
          const active = v.id === datasetVersionId
          return (
            <li key={v.id}>
              {i > 0 && <ChevronRight size={12} className="history-sep" aria-hidden="true" />}
              <button
                type="button"
                className={`history-chip${active ? ' active' : ''}`}
                disabled={active || busy || disabled}
                onClick={() => goTo(v.id)}
                title={active ? 'The active version' : `Go back to this version (${parseServerTime(v.created_at)?.toLocaleString() ?? ''})`}
              >
                {actionLabel(v.action_performed, labels)}
              </button>
            </li>
          )
        })}
      </ol>
      {error && <span className="history-error" title={error}>{error}</span>}
      <button type="button" className={`history-btn${showAll ? ' active' : ''}`} onClick={onToggleAll}
        title="Every version, including the branches you left">
        <History size={13} aria-hidden="true" /> {showAll ? 'Hide' : `All versions (${versionHistory.length})`}
      </button>
    </div>
  )
}
