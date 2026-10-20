import { useEffect, useState } from 'react'
import { Undo2, X } from 'lucide-react'
import { describeDiff } from '../utils/versions'

const VISIBLE_MS = 10_000

/** "What changed" after a run, with Undo. Hides itself after a few seconds unless hovered. */
export default function ChangeToast({ diff, label, onUndo, onClose }) {
  const [hover, setHover] = useState(false)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (hover) return undefined
    const timer = setTimeout(onClose, VISIBLE_MS)
    return () => clearTimeout(timer)
  }, [hover, onClose, diff])

  const summary = describeDiff(diff) || 'No visible change to rows or column values'
  const names = [
    ...diff.added.map((c) => `+ ${c}`),
    ...diff.removed.map((c) => `− ${c}`),
    ...diff.changed.map((c) => `~ ${c}`),
  ]

  return (
    <div className="change-toast" role="status" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
      <div className="change-toast-text">
        <strong>{label}</strong>
        <span>{summary}</span>
        {names.length > 0 && (
          <span className="change-toast-cols" title={names.join('\n')}>
            {names.slice(0, 4).join('  ')}{names.length > 4 ? `  +${names.length - 4} more` : ''}
          </span>
        )}
      </div>
      <button
        type="button"
        className="history-btn"
        disabled={busy}
        onClick={async () => {
          setBusy(true)
          try { await onUndo() } finally { setBusy(false) }
        }}
      >
        <Undo2 size={13} aria-hidden="true" /> Undo
      </button>
      <button type="button" className="change-toast-close" onClick={onClose} aria-label="Dismiss">
        <X size={14} />
      </button>
    </div>
  )
}
