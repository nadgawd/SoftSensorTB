import { useEffect, useRef, useState } from 'react'
import { BarChart3, ChartLine, PencilLine, Sigma, Trash2 } from 'lucide-react'
import { errorDetail } from '../api/client'
import Elapsed from './Elapsed'
import { RolePicker } from './VariablesTable'

/**
 * Actions on one column, opened from its header in the Data preview.
 * Each action runs the same tool the Tools panel would, so it is logged and undoable.
 */
export default function ColumnMenu({ column, anchor, actions, onClose }) {
  const ref = useRef(null)
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)
  const kind = actions.kinds.get(column)
  const numeric = kind === 'numeric' || kind === 'boolean'
  const time = actions.timeColumn

  useEffect(() => {
    function onDown(e) { if (!ref.current?.contains(e.target)) onClose() }
    function onKey(e) { if (e.key === 'Escape') onClose() }
    document.addEventListener('pointerdown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [onClose])

  async function run(key, tool, args, { closeAfter = true } = {}) {
    setBusy(key)
    setError(null)
    try {
      await actions.run(tool, args)
      if (closeAfter) onClose()
    } catch (err) {
      setError(errorDetail(err, 'The action failed.'))
    } finally {
      setBusy(null)
    }
  }

  const top = Math.min(anchor.bottom + 4, window.innerHeight - 330)
  const left = Math.min(anchor.left, window.innerWidth - 270)
  const item = (key, Icon, label, onClick, disabled, title) => (
    <button type="button" className="colmenu-item" disabled={Boolean(busy) || actions.busy || disabled} onClick={onClick} title={title}>
      <Icon size={14} aria-hidden="true" /> {busy === key ? <>Working… <Elapsed /></> : label}
    </button>
  )

  return (
    <div ref={ref} className="colmenu" style={{ top, left }} role="menu" aria-label={`Actions for ${column}`}>
      <p className="colmenu-title">{column}{kind ? <span> · {kind}</span> : null}</p>
      {item('time', ChartLine, 'Plot over time',
        () => run('time', 'generate_custom_plot', { plot_type: 'time_series', x_column: time, y_columns: [column] }),
        !time || !numeric || column === time, !time ? 'Pick a time column in the Variables table first' : undefined)}
      {item('hist', BarChart3, 'Histogram',
        () => run('hist', 'generate_custom_plot', { plot_type: 'histogram', x_column: column }), !numeric)}
      {item('stats', Sigma, 'Statistics',
        () => run('stats', 'get_column_statistics', { columns: [column] }))}
      {item('rename', PencilLine, 'Rename…', () => {
        const name = window.prompt(`New name for "${column}"`, column)?.trim()
        if (name && name !== column) run('rename', 'rename_columns', { column_mapping: { [column]: name } })
      })}
      {item('drop', Trash2, 'Drop column', () => {
        if (window.confirm(`Drop the column "${column}"? You can undo it afterwards.`)) {
          run('drop', 'drop_columns', { columns: [column] })
        }
      })}
      <div className="colmenu-roles">
        <span>Role</span>
        <RolePicker name={column} kind={kind} role={actions.roleOf(column)} onChange={(r) => actions.setRole(column, r)} />
      </div>
      {error && <p className="manual-error colmenu-error">{error}</p>}
    </div>
  )
}
