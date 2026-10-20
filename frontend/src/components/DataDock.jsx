import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import ColumnMenu from './ColumnMenu'
import ManualTools from './manual/ManualTools'
import PlotCanvas from './PlotCanvas'
import { usePersisted } from '../hooks/usePersisted'
import { formatCount } from '../utils/format'

const MIN_HEIGHT = 180
const DEFAULT_HEIGHT = 340
const TABS = [
  { id: 'preview', label: 'Data preview' },
  { id: 'graph', label: 'Graph' },
]

function cell(value) {
  return value === null || value === undefined ? '—' : String(value)
}

function PreviewControls({ meta, shown, limit, onLimitChange, onPageChange }) {
  const total = meta?.rowCount ?? shown
  const offset = meta?.offset ?? 0
  const sample = Boolean(meta?.sample)
  const first = shown ? offset + 1 : 0
  const last = offset + shown

  return (
    <div className="dock-limit">
      <div className="dock-mode" role="group" aria-label="Which rows to show">
        <button type="button" className={sample ? '' : 'active'} onClick={() => onPageChange?.({ offset: 0, sample: false, seed: 0 })}>
          In order
        </button>
        <button
          type="button"
          className={sample ? 'active' : ''}
          title="A random sample of rows from the whole version, in row order"
          onClick={() => onPageChange?.({ offset: 0, sample: true, seed: Math.floor(Math.random() * 1e6) })}
        >
          {sample ? '↻ New sample' : 'Random sample'}
        </button>
      </div>
      {sample ? (
        <span>{formatCount(shown)} random of {formatCount(total)}</span>
      ) : (
        <>
          <button type="button" className="dock-page" disabled={offset === 0} aria-label="Previous rows"
            onClick={() => onPageChange?.({ offset: Math.max(0, offset - limit), sample: false, seed: 0 })}>‹</button>
          <span>{formatCount(first)}–{formatCount(last)} of {formatCount(total)}</span>
          <button type="button" className="dock-page" disabled={last >= total} aria-label="Next rows"
            onClick={() => onPageChange?.({ offset: offset + limit, sample: false, seed: 0 })}>›</button>
        </>
      )}
      <label>
        Per page
        <input
          type="number"
          min="1"
          max="10000"
          value={limit}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10)
            if (!Number.isNaN(val) && val > 0) onLimitChange?.(val)
          }}
        />
      </label>
    </div>
  )
}

/**
 * Bottom dock on every step: the active version's rows, and a Graph tab to plot
 * any columns of it. Drag the top edge to resize; click the open tab to fold it.
 */
export default function DataDock({
  rows,
  previewMeta,
  previewLimit,
  onPreviewLimitChange,
  onPageChange,
  open,
  tab,
  onOpenChange,
  onTabChange,
  plotHistory,
  graphTools,
  columnChanges,
  columnActions,
}) {
  const [menu, setMenu] = useState(null)
  const closeMenu = useCallback(() => setMenu(null), [])
  const added = useMemo(() => new Set(columnChanges?.added || []), [columnChanges])
  const changed = useMemo(() => new Set(columnChanges?.changed || []), [columnChanges])
  const [savedHeight, setSavedHeight] = usePersisted('dockHeight', DEFAULT_HEIGHT)
  const [dragHeight, setDragHeight] = useState(null)
  const drag = useRef(null)
  const columns = rows?.length ? Object.keys(rows[0]) : []
  const maxHeight = Math.round(window.innerHeight * 0.75)
  const height = Math.min(Math.max(dragHeight ?? savedHeight, MIN_HEIGHT), maxHeight)
  const plotCount = plotHistory.plots.length

  // Plotly only re-measures on window resizes.
  useEffect(() => {
    if (dragHeight !== null) return undefined
    const timer = setTimeout(() => window.dispatchEvent(new Event('resize')), 60)
    return () => clearTimeout(timer)
  }, [open, tab, savedHeight, dragHeight])

  const onPointerMove = useCallback((e) => {
    if (drag.current) setDragHeight(drag.current.height + drag.current.y - e.clientY)
  }, [])

  const onPointerUp = useCallback((e) => {
    if (!drag.current) return
    const next = drag.current.height + drag.current.y - e.clientY
    drag.current = null
    setSavedHeight(Math.min(Math.max(next, MIN_HEIGHT), Math.round(window.innerHeight * 0.75)))
    setDragHeight(null)
    window.removeEventListener('pointermove', onPointerMove)
    window.removeEventListener('pointerup', onPointerUp)
  }, [onPointerMove, setSavedHeight])

  function startDrag(e) {
    e.preventDefault()
    drag.current = { y: e.clientY, height }
    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)
  }

  useEffect(() => () => {
    window.removeEventListener('pointermove', onPointerMove)
    window.removeEventListener('pointerup', onPointerUp)
  }, [onPointerMove, onPointerUp])

  function pickTab(id) {
    if (open && tab === id) {
      onOpenChange(false)
      return
    }
    onTabChange(id)
    onOpenChange(true)
  }

  return (
    <div className={`dock${open ? ' open' : ''}`} style={open ? { height } : undefined}>
      {open && (
        <div
          className="dock-resize"
          role="separator"
          aria-orientation="horizontal"
          aria-label="Resize the panel"
          title="Drag to resize · double-click to reset"
          onPointerDown={startDrag}
          onDoubleClick={() => setSavedHeight(DEFAULT_HEIGHT)}
        />
      )}
      <div className="dock-bar">
        <div className="dock-tabs" role="tablist" aria-label="Data panel">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={open && tab === t.id}
              className={`dock-tab${open && tab === t.id ? ' active' : ''}`}
              onClick={() => pickTab(t.id)}
            >
              <span aria-hidden="true">{open && tab === t.id ? '▾' : '▸'}</span>
              {t.label}
              <span className="dock-tab-count">
                {t.id === 'preview'
                  ? `${formatCount(previewMeta?.rowCount ?? rows?.length ?? 0)} rows · ${columns.length} cols`
                  : `${plotCount} plot${plotCount === 1 ? '' : 's'}`}
              </span>
            </button>
          ))}
        </div>
        {open && tab === 'preview' && (
          <PreviewControls
            meta={previewMeta}
            shown={rows?.length ?? 0}
            limit={previewLimit || 25}
            onLimitChange={onPreviewLimitChange}
            onPageChange={onPageChange}
          />
        )}
        {open && (
          <button type="button" className="dock-fold" onClick={() => onOpenChange(false)} aria-label="Fold the panel" title="Fold">
            ▾
          </button>
        )}
      </div>

      {open && tab === 'preview' && (
        <div className="dock-body dock-preview">
          {columns.length ? (
            <table className="data-table">
              <thead>
                <tr>
                  <th className="dock-rownum" title="Row number in this version">#</th>
                  {columns.map((c) => {
                    const change = added.has(c) ? 'added' : changed.has(c) ? 'changed' : null
                    return (
                      <th key={c} className={change ? `dock-col-${change}` : undefined}
                        title={change ? `${change === 'added' ? 'New' : 'Changed'} in the last step` : undefined}>
                        {columnActions ? (
                          <button type="button" className="dock-colbtn" aria-haspopup="menu"
                            onClick={(e) => setMenu({ column: c, anchor: e.currentTarget.getBoundingClientRect() })}>
                            {c}<ChevronDown size={11} aria-hidden="true" />
                          </button>
                        ) : c}
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={previewMeta?.rowNumbers?.[i] ?? i}>
                    <td className="dock-rownum">{(previewMeta?.rowNumbers?.[i] ?? i) + 1}</td>
                    {columns.map((c) => (
                      <td key={c} className={added.has(c) ? 'dock-col-added' : changed.has(c) ? 'dock-col-changed' : undefined}>
                        {cell(row[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="manual-hint dock-empty">Upload a dataset to see its rows here.</p>
          )}
        </div>
      )}

      {open && tab === 'graph' && (
        <div className="dock-body dock-graph">
          <div className="dock-graph-form">
            <ManualTools {...graphTools} fixedTool="generate_custom_plot" activeStep="graph" title="Plot columns of the active version" />
          </div>
          <div className="dock-graph-plot">
            <PlotCanvas
              compact
              label="Graph"
              plotHistory={plotHistory}
              emptyState={(
                <p className="manual-hint dock-empty">
                  Pick a plot type and columns on the left, then press Plot. Every plot, yours or the assistant’s, is kept in the strip above.
                </p>
              )}
            />
          </div>
        </div>
      )}
      {menu && columnActions && (
        <ColumnMenu column={menu.column} anchor={menu.anchor} actions={columnActions} onClose={closeMenu} />
      )}
    </div>
  )
}
