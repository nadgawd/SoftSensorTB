import { useEffect, useMemo, useRef, useState } from 'react'
import { Search } from 'lucide-react'
import { MODEL_TABS, STEP_BY_ID, uiPlaceOf } from '../constants/steps'
import { shortcutLabel } from '../utils/keys'

const STEP_ORDER = Object.keys(STEP_BY_ID)

function stepName(tool) {
  if (!tool.steps[0]) return 'Data preview › Graph'
  const { step, modelTab } = uiPlaceOf(tool.steps[0])
  const label = STEP_BY_ID[step]?.label || step
  return modelTab ? `${label} › ${MODEL_TABS.find((t) => t.id === modelTab).label}` : label
}

function stepIndex(tool) {
  return tool.steps[0] ? STEP_ORDER.indexOf(uiPlaceOf(tool.steps[0]).step) : STEP_ORDER.length
}

function score(tool, q) {
  const label = tool.label.toLowerCase()
  if (label.startsWith(q)) return 0
  if (label.includes(q)) return 1
  if (tool.name.includes(q.replace(/\s+/g, '_'))) return 2
  if ((tool.summary || '').toLowerCase().includes(q)) return 3
  if (stepName(tool).toLowerCase().includes(q)) return 4
  return null
}

/** Ctrl+K: find any tool by name or what it does and open it in its step. */
export default function ToolPalette({ tools, recent, onPick, onClose }) {
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)
  const listRef = useRef(null)

  const items = useMemo(() => {
    const q = query.trim().toLowerCase()
    const all = tools || []
    if (q) {
      return all
        .map((t) => ({ t, s: score(t, q) }))
        .filter((x) => x.s !== null)
        .sort((a, b) => a.s - b.s || a.t.label.localeCompare(b.t.label))
        .map((x) => ({ tool: x.t }))
    }
    const byName = new Map(all.map((t) => [t.name, t]))
    const recentTools = recent.map((n) => byName.get(n)).filter(Boolean)
    const rest = all
      .filter((t) => !recent.includes(t.name))
      .sort((a, b) => stepIndex(a) - stepIndex(b) || stepName(a).localeCompare(stepName(b)) || a.label.localeCompare(b.label))
    return [
      ...recentTools.map((tool, i) => ({ tool, group: i === 0 ? 'Recently used' : null })),
      ...rest.map((tool, i) => ({ tool, group: i === 0 ? 'All tools' : null })),
    ]
  }, [tools, recent, query])

  const active = Math.min(index, Math.max(items.length - 1, 0))

  useEffect(() => {
    listRef.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [active])

  function onKeyDown(e) {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setIndex(Math.min(active + 1, items.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setIndex(Math.max(active - 1, 0))
    } else if (e.key === 'Enter' && items[active]) {
      e.preventDefault()
      onPick(items[active].tool)
    } else if (e.key === 'Escape') {
      onClose()
    }
  }

  return (
    <div className="palette-backdrop" onMouseDown={onClose}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Find a tool" onMouseDown={(e) => e.stopPropagation()}>
        <div className="palette-search">
          <Search size={15} aria-hidden="true" />
          <input
            autoFocus
            className="palette-input"
            placeholder="Find a tool: missing, lag, outlier, PLS, parity…"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setIndex(0)
            }}
            onKeyDown={onKeyDown}
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-list"
          />
          <kbd className="palette-kbd">Esc</kbd>
        </div>
        <ul className="palette-list" id="palette-list" role="listbox" ref={listRef}>
          {!tools && <li className="palette-empty">Loading tools…</li>}
          {tools && !items.length && <li className="palette-empty">No tool matches “{query}”.</li>}
          {items.map(({ tool, group }, i) => (
            <li key={tool.name} role="presentation">
              {group && <p className="palette-group">{group}</p>}
              <button
                type="button"
                role="option"
                aria-selected={i === active}
                className="palette-item"
                onMouseMove={() => i !== active && setIndex(i)}
                onClick={() => onPick(tool)}
              >
                <span className="palette-item-main">
                  <span className="palette-item-label">{tool.label}</span>
                  <span className="palette-item-summary">{tool.summary}</span>
                </span>
                <span className="palette-item-step">{stepName(tool)}</span>
              </button>
            </li>
          ))}
        </ul>
        <p className="palette-foot">
          <kbd>↑</kbd><kbd>↓</kbd> move · <kbd>Enter</kbd> open · <kbd>{shortcutLabel('Enter')}</kbd> runs the open tool
        </p>
      </div>
    </div>
  )
}
