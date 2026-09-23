import { useLayoutEffect, useMemo, useRef } from 'react'
import { ancestorIds, layoutTree } from '../utils/lineage'

const CARD_W = 176
const CARD_H = 58
const COL_W = CARD_W + 44
const ROW_H = CARD_H + 16
const PAD = 8

function shortId(id) {
  return id ? `${id.slice(0, 8)}…` : '—'
}

function parseTime(ts) {
  return ts ? new Date(ts + (ts.endsWith('Z') ? '' : 'Z')) : null
}

/**
 * Version lineage as a left-to-right tree: the original upload is the root
 * and every version sits one column right of the version it was derived from.
 */
export default function VersionTree({ versionHistory, datasetVersionId, disabled, onSelect }) {
  const scrollRef = useRef(null)
  const { nodes, edges, depthCount, slotCount } = useMemo(() => layoutTree(versionHistory), [versionHistory])
  const onActivePath = useMemo(() => ancestorIds(nodes, datasetVersionId), [nodes, datasetVersionId])
  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes])

  const left = (n) => PAD + n.depth * COL_W
  const top = (n) => PAD + n.slot * ROW_H
  const width = PAD * 2 + (depthCount - 1) * COL_W + CARD_W
  const height = PAD * 2 + (slotCount - 1) * ROW_H + CARD_H
  const newestId = nodes[nodes.length - 1]?.id

  // Deep chains overflow sideways; keep the active version in view.
  useLayoutEffect(() => {
    const el = scrollRef.current
    const active = byId.get(datasetVersionId)
    if (!el || !active) return
    const x = left(active)
    if (x + CARD_W > el.scrollLeft + el.clientWidth || x < el.scrollLeft) {
      el.scrollLeft = Math.max(0, x + CARD_W / 2 - el.clientWidth / 2)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetVersionId, byId])

  return (
    <div ref={scrollRef} style={{ overflowX: 'auto', paddingBottom: '4px' }}>
      <div style={{ position: 'relative', width, height }}>
        <svg width={width} height={height} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
          {edges.map(({ from, to }) => {
            const p = byId.get(from)
            const c = byId.get(to)
            const x1 = left(p) + CARD_W
            const y1 = top(p) + CARD_H / 2
            const x2 = left(c)
            const y2 = top(c) + CARD_H / 2
            const mx = (x1 + x2) / 2
            const lit = onActivePath.has(to)
            return (
              <path
                key={`${from}-${to}`}
                d={`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`}
                fill="none"
                stroke={lit ? 'var(--color-signal)' : 'var(--color-line-2)'}
                strokeWidth={lit ? 2.5 : 1.5}
                strokeLinecap="round"
              />
            )
          })}
        </svg>

        {nodes.map((n) => {
          const isActive = n.id === datasetVersionId
          const lit = onActivePath.has(n.id)
          const time = parseTime(n.created_at)
          return (
            <button
              key={n.id}
              type="button"
              disabled={disabled}
              onClick={() => onSelect(n.id)}
              title={[
                n.action_performed,
                n.id,
                time?.toLocaleString(),
                isActive ? 'Active version' : 'Click to time-travel to this version',
              ].filter(Boolean).join('\n')}
              style={{
                position: 'absolute',
                left: left(n),
                top: top(n),
                width: CARD_W,
                height: CARD_H,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                gap: '3px',
                padding: '0 12px',
                textAlign: 'left',
                borderRadius: '10px',
                background: isActive ? 'var(--history-card-active-bg)' : 'var(--color-panel)',
                border: `1px solid ${isActive ? 'var(--history-card-active-border)' : 'var(--color-line)'}`,
                borderLeft: `3px solid ${lit ? 'var(--color-signal)' : 'var(--color-line-2)'}`,
                opacity: lit ? 1 : 0.75,
                cursor: disabled ? 'not-allowed' : isActive ? 'default' : 'pointer',
                transition: 'opacity 150ms ease, border-color 150ms ease',
                overflow: 'hidden',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.opacity = '1' }}
              onMouseLeave={(e) => { e.currentTarget.style.opacity = lit ? '1' : '0.75' }}
            >
              <span
                style={{
                  fontSize: '0.8125rem',
                  fontWeight: 500,
                  color: 'var(--color-foam)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {n.action_performed}
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px', whiteSpace: 'nowrap', fontFamily: 'var(--font-mono)', fontSize: '10px' }}>
                <span style={{ color: 'var(--color-signal)' }}>{shortId(n.id)}</span>
                {time && (
                  <span style={{ color: 'var(--color-mist)' }}>
                    {time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                )}
                {isActive && <span style={{ color: 'var(--color-signal)', fontWeight: 600 }}>● active</span>}
                {!isActive && n.id === newestId && <span style={{ color: 'var(--color-ember)' }}>latest</span>}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
