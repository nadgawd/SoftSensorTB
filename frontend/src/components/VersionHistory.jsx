import { useCallback, useMemo, useState } from 'react'
import { fetchDatasetHistory, rollbackVersion } from '../api/client'
import { ancestorIds, layoutLineage } from '../utils/lineage'
import VersionTree from './VersionTree'

const ROW_H = 72
const LANE_W = 20
const GUTTER_PAD = 12
const DOT_R = 6
const LANE_COLORS = [
  'var(--color-signal)',
  'var(--color-ember)',
  'var(--color-success)',
  '#a855f7',
  'var(--color-alert)',
  '#0ea5e9',
]

const laneColor = (lane) => LANE_COLORS[lane % LANE_COLORS.length]

function shortId(id) {
  if (!id) return '—'
  return `${id.slice(0, 8)}…`
}

function formatTime(ts) {
  if (!ts) return null
  return new Date(ts + (ts.endsWith('Z') ? '' : 'Z')).toLocaleString()
}

/**
 * Version lineage, shown either as a tree (``view="tree"``) or as a
 * branching timeline. Clicking any version time-travels to it.
 */
export default function VersionHistory({
  view = 'tree',
  versionHistory,
  datasetVersionId,
  onRollback,
  onHistoryRefresh,
}) {
  const [rollingBack, setRollingBack] = useState(false)
  const [error, setError] = useState(null)

  const handleRollback = useCallback(
    async (versionId) => {
      if (!versionId || versionId === datasetVersionId || rollingBack) return
      setRollingBack(true)
      setError(null)
      try {
        const result = await rollbackVersion(versionId)
        onRollback?.(result)
        if (result.dataset_id) {
          const history = await fetchDatasetHistory(result.dataset_id)
          onHistoryRefresh?.(history)
        }
      } catch (err) {
        const detail = err?.response?.data?.detail || err.message || 'Rollback failed'
        setError(typeof detail === 'string' ? detail : JSON.stringify(detail))
      } finally {
        setRollingBack(false)
      }
    },
    [datasetVersionId, rollingBack, onRollback, onHistoryRefresh],
  )

  if (!versionHistory?.length) {
    return (
      <p style={{ fontSize: '0.875rem', color: 'var(--color-mist)', padding: '20px 0' }}>
        Version lineage will appear here after upload and transformations.
      </p>
    )
  }

  const View = view === 'timeline' ? VersionTimeline : VersionTree

  return (
    <div>
      {error && (
        <div className="badge badge-alert" style={{ display: 'block', marginBottom: '12px', padding: '8px 12px', borderRadius: '8px', fontSize: '12px' }}>
          {error}
        </div>
      )}
      <View
        versionHistory={versionHistory}
        datasetVersionId={datasetVersionId}
        disabled={rollingBack}
        onSelect={handleRollback}
      />
    </div>
  )
}

/**
 * Branching timeline: time runs top (newest) to bottom, and every fork from
 * an earlier version opens its own lane to the right.
 */
function VersionTimeline({ versionHistory, datasetVersionId, disabled, onSelect }) {
  const { nodes, edges, laneCount } = useMemo(() => layoutLineage(versionHistory), [versionHistory])
  const onActivePath = useMemo(() => ancestorIds(nodes, datasetVersionId), [nodes, datasetVersionId])

  const total = nodes.length
  const gutter = laneCount * LANE_W + GUTTER_PAD
  const height = total * ROW_H
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const x = (n) => n.lane * LANE_W + LANE_W / 2
  const y = (n) => (total - 1 - n.row) * ROW_H + ROW_H / 2
  const newestId = nodes[total - 1]?.id

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ position: 'relative', height, minWidth: gutter + 260 }}>
        <svg
          width={gutter}
          height={height}
          style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none', zIndex: 1, overflow: 'visible' }}
        >
          {edges.map(({ from, to }) => {
            const p = byId.get(from)
            const c = byId.get(to)
            const px = x(p), py = y(p), cx = x(c), cy = y(c)
            const d = px === cx
              ? `M ${px} ${py} L ${cx} ${cy}`
              : `M ${px} ${py} C ${px} ${py - ROW_H / 2}, ${cx} ${py - ROW_H / 2}, ${cx} ${py - ROW_H} L ${cx} ${cy}`
            const lit = onActivePath.has(to)
            return (
              <path
                key={`${from}-${to}`}
                d={d}
                fill="none"
                stroke={laneColor(c.lane)}
                strokeWidth={lit ? 2.5 : 1.5}
                strokeOpacity={lit ? 1 : 0.35}
                strokeLinecap="round"
              />
            )
          })}
          {nodes.map((n) => {
            const isActive = n.id === datasetVersionId
            const lit = onActivePath.has(n.id)
            const color = laneColor(n.lane)
            return (
              <g key={n.id}>
                <line
                  x1={x(n) + DOT_R}
                  y1={y(n)}
                  x2={gutter}
                  y2={y(n)}
                  stroke={color}
                  strokeOpacity={lit ? 0.5 : 0.2}
                  strokeDasharray="2 3"
                />
                <circle
                  cx={x(n)}
                  cy={y(n)}
                  r={isActive ? DOT_R + 1 : DOT_R}
                  fill={isActive || lit ? color : 'var(--color-panel)'}
                  stroke={color}
                  strokeWidth={2}
                  strokeOpacity={lit ? 1 : 0.6}
                  style={isActive ? { animation: 'pulse-ember 2s ease-in-out infinite' } : undefined}
                />
              </g>
            )
          })}
        </svg>

        {nodes.map((n) => {
          const isActive = n.id === datasetVersionId
          const color = laneColor(n.lane)
          return (
            <button
              key={n.id}
              type="button"
              disabled={disabled}
              onClick={() => onSelect(n.id)}
              title={isActive ? 'Active version' : 'Click to time-travel to this version'}
              style={{
                position: 'absolute',
                top: (total - 1 - n.row) * ROW_H,
                left: 0,
                right: 0,
                height: ROW_H,
                padding: `4px 0 4px ${gutter}px`,
                background: 'none',
                border: 'none',
                textAlign: 'left',
                cursor: disabled ? 'not-allowed' : isActive ? 'default' : 'pointer',
              }}
              onMouseEnter={(e) => { if (!isActive) e.currentTarget.firstChild.style.background = 'var(--history-card-hover)' }}
              onMouseLeave={(e) => { if (!isActive) e.currentTarget.firstChild.style.background = 'transparent' }}
            >
              <div
                style={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  gap: '2px',
                  padding: '0 14px',
                  borderRadius: '10px',
                  background: isActive ? 'var(--history-card-active-bg)' : 'transparent',
                  border: `1px solid ${isActive ? 'var(--history-card-active-border)' : 'transparent'}`,
                  borderLeft: `3px solid ${color}`,
                  transition: 'background 150ms ease',
                  overflow: 'hidden',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', whiteSpace: 'nowrap' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-signal)' }}>
                    {shortId(n.id)}
                  </span>
                  {isActive && <span className="badge badge-signal" style={{ fontSize: '10px' }}>active</span>}
                  {n.id === newestId && !isActive && (
                    <span style={{ fontSize: '10px', color: 'var(--color-ember)', fontFamily: 'var(--font-mono)' }}>latest</span>
                  )}
                  {n.isFork && (
                    <span style={{ fontSize: '10px', color, fontFamily: 'var(--font-mono)' }}>
                      ⑂ branched from {shortId(n.parentId)}
                    </span>
                  )}
                  {!n.parentId && (
                    <span style={{ fontSize: '10px', color: 'var(--color-mist)', fontFamily: 'var(--font-mono)' }}>original</span>
                  )}
                </div>
                <p style={{ margin: 0, fontSize: '0.875rem', color: 'var(--color-foam)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {n.action_performed}
                </p>
                {n.created_at && (
                  <p style={{ margin: 0, fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-mist)' }}>
                    {formatTime(n.created_at)}
                  </p>
                )}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
