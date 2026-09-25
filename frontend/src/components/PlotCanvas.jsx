import { useEffect, useRef } from 'react'
import DynamicGraph from './DynamicGraph'

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/**
 * Plot panel with the session's plot history: the selected plot fills the
 * canvas, and a strip above it switches between earlier ones.
 */
export default function PlotCanvas({ label, plotHistory, emptyState }) {
  const { plots, activePlot, selectPlot, removePlot } = plotHistory
  const activeIndex = activePlot ? plots.findIndex((p) => p.id === activePlot.id) : -1
  const isLatest = activePlot && activeIndex === plots.length - 1
  const stripRef = useRef(null)

  // Scroll only the strip; scrollIntoView would also scroll the page.
  useEffect(() => {
    const strip = stripRef.current
    const chip = strip?.querySelector('[aria-selected="true"]')
    if (!chip) return
    const left = chip.offsetLeft - strip.offsetLeft
    if (left < strip.scrollLeft || left + chip.offsetWidth > strip.scrollLeft + strip.clientWidth) {
      strip.scrollTo({ left: Math.max(0, left - 12), behavior: 'smooth' })
    }
  }, [activePlot?.id])

  return (
    <div
      className="panel"
      style={{ borderRadius: '14px', flex: '1 0 auto', minHeight: '450px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}
    >
      <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--color-line)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: 0 }}>
          {label}
        </p>
        {activePlot && (
          <span className={`badge ${isLatest ? 'badge-signal' : 'badge-mist'}`} style={{ flexShrink: 0, whiteSpace: 'nowrap' }}>
            {isLatest ? 'Latest' : `Plot ${activeIndex + 1} of ${plots.length}`}
          </span>
        )}
      </div>

      {plots.length > 0 && (
        <div className="plot-history" role="tablist" aria-label="Generated plots" ref={stripRef}>
          {[...plots].reverse().map((plot) => {
            const selected = plot.id === activePlot?.id
            return (
              <div key={plot.id} className="plot-chip" aria-selected={selected} role="tab">
                <button type="button" className="plot-chip-open" onClick={() => selectPlot(plot.id)} title={plot.title}>
                  <span className="plot-chip-num">{plots.indexOf(plot) + 1}</span>
                  <span className="plot-chip-title">{plot.title}</span>
                  <span className="plot-chip-time">{formatTime(plot.createdAt)}</span>
                </button>
                <button
                  type="button"
                  className="plot-chip-remove"
                  onClick={() => removePlot(plot.id)}
                  aria-label={`Remove ${plot.title}`}
                  title="Remove from history"
                >
                  ×
                </button>
              </div>
            )
          })}
        </div>
      )}

      <div style={{ flex: 1, minHeight: '400px' }}>
        {activePlot ? <DynamicGraph key={activePlot.id} data={activePlot.data} /> : emptyState}
      </div>
    </div>
  )
}
