import { ChartScatter } from 'lucide-react'
import PlotCanvas from '../PlotCanvas'

/** Model step › Diagnostics: the plot made by the diagnostic tools above it. */
export default function DiagnosticsTab({ modelMetrics, plotHistory }) {
  if (!modelMetrics) {
    return (
      <p className="manual-hint" style={{ textAlign: 'center', padding: '20px' }}>
        Train a model on the Train &amp; test tab first; its diagnostic plots appear here.
      </p>
    )
  }
  return (
    <div className="animate-rise">
      <PlotCanvas
        label="Diagnostic plot"
        plotHistory={plotHistory}
        emptyState={
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--color-mist)', padding: '30px', gap: '8px' }}>
            <ChartScatter size={24} aria-hidden="true" />
            <p style={{ fontSize: '0.875rem', textAlign: 'center' }}>
              Pick a plot in the Tools panel above (parity, residuals, importances…) and press Plot.
            </p>
          </div>
        }
      />
    </div>
  )
}
