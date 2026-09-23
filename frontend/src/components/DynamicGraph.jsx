import Plot from 'react-plotly.js'
import { useTheme } from '../hooks/useTheme'

/**
 * Renders Plotly JSON from generate_custom_plot / generate_parity_plot.
 * Expects ``json.loads(fig.to_json())`` shape: { data, layout, ... }.
 */
export default function DynamicGraph({ data }) {
  const { theme } = useTheme()

  if (!data || !Array.isArray(data.data) || data.data.length === 0) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          minHeight: '280px',
          gap: '10px',
          padding: '24px',
          textAlign: 'center',
          color: 'var(--color-mist)',
        }}
      >
        <p style={{ fontSize: '1.5rem' }}>📈</p>
        <p style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', color: 'var(--color-foam)' }}>
          Visual canvas
        </p>
        <p style={{ fontSize: '0.875rem', maxWidth: '380px' }}>
          Ask the agent for an EDA chart or a soft-sensor parity plot. Live Plotly
          figures appear here when{' '}
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-signal)' }}>
            plot_data
          </span>{' '}
          arrives.
        </p>
      </div>
    )
  }

  const styles = getComputedStyle(document.documentElement)
  const plotFont = styles.getPropertyValue('--plot-font').trim() || (theme === 'light' ? '#1a2433' : '#e7f2f4')
  const plotBg = styles.getPropertyValue('--plot-bg').trim() || (theme === 'light' ? '#ffffff' : 'rgba(11,24,30,0.55)')
  const plotPaper = styles.getPropertyValue('--plot-paper').trim() || 'transparent'
  const plotGrid = styles.getPropertyValue('--plot-grid').trim() || 'rgba(36,53,69,0.6)'
  const plotZero = styles.getPropertyValue('--plot-zero').trim() || 'rgba(46,196,182,0.3)'

  const layout = {
    ...(data.layout || {}),
    autosize: true,
    paper_bgcolor: plotPaper,
    plot_bgcolor: plotBg,
    font: {
      ...(data.layout?.font || {}),
      family: 'Sora, sans-serif',
      color: plotFont,
    },
    margin: { t: 48, r: 24, b: 48, l: 56, ...(data.layout?.margin || {}) },
    xaxis: {
      ...(data.layout?.xaxis || {}),
      gridcolor: plotGrid,
      zerolinecolor: plotZero,
      linecolor: plotGrid,
      tickfont: { color: plotFont },
      title: data.layout?.xaxis?.title
        ? { ...(typeof data.layout.xaxis.title === 'object' ? data.layout.xaxis.title : { text: data.layout.xaxis.title }), font: { color: plotFont } }
        : data.layout?.xaxis?.title,
    },
    yaxis: {
      ...(data.layout?.yaxis || {}),
      gridcolor: plotGrid,
      zerolinecolor: plotZero,
      linecolor: plotGrid,
      tickfont: { color: plotFont },
      title: data.layout?.yaxis?.title
        ? { ...(typeof data.layout.yaxis.title === 'object' ? data.layout.yaxis.title : { text: data.layout.yaxis.title }), font: { color: plotFont } }
        : data.layout?.yaxis?.title,
    },
  }

  return (
    <div className="animate-fade-in" style={{ width: '100%', height: '100%', minHeight: '400px' }} key={theme}>
      <Plot
        data={data.data}
        layout={layout}
        frames={data.frames || []}
        config={{
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        }}
        useResizeHandler
        style={{ width: '100%', height: '100%', minHeight: 400 }}
      />
    </div>
  )
}
