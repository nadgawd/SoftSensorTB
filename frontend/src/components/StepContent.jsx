import { useMemo, useState } from 'react'
import DataIngestionStep from './steps/DataIngestionStep'
import DataQualityStep from './steps/DataQualityStep'
import PreprocessingStep from './steps/PreprocessingStep'
import FeatureSelectionStep from './steps/FeatureSelectionStep'
import ModelingStep from './steps/ModelingStep'
import ValidationStep from './steps/ValidationStep'
import ExportStep from './steps/ExportStep'
import VersionHistory from './VersionHistory'
import { getDynamicContext } from '../utils/tips'
import { usePersisted } from '../hooks/usePersisted'

const HISTORY_VIEWS = [
  { id: 'tree', label: 'Tree' },
  { id: 'timeline', label: 'Timeline' },
]

const STEP_LABELS = {
  ingestion: 'Data Ingestion',
  quality: 'Data Quality',
  preprocessing: 'Preprocessing',
  'feature-selection': 'Feature Selection',
  modeling: 'Modeling',
  validation: 'Validation',
  export: 'Export',
}

const STEP_SUBTITLES = {
  ingestion: 'Upload historian and lab data, review tag classifications',
  quality: 'Assess sensor health, identify data issues',
  preprocessing: 'Resample and clean data for analysis',
  'feature-selection': 'Select process variables for the soft-sensor model',
  modeling: 'Build and compare regression models',
  validation: 'Evaluate model performance on the reserved validation set',
  export: 'Package model, data, and decision log for deployment',
}

const STEP_NUMBERS = {
  ingestion: 1,
  quality: 2,
  preprocessing: 3,
  'feature-selection': 4,
  modeling: 5,
  validation: 6,
  export: 7,
}

/**
 * Central content area that renders the correct panel for the active step.
 */
export default function StepContent({
  activeStep,
  fileName,
  datasetVersionId,
  tablePreview,
  plotHistory,
  modelMetrics,
  versionHistory,
  targetVariable,
  previewLimit,
  onPreviewLimitChange,
  selectedFeatures,
  onUploadSuccess,
  onTargetSelect,
  onFeaturesChange,
  onRollback,
  onHistoryRefresh,
  onSuggestedAction, // ← NEW: fires the message to the assistant
}) {
  const [showHistory, setShowHistory] = useState(false)
  const [historyView, setHistoryView] = usePersisted('historyView', 'tree')
  const stepNum = STEP_NUMBERS[activeStep] || 1

  // Build dynamic context for tips engine
  const tipsCtx = useMemo(() => ({
    columns: tablePreview?.length ? Object.keys(tablePreview[0]) : [],
    rowCount: tablePreview?.length ?? 0,
    hasTarget: Boolean(targetVariable),
    targetVariable,
    selectedFeatures: selectedFeatures || [],
    hasModel: Boolean(modelMetrics),
    r2Score: modelMetrics?.r2_score ?? null,
    hasPlot: Boolean(plotHistory.activePlot),
  }), [tablePreview, targetVariable, selectedFeatures, modelMetrics, plotHistory.activePlot])

  const { actions, tips } = useMemo(
    () => getDynamicContext(activeStep, tipsCtx),
    [activeStep, tipsCtx],
  )

  function renderStepPanel() {
    const commonProps = { actions, tips, onAction: onSuggestedAction }

    switch (activeStep) {
      case 'ingestion':
        return (
          <DataIngestionStep
            fileName={fileName}
            tablePreview={tablePreview}
            targetVariable={targetVariable}
            onUploadSuccess={onUploadSuccess}
            onTargetSelect={onTargetSelect}
            {...commonProps}
          />
        )
      case 'quality':
        return (
          <DataQualityStep
            tablePreview={tablePreview}
            fileName={fileName}
            {...commonProps}
          />
        )
      case 'preprocessing':
        return (
          <PreprocessingStep
            tablePreview={tablePreview}
            plotHistory={plotHistory}
            datasetVersionId={datasetVersionId}
            {...commonProps}
          />
        )
      case 'feature-selection':
        return (
          <FeatureSelectionStep
            tablePreview={tablePreview}
            targetVariable={targetVariable}
            selectedFeatures={selectedFeatures}
            onFeaturesChange={onFeaturesChange}
            {...commonProps}
          />
        )
      case 'modeling':
        return (
          <ModelingStep
            modelMetrics={modelMetrics}
            datasetVersionId={datasetVersionId}
            {...commonProps}
          />
        )
      case 'validation':
        return (
          <ValidationStep
            modelMetrics={modelMetrics}
            plotHistory={plotHistory}
            {...commonProps}
          />
        )
      case 'export':
        return (
          <ExportStep
            fileName={fileName}
            modelMetrics={modelMetrics}
            tablePreview={tablePreview}
            datasetVersionId={datasetVersionId}
            {...commonProps}
          />
        )
      default:
        return null
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0, height: '100%', overflow: 'hidden' }}>
      {/* Step header */}
      <div
        style={{
          padding: '16px 28px 14px',
          borderBottom: '1px solid var(--color-line)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          flexShrink: 0,
        }}
      >
        <div>
          <p
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.14em',
              color: 'var(--color-mist)',
              margin: '0 0 3px',
            }}
          >
            Step {stepNum}
          </p>
          <h2
            style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1.4rem',
              fontWeight: 700,
              color: 'var(--color-foam)',
              margin: '0 0 3px',
              letterSpacing: '-0.01em',
            }}
          >
            {STEP_LABELS[activeStep]}
          </h2>
          <p style={{ fontSize: '0.8125rem', color: 'var(--color-mist)', margin: 0 }}>
            {STEP_SUBTITLES[activeStep]}
          </p>
        </div>

        {/* Version history toggle + active version badge */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexShrink: 0 }}>
          {datasetVersionId && (
            <div
              style={{
                padding: '5px 10px',
                borderRadius: '8px',
                background: 'var(--version-badge-bg)',
                border: '1px solid var(--color-line)',
                fontSize: '11px',
                fontFamily: 'var(--font-mono)',
                color: 'var(--color-mist)',
              }}
            >
              v/{datasetVersionId.slice(0, 8)}
            </div>
          )}
          {versionHistory?.length > 0 && (
            <button
              type="button"
              className="btn-ghost"
              onClick={() => setShowHistory((v) => !v)}
              style={{ fontSize: '12px', padding: '5px 10px' }}
            >
              {showHistory ? 'Hide history' : `History (${versionHistory.length})`}
            </button>
          )}
        </div>
      </div>

      {/* Version history drawer */}
      {showHistory && (
        <div
          style={{
            padding: '14px 28px',
            borderBottom: '1px solid var(--color-line)',
            background: 'var(--history-bg)',
            maxHeight: '260px',
            overflowY: 'auto',
            flexShrink: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', margin: '0 0 12px' }}>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--color-mist)', margin: 0 }}>
              Version History — Click to time-travel
            </p>
            <div
              role="tablist"
              aria-label="History view"
              style={{ display: 'flex', padding: '2px', borderRadius: '8px', border: '1px solid var(--color-line)', background: 'var(--color-panel)' }}
            >
              {HISTORY_VIEWS.map(({ id, label }) => {
                const selected = historyView === id
                return (
                  <button
                    key={id}
                    type="button"
                    role="tab"
                    aria-selected={selected}
                    onClick={() => setHistoryView(id)}
                    style={{
                      padding: '3px 10px',
                      borderRadius: '6px',
                      border: 'none',
                      fontSize: '11px',
                      fontFamily: 'var(--font-mono)',
                      cursor: 'pointer',
                      background: selected ? 'var(--history-card-active-bg)' : 'transparent',
                      color: selected ? 'var(--color-signal)' : 'var(--color-mist)',
                      fontWeight: selected ? 600 : 400,
                      transition: 'all 150ms ease',
                    }}
                  >
                    {label}
                  </button>
                )
              })}
            </div>
          </div>
          <VersionHistory
            view={historyView}
            versionHistory={versionHistory}
            datasetVersionId={datasetVersionId}
            onRollback={onRollback}
            onHistoryRefresh={onHistoryRefresh}
          />
        </div>
      )}

      {/* Main step content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px 28px' }}>
        {renderStepPanel()}
      </div>

      {/* Data preview footer (all steps except ingestion) */}
      {tablePreview?.length > 0 && activeStep !== 'ingestion' && (
        <DataPreviewFooter 
          columns={Object.keys(tablePreview[0])} 
          rows={tablePreview} 
          previewLimit={previewLimit}
          onPreviewLimitChange={onPreviewLimitChange}
        />
      )}
    </div>
  )
}

function DataPreviewFooter({ columns, rows, previewLimit, onPreviewLimitChange }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div style={{ borderTop: '1px solid var(--color-line)', flexShrink: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', width: '100%', paddingRight: '28px' }}>
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          style={{
            flex: 1,
            textAlign: 'left',
            padding: '9px 0 9px 28px',
            background: 'none',
            border: 'none',
            color: 'var(--color-mist)',
            fontSize: '12px',
            fontFamily: 'var(--font-mono)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            transition: 'color 150ms ease',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--color-foam)')}
          onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--color-mist)')}
        >
          <span>{expanded ? '▾' : '▸'}</span>
          Data Preview — {rows.length} rows shown · {columns.length} columns
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: 'var(--color-mist)', fontFamily: 'var(--font-mono)' }}>
          <span>Limit:</span>
          <input 
            type="number" 
            min="1" 
            max="10000" 
            value={previewLimit || 25} 
            onChange={(e) => {
              const val = parseInt(e.target.value, 10)
              if (!isNaN(val) && val > 0) {
                onPreviewLimitChange?.(val)
              }
            }}
            style={{
              width: '50px',
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid var(--color-line-2)',
              color: 'var(--color-foam)',
              padding: '2px 4px',
              borderRadius: '4px',
              fontFamily: 'inherit',
              fontSize: 'inherit',
              outline: 'none',
            }}
          />
        </div>
      </div>
      {expanded && (
        <div style={{ overflowX: 'auto', maxHeight: '180px', overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i}>
                  {columns.map((c) => (
                    <td key={c}>{row[c] === null || row[c] === undefined ? '—' : String(row[c])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
