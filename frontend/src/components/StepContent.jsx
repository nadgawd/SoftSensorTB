import { useState } from 'react'
import DataIngestionStep from './steps/DataIngestionStep'
import DataQualityStep from './steps/DataQualityStep'
import PreprocessingStep from './steps/PreprocessingStep'
import FeatureSelectionStep from './steps/FeatureSelectionStep'
import ModelingStep from './steps/ModelingStep'
import ValidationStep from './steps/ValidationStep'
import ExportStep from './steps/ExportStep'
import VersionHistory from './VersionHistory'
import ManualTools from './manual/ManualTools'
import ResultTable from './manual/ResultTable'
import DataDock from './DataDock'
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
  modeling: 'Split the data, train a model on the training rows, then test it on the held-out rows',
  validation: 'Evaluate model performance on the reserved validation set',
  export: 'Package model, data, and decision log for deployment',
}

// Steps whose own panel is an overview that tells you which tool to use, so it comes before the tools.
const PANEL_FIRST = new Set(['ingestion', 'quality'])

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
  previewMeta,
  onPreviewPageChange,
  profile,
  timeColumn,
  onTimeColumnChange,
  onOpenTool,
  selectedFeatures,
  onUploadSuccess,
  onTargetSelect,
  onFeaturesChange,
  onRollback,
  onHistoryRefresh,
  manualTools,
  resultTable,
  onResultTableClose,
  dock,
}) {
  const [showHistory, setShowHistory] = useState(false)
  const [historyView, setHistoryView] = usePersisted('historyView', 'tree')
  const stepNum = STEP_NUMBERS[activeStep] || 1

  function renderStepPanel() {
    switch (activeStep) {
      case 'ingestion':
        return (
          <DataIngestionStep
            fileName={fileName}
            tablePreview={tablePreview}
            targetVariable={targetVariable}
            onUploadSuccess={onUploadSuccess}
            onTargetSelect={onTargetSelect}
          />
        )
      case 'quality':
        return (
          <DataQualityStep
            profile={profile}
            targetVariable={targetVariable}
            onOpenTool={onOpenTool}
          />
        )
      case 'preprocessing':
        return (
          <PreprocessingStep
            tablePreview={tablePreview}
            datasetVersionId={datasetVersionId}
          />
        )
      case 'feature-selection':
        return (
          <FeatureSelectionStep
            tablePreview={tablePreview}
            targetVariable={targetVariable}
            selectedFeatures={selectedFeatures}
            onFeaturesChange={onFeaturesChange}
          />
        )
      case 'modeling':
        return (
          <ModelingStep
            modelMetrics={modelMetrics}
            datasetVersionId={datasetVersionId}
            onToolResult={manualTools.onResult}
            busy={manualTools.assistantBusy}
          />
        )
      case 'validation':
        return (
          <ValidationStep
            modelMetrics={modelMetrics}
            plotHistory={plotHistory}
          />
        )
      case 'export':
        return (
          <ExportStep
            datasetVersionId={datasetVersionId}
            modelMetrics={modelMetrics}
            profile={profile}
            targetVariable={targetVariable}
            timeColumn={timeColumn}
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
        {PANEL_FIRST.has(activeStep) && (
          <div style={{ marginBottom: '16px' }}>{renderStepPanel()}</div>
        )}
        <ManualTools
          activeStep={activeStep}
          datasetVersionId={datasetVersionId}
          targetVariable={targetVariable}
          selectedFeatures={selectedFeatures}
          modelKey={modelMetrics?.model_id}
          {...manualTools}
        />
        {resultTable && <ResultTable table={resultTable} onClose={onResultTableClose} />}
        {!PANEL_FIRST.has(activeStep) && renderStepPanel()}
      </div>

      {/* Data preview + Graph dock (every step, once there is data) */}
      {datasetVersionId && (
        <DataDock
          rows={tablePreview}
          previewMeta={previewMeta}
          previewLimit={previewLimit}
          onPreviewLimitChange={onPreviewLimitChange}
          onPageChange={onPreviewPageChange}
          open={dock.open}
          tab={dock.tab}
          onOpenChange={dock.onOpenChange}
          onTabChange={dock.onTabChange}
          plotHistory={plotHistory}
          graphTools={{
            ...manualTools,
            datasetVersionId,
            targetVariable,
            selectedFeatures,
            modelKey: modelMetrics?.model_id,
          }}
        />
      )}
    </div>
  )
}
