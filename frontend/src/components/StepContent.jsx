import { useState } from 'react'
import { History } from 'lucide-react'
import { STEP_BY_ID } from '../constants/steps'
import DataIngestionStep from './steps/DataIngestionStep'
import DataQualityStep from './steps/DataQualityStep'
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

// Steps whose own panel is an overview that tells you which tool to use, so it comes before the tools.
const PANEL_FIRST = new Set(['ingestion', 'quality'])

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
  const step = STEP_BY_ID[activeStep] || STEP_BY_ID.ingestion
  const roleProps = {
    profile, timeColumn, targetVariable, selectedFeatures,
    onTargetSelect, onFeaturesChange, onTimeColumnChange,
  }

  function renderStepPanel() {
    switch (activeStep) {
      case 'ingestion':
        return (
          <DataIngestionStep
            fileName={fileName}
            tablePreview={tablePreview}
            rowCount={previewMeta?.rowCount}
            {...roleProps}
            onUploadSuccess={onUploadSuccess}
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
      case 'feature-selection':
        return (
          <FeatureSelectionStep tablePreview={tablePreview} {...roleProps} />
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
      <div className="step-header">
        <span className="step-header-num">Step {step.number}</span>
        <h2 className="step-header-title">{step.label}</h2>
        <p className="step-header-desc" title={step.description}>{step.description}</p>
        {versionHistory?.length > 0 && (
          <button
            type="button"
            className={`app-header-btn${showHistory ? ' active' : ''}`}
            onClick={() => setShowHistory((v) => !v)}
          >
            <History size={13} aria-hidden="true" />
            {showHistory ? 'Hide history' : `History (${versionHistory.length})`}
          </button>
        )}
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
          timeColumn={timeColumn}
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
            timeColumn,
            modelKey: modelMetrics?.model_id,
          }}
        />
      )}
    </div>
  )
}
