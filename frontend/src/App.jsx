import { useCallback, useEffect, useRef, useState } from 'react'
import StepSidebar from './components/StepSidebar'
import { STEPS } from './constants/steps'
import StepContent from './components/StepContent'
import AssistantPanel from './components/AssistantPanel'
import DocumentationModal from './components/DocumentationModal'
import ThemeToggle from './components/ThemeToggle'
import AccountMenu from './components/AccountMenu'
import { errorDetail, fetchDatasetHistory, fetchVersionPreview } from './api/client'
import { usePersisted, clearPersistedSession } from './hooks/usePersisted'
import { resetProject } from './lib/projectSync'
import { usePlotHistory } from './hooks/usePlotHistory'
import { useTheme } from './hooks/useTheme'

const STEP_IDS = STEPS.map((s) => s.id)

const ASSISTANT_DEFAULT_WIDTH = 380
const ASSISTANT_MIN_WIDTH = 320
const ASSISTANT_MAX_WIDTH = 1200
// Space kept for the step sidebar plus a usable slice of the main canvas.
const MAIN_MIN_WIDTH = 480

/**
 * Soft Sensor Toolbox — main application shell.
 *
 * Pipeline state is persisted to localStorage and synced to the server
 * (lib/projectSync), so the project is kept until the user starts a new one.
 */
export default function App() {
  // ── Persisted pipeline state ─────────────────────────────
  const [activeStep, setActiveStep] = usePersisted('activeStep', 'ingestion')
  const [completedSteps, setCompletedSteps] = usePersisted('completedSteps', [])
  const [fileName, setFileName] = usePersisted('fileName', null)
  const [datasetId, setDatasetId] = usePersisted('datasetId', null)
  const [datasetVersionId, setDatasetVersionId] = usePersisted('datasetVersionId', null)
  const [targetVariable, setTargetVariable] = usePersisted('targetVariable', null)
  const [selectedFeatures, setSelectedFeatures] = usePersisted('selectedFeatures', [])
  const [modelMetrics, setModelMetrics] = usePersisted('modelMetrics', null)

  // Table preview is kept in memory (not persisted) to allow large row limits without breaking localStorage quotas.
  // It is automatically re-fetched on load via useEffect.
  const [tablePreview, setTablePreviewRaw] = useState([])
  const plotHistory = usePlotHistory()
  const { addPlot, clearActive: clearActivePlot, clearAll: clearPlots } = plotHistory
  const [versionHistory, setVersionHistory] = usePersisted('versionHistory', [])

  // Safe setter without row capping
  const setTablePreview = useCallback((rows) => {
    const arr = Array.isArray(rows) ? rows : []
    setTablePreviewRaw(arr)
  }, [setTablePreviewRaw])


  // ── Data Preview ──────────────────────────────────────────
  const [previewLimit, setPreviewLimit] = usePersisted('previewLimit', 25)
  const [docOpen, setDocOpen] = useState(false)

  // ── Restore state ─────────────────────────────────────────
  const [restoreStatus, setRestoreStatus] = useState(null) // 'restoring' | 'restored' | 'failed' | null
  const restoredRef = useRef(false)

  useEffect(() => {
    async function verifyAndRestore() {
      // Only verify if we have a saved dataset version ID
      if (!datasetVersionId || restoredRef.current) return
      restoredRef.current = true
      setRestoreStatus('restoring')

      try {
        // Ping the server to confirm this version still exists
        const preview = await fetchVersionPreview(datasetVersionId, previewLimit)
        // Update table preview with fresh server data if rows differ
        if (preview?.rows?.length && preview.rows.length !== tablePreview.length) {
          setTablePreview(preview.rows)
        }
        // Refresh history
        if (datasetId) {
          const history = await fetchDatasetHistory(datasetId)
          setVersionHistory(history)
        }
        setRestoreStatus('restored')
        // Auto-dismiss after 4 seconds
        setTimeout(() => setRestoreStatus(null), 4000)
      } catch {
        // Version no longer exists on server (backend restarted, DB reset, etc.)
        setRestoreStatus('failed')
      }
    }

    verifyAndRestore()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Auto-refresh Data Table on Version Change ─────────────
  useEffect(() => {
    if (!datasetVersionId) return
    let active = true
    fetchVersionPreview(datasetVersionId, previewLimit)
      .then((res) => {
        if (active && res?.table_preview) {
          setTablePreview(res.table_preview)
        }
      })
      .catch(() => {})
    return () => { active = false }
  }, [datasetVersionId, previewLimit, setTablePreview])

  // ── UI state (not persisted — just UI toggles) ────────────
  const [assistantOpen, setAssistantOpen] = useState(false)
  const [pendingAction, setPendingAction] = useState(null)
  const { theme, toggleTheme } = useTheme()

  // ── Assistant panel width (drag its left edge) ────────────
  const [savedAssistantWidth, setSavedAssistantWidth] = usePersisted('assistantWidth', ASSISTANT_DEFAULT_WIDTH)
  const [dragAssistantWidth, setDragAssistantWidth] = useState(null)
  const [viewportWidth, setViewportWidth] = useState(() => window.innerWidth)

  useEffect(() => {
    const onResize = () => setViewportWidth(window.innerWidth)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const clampAssistantWidth = useCallback((w) => {
    const max = Math.max(ASSISTANT_MIN_WIDTH, Math.min(ASSISTANT_MAX_WIDTH, viewportWidth - MAIN_MIN_WIDTH))
    return Math.round(Math.min(Math.max(w, ASSISTANT_MIN_WIDTH), max))
  }, [viewportWidth])

  const assistantWidth = clampAssistantWidth(dragAssistantWidth ?? savedAssistantWidth)
  const isResizingAssistant = dragAssistantWidth !== null

  const commitAssistantWidth = useCallback((w) => {
    setSavedAssistantWidth(clampAssistantWidth(w))
    setDragAssistantWidth(null)
  }, [clampAssistantWidth, setSavedAssistantWidth])

  // Plotly's resize handler only listens to window resizes, so nudge it once
  // the main area has settled after the panel opens, closes, or changes width.
  useEffect(() => {
    const timer = setTimeout(() => window.dispatchEvent(new Event('resize')), 340)
    return () => clearTimeout(timer)
  }, [assistantOpen, savedAssistantWidth])

  // ── Suggested action chip → assistant send ────────────────
  const handleSuggestedAction = useCallback((text) => {
    setPendingAction(text)
    setAssistantOpen(true)
  }, [])

  // ── History helpers ───────────────────────────────────────
  const refreshHistory = useCallback(async (id) => {
    if (!id) return
    try {
      const history = await fetchDatasetHistory(id)
      setVersionHistory(history)
    } catch {
      // best-effort
    }
  }, [setVersionHistory])

  // ── Upload success ────────────────────────────────────────
  const handleUploadSuccess = useCallback(
    async (result) => {
      setFileName(result.file_name)
      setDatasetId(result.dataset_id)
      setDatasetVersionId(result.dataset_version_id)
      setTablePreview(result.table_preview || [])
      clearPlots()
      setModelMetrics(null)
      setSelectedFeatures([])
      setTargetVariable(null)
      markStepComplete('ingestion')
      await refreshHistory(result.dataset_id)
      setActiveStep('quality')
      setRestoreStatus(null)
    },
    [refreshHistory, setFileName, setDatasetId, setDatasetVersionId, setTablePreview,
      clearPlots, setModelMetrics, setSelectedFeatures, setTargetVariable, setActiveStep],
  )

  // ── Chat response → UI sync ───────────────────────────────
  const handleChatResponse = useCallback(
    async (response) => {
      if (!response?.ui_update_required) return
      if (response.active_dataset_version_id) {
        setDatasetVersionId(response.active_dataset_version_id)
      }
      if (response.plot_data) {
        addPlot(response.plot_data)
        if (activeStep === 'ingestion' || activeStep === 'quality') {
          setActiveStep('preprocessing')
        }
      }
      if (Array.isArray(response.table_preview)) {
        setTablePreview(response.table_preview)
      }
      if (response.target_variable !== undefined && response.target_variable !== null) {
        setTargetVariable(response.target_variable)
      }
      if (Array.isArray(response.selected_features)) {
        setSelectedFeatures(response.selected_features)
        if (response.selected_features.length > 0 && response.target_variable) {
          markStepComplete('feature-selection')
        }
      }
      if (response.model_metrics) {
        setModelMetrics(response.model_metrics)
        markStepComplete('modeling')
        if (activeStep !== 'modeling' && activeStep !== 'validation') {
          setActiveStep('modeling')
        }
      }
      if (datasetId) await refreshHistory(datasetId)
    },
    [activeStep, datasetId, refreshHistory, setActiveStep, setDatasetVersionId,
      setModelMetrics, addPlot, setTablePreview, setSelectedFeatures, setTargetVariable],
  )

  // ── Rollback ──────────────────────────────────────────────
  const handleRollback = useCallback((result) => {
    setDatasetId(result.dataset_id)
    setDatasetVersionId(result.active_dataset_version_id)
    setTablePreview(result.table_preview || [])
    clearActivePlot()
  }, [setDatasetId, setDatasetVersionId, setTablePreview, clearActivePlot])

  // ── Step helpers ──────────────────────────────────────────
  function markStepComplete(stepId) {
    setCompletedSteps((prev) =>
      prev.includes(stepId) ? prev : [...prev, stepId],
    )
  }

  function handleStepChange(stepId) {
    const from = STEP_IDS.indexOf(activeStep)
    const to = STEP_IDS.indexOf(stepId)
    if (to > from) {
      const newCompleted = STEP_IDS.slice(0, to)
      setCompletedSteps((prev) => [...new Set([...prev, ...newCompleted])])
    }
    setActiveStep(stepId)
  }

  // ── New project (deletes the project on the server and here) ──
  async function handleNewProject() {
    const confirmed = window.confirm(
      'Start a new project?\n\nThis permanently deletes your uploaded data, every version, '
      + 'trained models, plots and the chat. It cannot be undone.',
    )
    if (!confirmed) return
    try {
      await resetProject()
    } catch (err) {
      window.alert(`Could not delete the project: ${errorDetail(err)}\nNothing was cleared; try again.`)
      return
    }
    clearPersistedSession()
    // Hard reload to reset all React state cleanly
    window.location.reload()
  }

  const stepLabel = STEPS.find((s) => s.id === activeStep)?.label || activeStep
  const hasSession = Boolean(datasetVersionId)

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', position: 'relative' }}>

      {/* ── Restore status banner ──────────────────────────── */}
      {restoreStatus && (
        <RestoreBanner
          status={restoreStatus}
          fileName={fileName}
          onNewProject={handleNewProject}
          onDismiss={() => setRestoreStatus(null)}
        />
      )}

      {/* ── Left sidebar ───────────────────────────────────── */}
      <StepSidebar
        fileName={fileName}
        targetVariable={targetVariable}
        activeStep={activeStep}
        completedSteps={completedSteps}
        tablePreview={tablePreview}
        modelMetrics={modelMetrics}
        onStepChange={handleStepChange}
        onNewProject={hasSession ? handleNewProject : null}
      />

      {/* ── Main content area ──────────────────────────────── */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
          height: '100%',
          overflow: 'hidden',
          transition: isResizingAssistant ? 'none' : 'margin-right 320ms cubic-bezier(0.22,1,0.36,1)',
          marginRight: assistantOpen ? `${assistantWidth}px` : '0',
        }}
      >
        {/* Top bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'flex-end',
            padding: '10px 20px',
            borderBottom: '1px solid var(--color-line)',
            background: 'var(--header-bg)',
            backdropFilter: 'blur(8px)',
            flexShrink: 0,
            gap: '10px',
          }}
        >
          <span style={{ flex: 1, fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-mist)' }}>
            Soft Sensor Toolbox
            {fileName && <> · <span style={{ color: 'var(--color-foam)' }}>{fileName}</span></>}
          </span>

          <ThemeToggle theme={theme} onToggle={toggleTheme} />

          <AccountMenu />

          {/* Documentation button */}
          <button
            type="button"
            onClick={() => setDocOpen(true)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              borderRadius: '6px',
              background: 'var(--header-btn-bg)',
              border: '1px solid var(--color-line-2)',
              color: 'var(--color-foam)',
              cursor: 'pointer',
              fontSize: '0.8rem',
              fontFamily: 'var(--font-heading)',
              transition: 'background 0.2s',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--header-btn-bg-hover)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--header-btn-bg)')}
          >
            <span style={{ fontSize: '14px' }}>📄</span> Documentation
          </button>

          {/* Session save indicator */}
          {hasSession && (
            <span
              title="Project saved until you start a new one"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                fontFamily: 'var(--font-mono)',
                fontSize: '10px',
                color: 'var(--color-success)',
                padding: '4px 8px',
                borderRadius: '6px',
                background: 'rgba(60,179,113,0.08)',
                border: '1px solid rgba(60,179,113,0.2)',
              }}
            >
              <span style={{ fontSize: '8px', animation: 'pulse-ember 3s ease-in-out infinite' }}>●</span>
              auto-saved
            </span>
          )}

          {/* ✦ Assistant button */}
          <button
            type="button"
            onClick={() => setAssistantOpen((v) => !v)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '7px 14px',
              borderRadius: '8px',
              border: `1px solid ${assistantOpen ? 'rgba(212,165,53,0.5)' : 'var(--color-line-2)'}`,
              background: assistantOpen ? 'rgba(212,165,53,0.1)' : 'transparent',
              color: assistantOpen ? 'var(--color-ember)' : 'var(--color-mist)',
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontWeight: 500,
              transition: 'all 150ms ease',
            }}
            onMouseEnter={(e) => {
              if (!assistantOpen) {
                e.currentTarget.style.color = 'var(--color-foam)'
                e.currentTarget.style.borderColor = 'var(--color-line)'
              }
            }}
            onMouseLeave={(e) => {
              if (!assistantOpen) {
                e.currentTarget.style.color = 'var(--color-mist)'
                e.currentTarget.style.borderColor = 'var(--color-line-2)'
              }
            }}
          >
            <span style={{ fontSize: '15px' }}>✦</span>
            <span>Assistant</span>
          </button>
        </div>

        {/* Step content */}
        <StepContent
          activeStep={activeStep}
          fileName={fileName}
          datasetVersionId={datasetVersionId}
          tablePreview={tablePreview}
          plotHistory={plotHistory}
          modelMetrics={modelMetrics}
          versionHistory={versionHistory}
          targetVariable={targetVariable}
          previewLimit={previewLimit}
          onPreviewLimitChange={setPreviewLimit}
          selectedFeatures={selectedFeatures}
          onUploadSuccess={handleUploadSuccess}
          onTargetSelect={(col) => {
            setTargetVariable(col)
            markStepComplete('ingestion')
          }}
          onFeaturesChange={(features) => {
            setSelectedFeatures(features)
            if (features.length > 0) markStepComplete('feature-selection')
          }}
          onRollback={handleRollback}
          onHistoryRefresh={setVersionHistory}
          onSuggestedAction={handleSuggestedAction}
        />
      </div>

      {/* ── Assistant panel ────────────────────────────────── */}
      <AssistantPanel
        open={assistantOpen}
        onClose={() => setAssistantOpen(false)}
        datasetVersionId={datasetVersionId}
        stepHint={stepLabel}
        uiContext={JSON.stringify({
          targetVariable,
          selectedFeatures,
          activePlot: plotHistory.activePlot?.title ?? null,
          modelMetrics: modelMetrics || null,
        })}
        onChatResponse={handleChatResponse}
        pendingAction={pendingAction}
        onPendingActionConsumed={() => setPendingAction(null)}
        width={assistantWidth}
        onWidthPreview={setDragAssistantWidth}
        onWidthCommit={commitAssistantWidth}
        onWidthReset={() => commitAssistantWidth(ASSISTANT_DEFAULT_WIDTH)}
      />

      {/* ── Documentation Modal ────────────────────────────── */}
      {docOpen && <DocumentationModal onClose={() => setDocOpen(false)} />}
    </div>
  )
}

// ── Restore status banner ──────────────────────────────────
function RestoreBanner({ status, fileName, onNewProject, onDismiss }) {
  const isRestoring = status === 'restoring'
  const isFailed = status === 'failed'
  const isRestored = status === 'restored'

  const bgColor = isFailed
    ? 'rgba(228,87,46,0.12)'
    : isRestored
      ? 'rgba(60,179,113,0.1)'
      : 'rgba(46,196,182,0.08)'
  const borderColor = isFailed
    ? 'rgba(228,87,46,0.35)'
    : isRestored
      ? 'rgba(60,179,113,0.3)'
      : 'rgba(46,196,182,0.25)'

  return (
    <div
      style={{
        position: 'fixed',
        top: '12px',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 1000,
        padding: '10px 18px',
        borderRadius: '12px',
        background: bgColor,
        border: `1px solid ${borderColor}`,
        backdropFilter: 'blur(12px)',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
        minWidth: '320px',
        maxWidth: '520px',
        animation: 'slide-up 300ms ease',
      }}
    >
      <span style={{ fontSize: '16px', flexShrink: 0 }}>
        {isRestoring ? '🔄' : isFailed ? '⚠️' : '✅'}
      </span>
      <div style={{ flex: 1 }}>
        <p style={{ margin: 0, fontSize: '0.875rem', fontWeight: 600, color: 'var(--color-foam)', lineHeight: 1.3 }}>
          {isRestoring && 'Restoring your session…'}
          {isRestored && `Session restored — ${fileName || 'dataset'}`}
          {isFailed && 'Session data exists but the server was restarted'}
        </p>
        <p style={{ margin: '2px 0 0', fontSize: '12px', color: 'var(--color-mist)' }}>
          {isRestoring && 'Verifying dataset versions with server…'}
          {isRestored && 'All step progress, metrics, and version history recovered.'}
          {isFailed && 'Re-upload your file to continue. Your settings are preserved.'}
        </p>
      </div>
      <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
        {isFailed && (
          <button
            type="button"
            onClick={onNewProject}
            style={{
              padding: '5px 10px', borderRadius: '7px', fontSize: '12px',
              border: '1px solid rgba(228,87,46,0.4)', background: 'transparent',
              color: 'var(--color-alert)', cursor: 'pointer',
            }}
          >
            Clear all
          </button>
        )}
        <button
          type="button"
          onClick={onDismiss}
          style={{
            padding: '5px 10px', borderRadius: '7px', fontSize: '12px',
            border: '1px solid var(--color-line-2)', background: 'transparent',
            color: 'var(--color-mist)', cursor: 'pointer',
          }}
        >
          ×
        </button>
      </div>
    </div>
  )
}
