import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { BookOpen, Check, CircleCheck, RefreshCw, Search, Sparkles, TriangleAlert } from 'lucide-react'
import StepSidebar from './components/StepSidebar'
import { MODEL_TABS, STEPS, uiPlaceOf } from './constants/steps'
import StepContent from './components/StepContent'
import AssistantPanel from './components/AssistantPanel'
import DocumentationModal from './components/DocumentationModal'
import ThemeToggle from './components/ThemeToggle'
import AccountMenu from './components/AccountMenu'
import {
  errorDetail, fetchDatasetHistory, fetchModelReport, fetchVersionDiff, fetchVersionPreview, rollbackVersion, runManualTool,
} from './api/client'
import { useColumnRoles } from './hooks/useColumnRoles'
import ChangeToast from './components/ChangeToast'
import ToolPalette from './components/ToolPalette'
import DataSummaryBar from './components/DataSummaryBar'
import { shortcutLabel } from './utils/keys'
import { actionLabel } from './utils/versions'
import { usePersisted, clearPersistedSession } from './hooks/usePersisted'
import { resetProject } from './lib/projectSync'
import { usePlotHistory } from './hooks/usePlotHistory'
import { useTheme } from './hooks/useTheme'
import { getSuggestions } from './utils/suggestions'
import { useManualTools } from './hooks/useManualTools'
import { useVersionProfile } from './hooks/useVersionProfile'
import { editsFromArguments } from './lib/manualForm'

const STEP_IDS = STEPS.map((s) => s.id)

const DIAGNOSTIC_PLOT_TOOLS = new Set([
  'generate_parity_plot',
  'generate_residuals_plot',
  'generate_importance_plot',
  'generate_coefficients_plot',
  'generate_roc_curve',
  'generate_pr_curve',
  'generate_learning_curve',
])

const ASSISTANT_DEFAULT_WIDTH = 380
const ASSISTANT_MIN_WIDTH = 320
const ASSISTANT_MAX_WIDTH = 1200
// Space kept for the step sidebar plus a usable slice of the main canvas.
const MAIN_MIN_WIDTH = 480
const SIDEBAR_COMPACT_WIDTH = 56
const SIDEBAR_COMPACT_BELOW = 1000

const CHAT_MAX_AGE = 24 * 60 * 60 * 1000
const MAX_MANUAL_ACTIONS_IN_CONTEXT = 10
const RECENT_TOOLS = 6
const WELCOME_MESSAGE = {
  role: 'assistant',
  intent: null,
  content:
    'Hello! I\'m your soft-sensor assistant. Upload a dataset to get started, or ask me a conceptual question about process analytics, PLS, linear regression (OLS), Ridge, Lasso, PCR, k-NN, or EDA.',
  toolCalls: [],
}

/**
 * Soft Sensor Toolbox — main application shell.
 *
 * Pipeline state is persisted to localStorage and synced to the server
 * (lib/projectSync), so the project is kept until the user starts a new one.
 */
export default function App() {
  // ── Persisted pipeline state ─────────────────────────────
  const [activeStep, setActiveStep] = usePersisted('activeStep', 'ingestion')
  const [modelTab, setModelTab] = usePersisted('modelTab', MODEL_TABS[0].id)
  // The step whose tools are shown: the Model step's tabs each show one backend step.
  const toolStep = activeStep === 'modeling'
    ? (MODEL_TABS.find((t) => t.id === modelTab) || MODEL_TABS[0]).toolStep
    : activeStep
  // Validation used to be a step of its own; it is now the Model step's Diagnostics tab.
  useEffect(() => {
    if (STEPS.some((s) => s.id === activeStep)) return
    const place = uiPlaceOf(activeStep)
    setActiveStep(STEPS.some((s) => s.id === place.step) ? place.step : 'ingestion')
    if (place.modelTab) setModelTab(place.modelTab)
  }, [activeStep, setActiveStep, setModelTab])
  const [completedSteps, setCompletedSteps] = usePersisted('completedSteps', [])
  const [fileName, setFileName] = usePersisted('fileName', null)
  const [datasetId, setDatasetId] = usePersisted('datasetId', null)
  const [datasetVersionId, setDatasetVersionId] = usePersisted('datasetVersionId', null)
  const [targetVariable, setTargetVariable] = usePersisted('targetVariable', null)
  const [selectedFeatures, setSelectedFeatures] = usePersisted('selectedFeatures', [])
  const [timeColumn, setTimeColumn] = usePersisted('timeColumn', null)
  const [modelMetrics, setModelMetrics] = usePersisted('modelMetrics', null)
  const profile = useVersionProfile(datasetVersionId, { target: targetVariable, timeColumn })
  // A chosen time column, else the one the profile detected.
  const effectiveTimeColumn = timeColumn || profile.data?.time_column || null

  // Table preview is kept in memory (not persisted) to allow large row limits without breaking localStorage quotas.
  // It is automatically re-fetched on load via useEffect.
  const [tablePreview, setTablePreviewRaw] = useState([])
  // Total rows of the active version and the positions of the preview rows in it.
  const [previewMeta, setPreviewMeta] = useState({ rowCount: null, rowNumbers: [], offset: 0, sample: false })
  const plotHistory = usePlotHistory()
  const { addPlot, clearActive: clearActivePlot, clearAll: clearPlots } = plotHistory
  const [versionHistory, setVersionHistory] = usePersisted('versionHistory', [])

  const setTablePreview = useCallback((rows) => {
    setTablePreviewRaw(Array.isArray(rows) ? rows : [])
  }, [setTablePreviewRaw])

  const applyPreview = useCallback((res) => {
    if (!res) return
    setTablePreview(res.table_preview || [])
    setPreviewMeta({
      rowCount: res.row_count ?? null,
      rowNumbers: res.row_numbers || [],
      offset: res.offset || 0,
      sample: Boolean(res.sample),
    })
  }, [setTablePreview])


  // ── Data Preview ──────────────────────────────────────────
  const [previewLimit, setPreviewLimit] = usePersisted('previewLimit', 25)
  const [previewPage, setPreviewPage] = useState({ offset: 0, sample: false, seed: 0 })
  const [dockOpen, setDockOpen] = usePersisted('dockOpen', false)
  const [dockTab, setDockTab] = usePersisted('dockTab', 'preview')
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
        applyPreview(await fetchVersionPreview(datasetVersionId, { limit: previewLimit }))
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
    fetchVersionPreview(datasetVersionId, { limit: previewLimit, ...previewPage })
      .then((res) => { if (active) applyPreview(res) })
      .catch(() => {})
    return () => { active = false }
  }, [datasetVersionId, previewLimit, previewPage, applyPreview])

  // ── What the last run changed ─────────────────────────────
  // A new active version whose parent was the previous active one came from a run:
  // its diff colours the preview's column headers and shows a toast with Undo.
  const [lastDiff, setLastDiff] = useState(null)
  const [changeToast, setChangeToast] = useState(false)
  const prevVersionRef = useRef(datasetVersionId)
  useEffect(() => {
    const previous = prevVersionRef.current
    prevVersionRef.current = datasetVersionId
    if (!datasetVersionId || !previous || previous === datasetVersionId) return undefined
    let active = true
    fetchVersionDiff(datasetVersionId)
      .then((diff) => {
        if (!active) return
        const fromRun = diff.parent_version_id === previous
        setLastDiff(fromRun ? diff : null)
        setChangeToast(fromRun)
      })
      .catch(() => { if (active) setLastDiff(null) })
    return () => { active = false }
  }, [datasetVersionId])
  const columnChanges = lastDiff?.dataset_version_id === datasetVersionId ? lastDiff : null

  // Tool output tables (statistics, SQL, model list) — kept apart from the dataset preview.
  const [resultTable, setResultTable] = useState(null)

  // ── Tools panel + assistant, side by side, sharing state ──
  // The chat log also records Tools-panel runs, so the assistant sees them.
  const [chatMessages, setChatMessages] = usePersisted('chatMessages', [WELCOME_MESSAGE], { maxAge: CHAT_MAX_AGE })
  // Per tool: form values the user (or the assistant's last call) set.
  const [manualEdits, setManualEdits] = usePersisted('manualEdits', {})
  const [manualEditsFrom, setManualEditsFrom] = usePersisted('manualEditsFrom', {})
  const [manualToolByStep, setManualToolByStep] = usePersisted('manualToolByStep', {})
  const [recentTools, setRecentTools] = usePersisted('recentTools', [])
  const [paletteOpen, setPaletteOpen] = useState(false)
  const { tools: manualToolSpecs, error: manualSpecError, retry: retryManualSpec } = useManualTools()
  const toolLabels = useMemo(() => new Map((manualToolSpecs || []).map((t) => [t.name, t.label])), [manualToolSpecs])
  const closeChangeToast = useCallback(() => setChangeToast(false), [])
  const [assistantBusy, setAssistantBusy] = useState(false)
  const [toolsBusy, setToolsBusy] = useState(false)

  const [assistantOpen, setAssistantOpen] = usePersisted('assistantOpen', true)
  const { theme, toggleTheme } = useTheme()

  // Suggested prompts shown in the assistant chat.
  const suggestions = useMemo(() => getSuggestions(toolStep || activeStep, {
    columns: tablePreview?.length ? Object.keys(tablePreview[0]) : [],
    hasTarget: Boolean(targetVariable),
    targetVariable,
    selectedFeatures: selectedFeatures || [],
    hasModel: Boolean(modelMetrics),
    r2Score: modelMetrics?.r2_score ?? null,
  }), [toolStep, activeStep, tablePreview, targetVariable, selectedFeatures, modelMetrics])

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
  // Narrow windows: the Assistant floats over the page instead of squeezing it, and
  // the sidebar shrinks to the step icons once the main area gets tight.
  const assistantFloats = assistantOpen && viewportWidth - assistantWidth - SIDEBAR_COMPACT_WIDTH < MAIN_MIN_WIDTH
  const mainAreaWidth = viewportWidth - (assistantOpen && !assistantFloats ? assistantWidth : 0)
  const sidebarCompact = mainAreaWidth < SIDEBAR_COMPACT_BELOW

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
      setPreviewPage({ offset: 0, sample: false, seed: 0 })
      applyPreview({ ...result, row_numbers: (result.table_preview || []).map((_, i) => i) })
      clearPlots()
      setResultTable(null)
      setModelMetrics(null)
      setSelectedFeatures([])
      setTargetVariable(null)
      setTimeColumn(null)
      markStepComplete('ingestion')
      await refreshHistory(result.dataset_id)
      setActiveStep('quality')
      setRestoreStatus(null)
    },
    [refreshHistory, setFileName, setDatasetId, setDatasetVersionId, applyPreview,
      clearPlots, setModelMetrics, setSelectedFeatures, setTargetVariable, setTimeColumn, setActiveStep],
  )

  // ── Chat response / manual tool run → UI sync ─────────────
  const handleChatResponse = useCallback(
    async (response) => {
      if (!response?.ui_update_required) return
      if (response.active_dataset_version_id) {
        setDatasetVersionId(response.active_dataset_version_id)
      }
      let nextStep = null
      if (response.plot_data) {
        addPlot(response.plot_data)
        const diagnostic = (response.tool_calls_made || []).some((t) => DIAGNOSTIC_PLOT_TOOLS.has(t))
        if (diagnostic) {
          nextStep = 'modeling'
          setModelTab('diagnostics')
        } else {
          setDockTab('graph')
          setDockOpen(true)
        }
      }
      if (Array.isArray(response.table_preview)) {
        setResultTable({ title: response.table_title || 'Result', rows: response.table_preview })
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
        if (!nextStep && activeStep !== 'modeling') {
          nextStep = 'modeling'
          setModelTab('train')
        }
      }
      if (nextStep && nextStep !== activeStep) setActiveStep(nextStep)
      if (datasetId) await refreshHistory(datasetId)
    },
    [activeStep, datasetId, refreshHistory, setActiveStep, setDatasetVersionId, setModelTab,
      setModelMetrics, addPlot, setSelectedFeatures, setTargetVariable, setDockOpen, setDockTab],
  )

  // ── Tools panel run → same UI sync as the chat, plus a chat log entry ──
  const handleToolResult = useCallback(async (response, { tool, label, args }) => {
    setRecentTools((prev) => [tool, ...prev.filter((t) => t !== tool)].slice(0, RECENT_TOOLS))
    await handleChatResponse(response)
    setChatMessages((prev) => [...prev, {
      role: 'manual',
      tool,
      label,
      args,
      content: response.message || '',
      at: Date.now(),
    }])
  }, [handleChatResponse, setChatMessages, setRecentTools])

  const editManualField = useCallback((tool, field, value) => {
    setManualEdits((prev) => ({ ...prev, [tool]: { ...prev[tool], [field]: value } }))
    setManualEditsFrom((prev) => {
      if (!prev[tool]) return prev
      const next = { ...prev }
      delete next[tool]
      return next
    })
  }, [setManualEdits, setManualEditsFrom])

  const resetManualTool = useCallback((tool) => {
    setManualEdits((prev) => ({ ...prev, [tool]: {} }))
    setManualEditsFrom((prev) => {
      const next = { ...prev }
      delete next[tool]
      return next
    })
  }, [setManualEdits, setManualEditsFrom])

  const selectManualTool = useCallback((step, tool) => {
    setManualToolByStep((prev) => ({ ...prev, [step]: tool }))
  }, [setManualToolByStep])

  const openTool = useCallback((step, tool) => {
    const place = uiPlaceOf(step)
    selectManualTool(step, tool)
    setActiveStep(place.step)
    if (place.modelTab) setModelTab(place.modelTab)
  }, [selectManualTool, setActiveStep, setModelTab])

  const pickPaletteTool = useCallback((tool) => {
    setPaletteOpen(false)
    if (!tool.steps.length) {
      setDockOpen(true)
      setDockTab('graph')
      return
    }
    openTool(tool.steps.includes(toolStep) ? toolStep : tool.steps[0], tool.name)
    setTimeout(() => {
      const panel = document.querySelector('.manual-panel:not(.manual-panel-fixed)')
      panel?.scrollIntoView({ block: 'start', behavior: 'smooth' })
      panel?.querySelector('form input, form select, form textarea')?.focus({ preventScroll: true })
    }, 80)
  }, [toolStep, openTool, setDockOpen, setDockTab])

  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen((v) => !v)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // ── Column roles and the Data preview's column menu ───────
  const roleColumns = useMemo(
    () => profile.data?.columns || (tablePreview?.length ? Object.keys(tablePreview[0]).map((name) => ({ name, kind: null })) : []),
    [profile.data, tablePreview],
  )
  const columnRoles = useColumnRoles({
    columns: roleColumns,
    timeColumn: effectiveTimeColumn,
    targetVariable,
    selectedFeatures,
    onTimeColumnChange: setTimeColumn,
    onTargetSelect: setTargetVariable,
    onFeaturesChange: setSelectedFeatures,
  })
  const runToolNow = useCallback(async (tool, args) => {
    setToolsBusy(true)
    try {
      const result = await runManualTool(tool, datasetVersionId, args)
      await handleToolResult(result, { tool, label: toolLabels.get(tool) || tool, args })
    } finally {
      setToolsBusy(false)
    }
  }, [datasetVersionId, handleToolResult, toolLabels])
  const columnActions = useMemo(() => ({
    kinds: new Map(roleColumns.map((c) => [c.name, c.kind])),
    timeColumn: effectiveTimeColumn,
    roleOf: columnRoles.roleOf,
    setRole: columnRoles.setRole,
    run: runToolNow,
    busy: toolsBusy || assistantBusy,
  }), [roleColumns, effectiveTimeColumn, columnRoles.roleOf, columnRoles.setRole, runToolNow, toolsBusy, assistantBusy])

  const activateModel = useCallback(async (modelId) => {
    setModelMetrics(await fetchModelReport(datasetVersionId, modelId))
  }, [datasetVersionId, setModelMetrics])

  // ── Assistant tool calls → the same tool's form shows what it used ──
  const handleAssistantToolCalls = useCallback((calls) => {
    const specs = new Map((manualToolSpecs || []).map((t) => [t.name, t]))
    const filled = calls.filter((c) => specs.has(c.name))
    if (!filled.length) return
    setManualEdits((prev) => {
      const next = { ...prev }
      for (const c of filled) next[c.name] = editsFromArguments(specs.get(c.name), c.args)
      return next
    })
    setManualEditsFrom((prev) => {
      const next = { ...prev }
      for (const c of filled) next[c.name] = 'assistant'
      return next
    })
    setManualToolByStep((prev) => {
      const next = { ...prev }
      for (const c of filled) {
        const { steps } = specs.get(c.name)
        if (steps.length) next[steps.includes(toolStep) ? toolStep : steps[0]] = c.name
      }
      return next
    })
  }, [manualToolSpecs, toolStep, setManualEdits, setManualEditsFrom, setManualToolByStep])

  // Tools-panel runs the assistant should know about (since its last reply are marked).
  const manualActions = useMemo(() => {
    let lastReply = -1
    chatMessages.forEach((m, i) => { if (m.role === 'assistant') lastReply = i })
    return chatMessages
      .map((m, i) => ({ m, i }))
      .filter(({ m }) => m.role === 'manual')
      .slice(-MAX_MANUAL_ACTIONS_IN_CONTEXT)
      .map(({ m, i }) => ({
        tool: m.tool,
        args: m.args,
        result: (m.content || '').slice(0, 300),
        sinceLastReply: i > lastReply,
      }))
  }, [chatMessages])

  // ── Rollback ──────────────────────────────────────────────
  const handleRollback = useCallback((result) => {
    setDatasetId(result.dataset_id)
    setDatasetVersionId(result.active_dataset_version_id)
    setPreviewPage({ offset: 0, sample: false, seed: 0 })
    applyPreview({ ...result, row_numbers: (result.table_preview || []).map((_, i) => i) })
    clearActivePlot()
  }, [setDatasetId, setDatasetVersionId, applyPreview, clearActivePlot])

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
        profile={profile}
        rowCount={previewMeta.rowCount}
        columnCount={tablePreview?.length ? Object.keys(tablePreview[0]).length : null}
        selectedFeatures={selectedFeatures}
        modelMetrics={modelMetrics}
        versionCount={versionHistory?.length || 0}
        onStepChange={handleStepChange}
        onNewProject={hasSession ? handleNewProject : null}
        compact={sidebarCompact}
      />

      {/* ── Main content area ──────────────────────────────── */}
      <div
        className="app-main"
        style={{
          transition: isResizingAssistant ? 'none' : 'margin-right 320ms cubic-bezier(0.22,1,0.36,1)',
          marginRight: assistantOpen && !assistantFloats ? `${assistantWidth}px` : '0',
        }}
      >
        <header className="app-header">
          <span className="app-brand">Soft Sensor Toolbox</span>
          {hasSession && (
            <span className="app-saved" title="Every change is saved to your account until you start a new project">
              <Check size={12} aria-hidden="true" /> <span className="btn-label">Saved</span>
            </span>
          )}
          <button type="button" className="app-header-btn app-find-btn" onClick={() => setPaletteOpen(true)}
            title="Find any tool by name or by what it does" aria-label="Find a tool">
            <Search size={14} aria-hidden="true" /> <span className="btn-label">Find a tool</span> <kbd>{shortcutLabel('K')}</kbd>
          </button>
          <button type="button" className="app-header-btn" onClick={() => setDocOpen(true)} title="Documentation" aria-label="Docs">
            <BookOpen size={14} aria-hidden="true" /> <span className="btn-label">Docs</span>
          </button>
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
          <AccountMenu />
          <button
            type="button"
            className={`app-header-btn app-assistant-btn${assistantOpen ? ' active' : ''}`}
            onClick={() => setAssistantOpen((v) => !v)}
            aria-label="Assistant"
          >
            <Sparkles size={14} aria-hidden="true" /> <span className="btn-label">Assistant</span>
          </button>
        </header>

        {hasSession && (
          <DataSummaryBar
            profile={profile}
            rowCount={previewMeta.rowCount}
            columnCount={tablePreview?.length ? Object.keys(tablePreview[0]).length : null}
            targetVariable={targetVariable}
            onOpenQuality={() => handleStepChange('quality')}
          />
        )}

        {/* Step content */}
        <StepContent
          activeStep={activeStep}
          toolStep={toolStep}
          modelTab={modelTab}
          onModelTabChange={setModelTab}
          onModelActivate={activateModel}
          fileName={fileName}
          datasetVersionId={datasetVersionId}
          tablePreview={tablePreview}
          plotHistory={plotHistory}
          modelMetrics={modelMetrics}
          versionHistory={versionHistory}
          targetVariable={targetVariable}
          previewLimit={previewLimit}
          onPreviewLimitChange={setPreviewLimit}
          previewMeta={previewMeta}
          onPreviewPageChange={setPreviewPage}
          columnChanges={columnChanges}
          columnActions={columnActions}
          profile={profile}
          timeColumn={effectiveTimeColumn}
          onTimeColumnChange={setTimeColumn}
          onOpenTool={openTool}
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
          manualTools={{
            tools: manualToolSpecs,
            specError: manualSpecError,
            onRetrySpec: retryManualSpec,
            edits: manualEdits,
            editsFrom: manualEditsFrom,
            onEdit: editManualField,
            onReset: resetManualTool,
            toolByStep: manualToolByStep,
            onToolSelect: selectManualTool,
            assistantBusy,
            onBusyChange: setToolsBusy,
            onResult: handleToolResult,
          }}
          resultTable={resultTable}
          onResultTableClose={() => setResultTable(null)}
          dock={{ open: dockOpen, tab: dockTab, onOpenChange: setDockOpen, onTabChange: setDockTab }}
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
          timeColumn: effectiveTimeColumn,
          activePlot: plotHistory.activePlot?.title ?? null,
          modelMetrics: modelMetrics || null,
          manualActions,
        })}
        onChatResponse={handleChatResponse}
        onToolCalls={handleAssistantToolCalls}
        onBusyChange={setAssistantBusy}
        toolsBusy={toolsBusy}
        messages={chatMessages}
        setMessages={setChatMessages}
        suggestions={suggestions}
        width={assistantWidth}
        onWidthPreview={setDragAssistantWidth}
        onWidthCommit={commitAssistantWidth}
        onWidthReset={() => commitAssistantWidth(ASSISTANT_DEFAULT_WIDTH)}
        floating={assistantFloats}
      />

      {changeToast && columnChanges && (
        <ChangeToast
          diff={columnChanges}
          label={actionLabel(columnChanges.action_performed, toolLabels)}
          onClose={closeChangeToast}
          onUndo={async () => {
            try {
              const result = await rollbackVersion(columnChanges.parent_version_id)
              handleRollback(result)
              await refreshHistory(result.dataset_id)
              setChangeToast(false)
            } catch (err) {
              window.alert(`Could not undo: ${errorDetail(err)}`)
            }
          }}
        />
      )}

      {/* ── Documentation Modal ────────────────────────────── */}
      {docOpen && <DocumentationModal onClose={() => setDocOpen(false)} manualSpecs={manualToolSpecs} />}

      {paletteOpen && (
        <ToolPalette tools={manualToolSpecs} recent={recentTools} onPick={pickPaletteTool} onClose={() => setPaletteOpen(false)} />
      )}
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
      <span style={{ display: 'flex', flexShrink: 0, color: isFailed ? 'var(--color-alert)' : isRestoring ? 'var(--color-mist)' : 'var(--color-success)' }}>
        {isRestoring ? <RefreshCw size={16} /> : isFailed ? <TriangleAlert size={16} /> : <CircleCheck size={16} />}
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
