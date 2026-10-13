import { useMemo, useState } from 'react'
import ToolForm from './ToolForm'
import { errorDetail, runManualTool } from '../../api/client'
import { useDatasetSchema, useManualTools } from '../../hooks/useManualTools'
import { usePersisted } from '../../hooks/usePersisted'
import { buildArguments, formValues } from '../../lib/manualForm'

const PREFERRED_TOOL = {
  ingestion: 'get_column_statistics',
  quality: 'remove_missing_data',
  preprocessing: 'normalize_data',
  'feature-selection': 'select_features_rfe',
  modeling: 'train_soft_sensor',
  validation: 'generate_parity_plot',
  export: 'run_duckdb_query',
}

const RUN_LABELS = {
  train: 'Train model',
  transform: 'Apply',
  plot: 'Plot',
  table: 'Run',
  run: 'Run',
}

const KIND_NOTES = {
  transform: 'Creates a new dataset version. Use History to go back.',
  train: 'Trains on the active dataset version and evaluates on the held-out test rows.',
}

/** Pick a tool for the current step, set its parameters and run it without the assistant. */
export default function ManualTools({ activeStep, datasetVersionId, targetVariable, selectedFeatures, modelKey, onResult }) {
  const { tools, error: specError, retry } = useManualTools()
  const { schema, error: schemaError, loading: schemaLoading } = useDatasetSchema(datasetVersionId, modelKey)
  const [toolByStep, setToolByStep] = usePersisted('manualToolByStep', {})
  const [edits, setEdits] = useState({})
  const [errors, setErrors] = useState({})
  const [running, setRunning] = useState(null)
  const [status, setStatus] = useState(null)

  const stepTools = useMemo(
    () => (tools || []).filter((t) => t.steps.includes(activeStep)),
    [tools, activeStep],
  )
  const chosen = toolByStep[activeStep]
  const tool = stepTools.find((t) => t.name === chosen)
    || stepTools.find((t) => t.name === PREFERRED_TOOL[activeStep])
    || stepTools[0]

  const app = useMemo(() => ({ targetVariable, selectedFeatures }), [targetVariable, selectedFeatures])
  const values = useMemo(
    () => (tool ? formValues(tool, edits[tool.name], schema, app) : {}),
    [tool, edits, schema, app],
  )

  if (specError) {
    return (
      <section className="panel manual-panel">
        <p className="manual-error">Could not load the tool forms: {specError}</p>
        <button type="button" className="btn-ghost" onClick={retry}>Retry</button>
      </section>
    )
  }
  if (!tools) {
    return <section className="panel manual-panel"><p className="manual-hint">Loading tools…</p></section>
  }
  if (!tool) return null

  const toolErrors = errors[tool.name] || {}
  const toolStatus = status?.tool === tool.name ? status : null
  const isRunning = running === tool.name
  const edited = Boolean(edits[tool.name] && Object.keys(edits[tool.name]).length)

  function setField(name, value) {
    setEdits((prev) => ({ ...prev, [tool.name]: { ...prev[tool.name], [name]: value } }))
    if (toolErrors[name]) {
      setErrors((prev) => ({ ...prev, [tool.name]: { ...prev[tool.name], [name]: undefined } }))
    }
  }

  function reset() {
    setEdits((prev) => ({ ...prev, [tool.name]: {} }))
    setErrors((prev) => ({ ...prev, [tool.name]: {} }))
    setStatus(null)
  }

  async function run() {
    const { args, errors: found } = buildArguments(tool, values)
    setErrors((prev) => ({ ...prev, [tool.name]: found }))
    if (Object.keys(found).length) {
      setStatus({ tool: tool.name, ok: false, text: 'Fix the highlighted fields first.' })
      return
    }
    setRunning(tool.name)
    setStatus(null)
    try {
      const result = await runManualTool(tool.name, datasetVersionId, args)
      onResult(result)
      setStatus({ tool: tool.name, ok: true, text: result.message || 'Done.' })
    } catch (err) {
      setStatus({ tool: tool.name, ok: false, text: errorDetail(err, 'The tool failed.') })
    } finally {
      setRunning(null)
    }
  }

  return (
    <section className="panel manual-panel" aria-label="Manual tools">
      <div className="manual-head">
        <p className="manual-title">Manual tools</p>
        <div className="manual-tabs" role="tablist" aria-label="Tools for this step">
          {stepTools.map((t) => (
            <button
              key={t.name}
              type="button"
              role="tab"
              aria-selected={t.name === tool.name}
              className={`manual-tab${t.name === tool.name ? ' active' : ''}`}
              onClick={() => setToolByStep((prev) => ({ ...prev, [activeStep]: t.name }))}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <p className="manual-tool-desc" title={tool.description}>{tool.description}</p>

      {schemaError && <p className="manual-error">Could not load the dataset columns: {schemaError}</p>}
      {!schema && schemaLoading ? (
        <p className="manual-hint">Loading columns…</p>
      ) : (
        <form
          onSubmit={(e) => {
            e.preventDefault()
            run()
          }}
        >
          <ToolForm tool={tool} values={values} onChange={setField} schema={schema} errors={toolErrors} />
          <div className="manual-actions">
            <button type="submit" className="btn-signal" disabled={isRunning || !schema}>
              {isRunning ? 'Running…' : RUN_LABELS[tool.kind] || 'Run'}
            </button>
            <button type="button" className="btn-ghost" onClick={reset} disabled={!edited || isRunning}>
              Reset to defaults
            </button>
            {KIND_NOTES[tool.kind] && <span className="manual-hint">{KIND_NOTES[tool.kind]}</span>}
          </div>
        </form>
      )}

      {toolStatus && (
        <p className={toolStatus.ok ? 'manual-status' : 'manual-status error'} role="status">
          {toolStatus.text}
        </p>
      )}
    </section>
  )
}
