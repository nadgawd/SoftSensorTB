import { useMemo, useState } from 'react'
import ToolForm from './ToolForm'
import { errorDetail, runManualTool } from '../../api/client'
import { useDatasetSchema } from '../../hooks/useManualTools'
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

/**
 * The step's tools: pick one, set its parameters, run it.
 * Form values live in App so the assistant's tool calls fill them in too.
 */
export default function ManualTools({
  activeStep,
  datasetVersionId,
  targetVariable,
  selectedFeatures,
  modelKey,
  tools,
  specError,
  onRetrySpec,
  edits,
  editsFrom,
  onEdit,
  onReset,
  toolByStep,
  onToolSelect,
  assistantBusy,
  onBusyChange,
  onResult,
}) {
  const { schema, error: schemaError, loading: schemaLoading } = useDatasetSchema(datasetVersionId, modelKey)
  const [errors, setErrors] = useState({})
  const [running, setRunning] = useState(null)
  const [status, setStatus] = useState(null)

  const stepTools = useMemo(
    () => (tools || []).filter((t) => t.steps.includes(activeStep)),
    [tools, activeStep],
  )
  const tool = stepTools.find((t) => t.name === toolByStep[activeStep])
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
        <p className="manual-error">Could not load the tools: {specError}</p>
        <button type="button" className="btn-ghost" onClick={onRetrySpec}>Retry</button>
      </section>
    )
  }
  if (!tools) {
    return <section className="panel manual-panel"><p className="manual-hint">Loading tools…</p></section>
  }
  if (!tool) return null

  const toolErrors = errors[tool.name] || {}
  const toolStatus = status?.tool === tool.name ? status : null
  const isRunning = running !== null
  const edited = Boolean(edits[tool.name] && Object.keys(edits[tool.name]).length)

  function setField(name, value) {
    onEdit(tool.name, name, value)
    if (toolErrors[name]) {
      setErrors((prev) => ({ ...prev, [tool.name]: { ...prev[tool.name], [name]: undefined } }))
    }
  }

  function reset() {
    onReset(tool.name)
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
    onBusyChange?.(true)
    setStatus(null)
    try {
      const result = await runManualTool(tool.name, datasetVersionId, args)
      await onResult(result, { tool: tool.name, label: tool.label, args })
      setStatus({ tool: tool.name, ok: true, text: result.message || 'Done.' })
    } catch (err) {
      setStatus({ tool: tool.name, ok: false, text: errorDetail(err, 'The tool failed.') })
    } finally {
      setRunning(null)
      onBusyChange?.(false)
    }
  }

  const selectId = `manual-tool-${activeStep}`
  return (
    <section className="panel manual-panel" aria-label="Tools">
      <div className="manual-head">
        <label className="manual-title" htmlFor={selectId}>Tools</label>
        <select
          id={selectId}
          className="input-base manual-input manual-tool-select"
          value={tool.name}
          onChange={(e) => onToolSelect(activeStep, e.target.value)}
        >
          {stepTools.map((t) => <option key={t.name} value={t.name}>{t.label}</option>)}
        </select>
        {editsFrom[tool.name] === 'assistant' && (
          <span className="badge badge-ember manual-source">Set by the assistant’s last run</span>
        )}
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
            <button type="submit" className="btn-signal" disabled={isRunning || assistantBusy || !schema}>
              {running === tool.name ? 'Running…' : RUN_LABELS[tool.kind] || 'Run'}
            </button>
            <button type="button" className="btn-ghost" onClick={reset} disabled={!edited || isRunning}>
              Reset to defaults
            </button>
            {!datasetVersionId ? (
              <span className="manual-hint">Upload a dataset in Step 1 to run tools.</span>
            ) : assistantBusy ? (
              <span className="manual-hint">The assistant is working on this dataset; wait for it to finish.</span>
            ) : (
              KIND_NOTES[tool.kind] && <span className="manual-hint">{KIND_NOTES[tool.kind]}</span>
            )}
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
