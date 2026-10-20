import { useMemo, useState } from 'react'
import ToolForm from './ToolForm'
import { errorDetail, runManualTool } from '../../api/client'
import { useDatasetSchema } from '../../hooks/useManualTools'
import { buildArguments, formValues } from '../../lib/manualForm'

const PREFERRED_TOOL = {
  ingestion: 'inspect_data',
  quality: 'remove_missing_data',
  preprocessing: 'normalize_data',
  'feature-selection': 'select_features_rfe',
  modeling: 'train_soft_sensor',
  validation: 'generate_parity_plot',
  export: 'run_duckdb_query',
}

const TOOL_LAYOUTS = {
  train_soft_sensor: {
    sections: [
      {
        title: 'Model',
        note: 'Pick a model; its parameters appear below it.',
        fields: ['algorithm', 'n_components', 'pls_scale', 'alpha', 'max_iter', 'fit_intercept', 'n_neighbors', 'knn_weights'],
      },
      { title: 'Data', fields: ['target_column', 'feature_columns'] },
      {
        title: 'Train / validation / test split',
        note: 'The rows are split before training. The model is fitted on the training rows only; the test rows stay untouched until you test it.',
        fields: ['split_strategy', 'test_size', 'validation_size', 'stratify_column', 'n_strata', 'group_column', 'random_state'],
      },
      {
        title: 'Cross-validation (optional)',
        note: 'Repeatedly re-trains on parts of the training rows to check that the score is stable.',
        fields: ['cv_strategy', 'cv_folds', 'cv_repeats', 'cv_test_size', 'cv_window', 'n_bootstrap', 'cv_shuffle'],
      },
    ],
    optionHelp: {
      algorithm: {
        OLS: 'Ordinary least squares linear regression. Fast and easy to interpret; nothing to tune except the intercept.',
        PLS: 'Partial least squares: regresses on a few latent components built from the inputs and the target. Suits many correlated sensors. Tune the number of components.',
        RIDGE: 'Linear regression with an L2 penalty that shrinks the coefficients, which steadies correlated inputs. Tune alpha (larger means more shrinkage).',
        LASSO: 'Linear regression with an L1 penalty that can set coefficients to zero, so it also selects features. Tune alpha.',
        PCR: 'Principal component regression: PCA on the inputs, then linear regression on the first components. Tune the number of components.',
        KNN: 'k-nearest neighbours: predicts the average target of the k most similar training rows, so it can follow non-linear behaviour. Tune k and the weighting.',
      },
    },
  },
}

const RUN_LABELS = {
  train: 'Split & train',
  test: 'Test',
  transform: 'Apply',
  plot: 'Plot',
  table: 'Run',
  run: 'Run',
}

const KIND_NOTES = {
  transform: 'Creates a new dataset version. Use History to go back.',
  train: 'Fits the model on the training rows only (and scores the validation rows, if set). The test rows are held out; test the model next.',
  test: 'Scores the chosen model (the newest one if left empty) on the test rows it held out at training time.',
}

const NEXT_TOOL = { train_soft_sensor: 'evaluate_on_test_set' }

/**
 * The step's tools: pick one, set its parameters, run it.
 * Form values live in App so the assistant's tool calls fill them in too.
 * With `fixedTool` it shows that one tool without the picker (the Graph tab).
 */
export default function ManualTools({
  activeStep,
  fixedTool,
  title = 'Tools',
  datasetVersionId,
  targetVariable,
  selectedFeatures,
  timeColumn,
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
    () => (tools || []).filter((t) => (fixedTool ? t.name === fixedTool : t.steps.includes(activeStep))),
    [tools, activeStep, fixedTool],
  )
  const tool = stepTools.find((t) => t.name === toolByStep[activeStep])
    || stepTools.find((t) => t.name === PREFERRED_TOOL[activeStep])
    || stepTools[0]

  const app = useMemo(
    () => ({ targetVariable, selectedFeatures, timeColumn }),
    [targetVariable, selectedFeatures, timeColumn],
  )
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
  // A train run switches to the test tool; its status line stays visible there.
  const toolStatus = status && (status.tool === tool.name || NEXT_TOOL[status.tool] === tool.name) ? status : null
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
      const next = NEXT_TOOL[tool.name]
      if (!fixedTool && next && stepTools.some((t) => t.name === next)) onToolSelect(activeStep, next)
    } catch (err) {
      setStatus({ tool: tool.name, ok: false, text: errorDetail(err, 'The tool failed.') })
    } finally {
      setRunning(null)
      onBusyChange?.(false)
    }
  }

  const selectId = `manual-tool-${activeStep}`
  return (
    <section className={`panel manual-panel${fixedTool ? ' manual-panel-fixed' : ''}`} aria-label={title}>
      <div className="manual-head">
        {fixedTool ? (
          <p className="manual-title">{title}</p>
        ) : (
          <>
            <label className="manual-title" htmlFor={selectId}>{title}</label>
            <select
              id={selectId}
              className="input-base manual-input manual-tool-select"
              value={tool.name}
              onChange={(e) => onToolSelect(activeStep, e.target.value)}
            >
              {stepTools.map((t) => <option key={t.name} value={t.name}>{t.label}</option>)}
            </select>
          </>
        )}
        {editsFrom[tool.name] === 'assistant' && (
          <span className="badge badge-ember manual-source">Set by the assistant’s last run</span>
        )}
      </div>

      {!fixedTool && <p className="manual-tool-desc">{tool.summary || tool.description}</p>}

      {schemaError && <p className="manual-error">Could not load the dataset columns: {schemaError}</p>}
      {!schema && schemaLoading ? (
        <p className="manual-hint">Loading columns…</p>
      ) : (
        <form
          noValidate
          onSubmit={(e) => {
            e.preventDefault()
            run()
          }}
        >
          <ToolForm tool={tool} values={values} onChange={setField} schema={schema} errors={toolErrors} layout={TOOL_LAYOUTS[tool.name]} />
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
