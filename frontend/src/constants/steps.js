/**
 * Canonical step definitions — single source of truth.
 * Separate file so STEPS can be imported without causing HMR issues.
 * `description` must match the tools the backend places in the step (MANUAL_TOOLS in backend/api/tools.py).
 */
export const STEPS = [
  { id: 'ingestion',         label: 'Data ingestion',    description: 'Upload plant data, look it over, and fix column names and types.' },
  { id: 'quality',           label: 'Data quality',      description: 'Check every tag for gaps, stuck sensors and spikes, then clean missing values, outliers and bad rows.' },
  { id: 'preprocessing',     label: 'Preprocessing',     description: 'Scale, transform, encode and resample; build lag, window and formula features.' },
  { id: 'feature-selection', label: 'Feature selection', description: 'Pick the target and the input tags, and drop redundant ones.' },
  { id: 'modeling',          label: 'Model',             description: 'Split the data, train and test a model, then check it with diagnostic plots.' },
  { id: 'export',            label: 'Export',            description: 'Download the prepared data, the model, the pipeline recipe and a report.' },
]

export const STEP_BY_ID = Object.fromEntries(STEPS.map((s, i) => [s.id, { ...s, number: i + 1 }]))

/** The Model step's tabs; `toolStep` is the step the backend files their tools under. */
export const MODEL_TABS = [
  { id: 'train', label: 'Train & test', toolStep: 'modeling' },
  { id: 'diagnostics', label: 'Diagnostics', toolStep: 'validation' },
]

/** Where a tool of backend step `toolStep` lives in the UI: `{ step, modelTab }`. */
export function uiPlaceOf(toolStep) {
  const tab = MODEL_TABS.find((t) => t.toolStep === toolStep)
  return tab ? { step: 'modeling', modelTab: tab.id } : { step: toolStep, modelTab: null }
}
