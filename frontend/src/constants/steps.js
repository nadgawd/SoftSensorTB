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
  { id: 'modeling',          label: 'Modeling',          description: 'Split the data, train a model on the training rows, then test it on the held-out rows.' },
  { id: 'validation',        label: 'Validation',        description: 'Check the model with parity, residual, importance and learning-curve plots.' },
  { id: 'export',            label: 'Export',            description: 'Download the prepared data, the model, the pipeline recipe and a report.' },
]

export const STEP_BY_ID = Object.fromEntries(STEPS.map((s, i) => [s.id, { ...s, number: i + 1 }]))
