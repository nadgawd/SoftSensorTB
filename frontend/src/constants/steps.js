/**
 * Canonical step definitions — single source of truth.
 * Separate file so STEPS can be imported without causing HMR issues.
 */
export const STEPS = [
  { id: 'ingestion',         label: 'Data ingestion',    tooltip: 'Upload and parse raw historian or lab datasets.' },
  { id: 'quality',           label: 'Data quality',      tooltip: 'Handle missing values, outliers, and data cleaning.' },
  { id: 'preprocessing',     label: 'Preprocessing',     tooltip: 'Normalize, transform, or engineer new features.' },
  { id: 'feature-selection', label: 'Feature selection', tooltip: 'Select inputs (X) and target (y) for modeling.' },
  { id: 'modeling',          label: 'Modeling',          tooltip: 'Train soft-sensor regression models like PLS or Ridge.' },
  { id: 'validation',        label: 'Validation',        tooltip: 'Evaluate model fit with parity and residual plots.' },
  { id: 'export',            label: 'Export',            tooltip: 'Export predictions, pipeline, or model artifacts.' },
]
