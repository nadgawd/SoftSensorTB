import { Check, Lock, RotateCcw } from 'lucide-react'
import { STEPS } from '../constants/steps'
import { formatCount } from '../utils/format'

/**
 * Left sidebar: the steps, each with its live status (what has been done, what needs attention).
 */
export default function StepSidebar({
  fileName,
  targetVariable,
  activeStep,
  onStepChange,
  completedSteps,
  profile,
  rowCount,
  columnCount,
  selectedFeatures,
  modelMetrics,
  versionCount,
  onNewProject,
  compact,
}) {
  function isStepDisabled(stepId, index) {
    if (index > 0 && !fileName) return true
    if (stepId === 'modeling' && !targetVariable) return true
    return false
  }

  /** `[text, tone]` — tone is one of ok / warn / alert / undefined. */
  function status(stepId) {
    if (!fileName) return stepId === 'ingestion' ? ['Upload a file'] : [null]
    switch (stepId) {
      case 'ingestion':
        return [rowCount != null ? `${formatCount(rowCount)} rows · ${columnCount ?? '?'} tags` : 'Loading…']
      case 'quality': {
        const s = profile?.data?.summary
        if (!s) return [profile?.loading ? 'Checking…' : null]
        if (s.problem) return [`${s.problem} problem${s.problem > 1 ? 's' : ''}${s.warning ? ` · ${s.warning} to check` : ''}`, 'alert']
        if (s.warning) return [`${s.warning} tag${s.warning > 1 ? 's' : ''} to check`, 'warn']
        return ['All tags look fine', 'ok']
      }
      case 'preprocessing':
        return [versionCount > 1 ? `${versionCount - 1} change${versionCount > 2 ? 's' : ''} so far` : null]
      case 'feature-selection':
        if (!targetVariable) return ['Pick a target', 'warn']
        if (!selectedFeatures?.length) return [`${targetVariable} ← all numeric tags`]
        return [`${targetVariable} ← ${selectedFeatures.length} input${selectedFeatures.length === 1 ? '' : 's'}`]
      case 'modeling':
        if (!targetVariable) return ['Needs a target']
        if (!modelMetrics) return ['No model yet']
        return typeof modelMetrics.r2_score === 'number'
          ? [`${modelMetrics.algorithm} · test R² ${modelMetrics.r2_score.toFixed(2)}`, modelMetrics.r2_score >= 0.5 ? 'ok' : 'warn']
          : [`${modelMetrics.algorithm} · not tested yet`, 'warn']
      case 'deployment':
        return [modelMetrics ? 'Put a model on live data' : 'Needs a trained model']
      default:
        return [null]
    }
  }

  return (
    <aside className={`panel step-sidebar${compact ? ' compact' : ''}`}>
      {!compact && (
        <div className="step-sidebar-project">
          <p className="step-sidebar-kicker">Project</p>
          <p className="step-sidebar-file" title={fileName || undefined}>{fileName || 'No dataset loaded'}</p>
        </div>
      )}

      <nav className="step-sidebar-nav">
        {STEPS.map((step, index) => {
          const isDone = completedSteps?.includes(step.id)
          const isActive = activeStep === step.id
          const isDisabled = isStepDisabled(step.id, index)
          const [text, tone] = status(step.id)

          return (
            <div key={step.id}>
              <button
                type="button"
                title={compact
                  ? [step.label, text, isDisabled ? 'Complete the previous steps first' : null].filter(Boolean).join(' · ')
                  : isDisabled ? 'Complete the previous steps first' : step.description}
                aria-label={compact ? step.label : undefined}
                disabled={isDisabled}
                onClick={() => onStepChange(step.id)}
                className={`step-item${isActive ? ' active' : isDone ? ' completed' : ''}${isDisabled ? ' disabled' : ''}`}
              >
                <div className={`step-dot ${isDone ? 'done' : isActive ? 'active' : 'pending'}`}>
                  {isDone ? <Check size={12} strokeWidth={3} color="#fff" />
                    : isDisabled ? <Lock size={10} />
                    : <span className="step-dot-num">{index + 1}</span>}
                </div>
                {!compact && (
                  <div style={{ minWidth: 0 }}>
                    <div className="step-item-label">{step.label}</div>
                    {text && <div className={`step-item-status${tone ? ` tone-${tone}` : ''}`}>{text}</div>}
                  </div>
                )}
                {compact && tone && <span className={`step-item-flag tone-${tone}`} aria-hidden="true" />}
              </button>
              {index < STEPS.length - 1 && <div className={`step-connector${!isDone ? ' incomplete' : ''}`} />}
            </div>
          )
        })}
      </nav>

      {onNewProject && (
        <div className="step-sidebar-footer">
          <button
            type="button"
            className="step-sidebar-new"
            title="New project"
            aria-label="New project"
            onClick={() => {
              if (window.confirm('Start a new project? This will clear all current progress.')) onNewProject()
            }}
          >
            <RotateCcw size={12} aria-hidden="true" /> {!compact && 'New project'}
          </button>
        </div>
      )}
    </aside>
  )
}
