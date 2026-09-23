import { STEPS } from '../constants/steps'
import LlmStatus from './LlmStatus'


function CheckIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
      <path d="M2 6l3 3 5-5" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

/**
 * Left sidebar with 7-step pipeline navigation.
 */
export default function StepSidebar({
  fileName,
  targetVariable,
  activeStep,
  onStepChange,
  completedSteps,
  tablePreview,
  modelMetrics,
  onNewProject,
}) {
  function isStepDisabled(stepId, index) {
    if (index > 0 && !fileName) return true
    if ((stepId === 'modeling' || stepId === 'validation') && !targetVariable) return true
    return false
  }

  function getSubText(stepId) {
    if (stepId === 'ingestion' && fileName) {
      if (tablePreview && tablePreview.length > 0) {
        return `${Object.keys(tablePreview[0]).length} columns loaded`
      }
      return 'Dataset loaded'
    }
    if (stepId === 'feature-selection' && targetVariable) return `Target: ${targetVariable}`
    if (stepId === 'modeling' && modelMetrics) return `${modelMetrics.algorithm} R²: ${modelMetrics.r2_score?.toFixed(2)}`
    return null
  }

  return (
    <aside
      className="panel"
      style={{
        width: '220px',
        minWidth: '220px',
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        borderRadius: '0',
        borderTop: 'none',
        borderBottom: 'none',
        borderLeft: 'none',
      }}
    >
      {/* Project info */}
      <div style={{ padding: '20px 16px 16px', borderBottom: '1px solid var(--color-line)' }}>
        <p
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '10px',
            textTransform: 'uppercase',
            letterSpacing: '0.12em',
            color: 'var(--color-mist)',
            margin: '0 0 4px',
          }}
        >
          Project
        </p>
        <p
          style={{
            fontWeight: 600,
            fontSize: '0.9375rem',
            color: 'var(--color-foam)',
            margin: '0 0 6px',
            lineHeight: 1.3,
            wordBreak: 'break-all',
          }}
        >
          {fileName || 'No dataset loaded'}
        </p>
        {targetVariable && (
          <span className="badge badge-signal" style={{ fontSize: '10px' }}>
            {targetVariable}
          </span>
        )}
      </div>

      {/* Steps */}
      <nav style={{ flex: 1, overflowY: 'auto', padding: '12px 8px' }}>
        {STEPS.map((step, index) => {
          const isDone = completedSteps?.includes(step.id)
          const isActive = activeStep === step.id
          const isDisabled = isStepDisabled(step.id, index)
          const dynamicSubText = getSubText(step.id)

          return (
            <div key={step.id}>
              <button
                type="button"
                title={isDisabled ? 'Complete previous steps first' : step.tooltip}
                disabled={isDisabled}
                onClick={() => onStepChange(step.id)}
                className={`step-item${isActive ? ' active' : isDone ? ' completed' : ''}${isDisabled ? ' disabled' : ''}`}
                style={{ 
                  width: '100%', 
                  textAlign: 'left', 
                  background: 'none', 
                  border: 'none',
                  opacity: isDisabled ? 0.4 : 1,
                  cursor: isDisabled ? 'not-allowed' : 'pointer'
                }}
              >
                {/* Step dot */}
                <div
                  className={`step-dot ${isDone ? 'done' : isActive ? 'active' : 'pending'}`}
                  style={{ filter: isDisabled ? 'grayscale(100%)' : 'none' }}
                >
                  {isDone ? <CheckIcon /> : (
                    <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)' }}>
                      {isDisabled ? '🔒' : index + 1}
                    </span>
                  )}
                </div>

                <div>
                  <div style={{ fontSize: '0.8125rem', fontWeight: 500, lineHeight: 1.3 }}>
                    {step.label}
                  </div>
                  {(dynamicSubText || step.tooltip) && (
                    <div style={{ fontSize: '10px', color: 'var(--color-mist)', marginTop: '2px', lineHeight: 1.2 }}>
                      {dynamicSubText || step.tooltip}
                    </div>
                  )}
                </div>
              </button>

              {/* Connector line between steps */}
              {index < STEPS.length - 1 && (
                <div
                  className={`step-connector${!isDone ? ' incomplete' : ''}`}
                />
              )}
            </div>
          )
        })}
      </nav>

      {/* Footer */}
      <div
        style={{
          padding: '12px 16px',
          borderTop: '1px solid var(--color-line)',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        }}
      >
        <LlmStatus />
        {onNewProject && (
          <button
            type="button"
            onClick={() => {
              if (window.confirm('Start a new project? This will clear all current progress.')) {
                onNewProject()
              }
            }}
            style={{
              width: '100%',
              padding: '7px',
              borderRadius: '8px',
              background: 'transparent',
              border: '1px solid var(--color-line-2)',
              color: 'var(--color-mist)',
              fontSize: '11px',
              cursor: 'pointer',
              fontFamily: 'var(--font-mono)',
              transition: 'all 150ms ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'rgba(228,87,46,0.4)'
              e.currentTarget.style.color = 'var(--color-alert)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--color-line-2)'
              e.currentTarget.style.color = 'var(--color-mist)'
            }}
          >
            ✕ New project
          </button>
        )}
      </div>
    </aside>
  )
}
