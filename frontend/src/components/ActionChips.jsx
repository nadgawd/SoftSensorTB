/**
 * Clickable suggested action chips + contextual tip cards.
 * Clicking an action chip sends the text to the AI assistant.
 */
export default function ActionChips({ actions = [], tips = [], onAction, label = 'Suggested Actions' }) {
  if (!actions.length && !tips.length) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {/* Tips */}
      {tips.length > 0 && (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
          }}
        >
          {tips.map((tip, i) => (
            <div
              key={i}
              style={{
                padding: '10px 14px',
                borderRadius: '10px',
                background: 'var(--panel-elevated-bg)',
                border: '1px solid var(--color-line)',
                fontSize: '0.8125rem',
                color: 'var(--color-foam-2)',
                lineHeight: 1.55,
              }}
            >
              <span dangerouslySetInnerHTML={{ __html: tip.replace(/\*\*(.*?)\*\*/g, '<strong style="color:var(--color-foam)">$1</strong>') }} />
            </div>
          ))}
        </div>
      )}

      {/* Action chips */}
      {actions.length > 0 && (
        <div>
          <p
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              color: 'var(--color-mist)',
              margin: '0 0 8px',
            }}
          >
            {label}
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {actions.map((action, i) => (
              <button
                key={i}
                type="button"
                onClick={() => onAction?.(action)}
                style={{
                  padding: '7px 13px',
                  borderRadius: '99px',
                  border: '1px solid color-mix(in srgb, var(--color-signal) 35%, transparent)',
                  background: 'color-mix(in srgb, var(--color-signal) 8%, transparent)',
                  color: 'var(--color-signal)',
                  fontSize: '0.8125rem',
                  cursor: 'pointer',
                  transition: 'all 150ms ease',
                  textAlign: 'left',
                  lineHeight: 1.4,
                  fontFamily: 'var(--font-display)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--color-signal)'
                  e.currentTarget.style.background = 'color-mix(in srgb, var(--color-signal) 14%, transparent)'
                  e.currentTarget.style.color = 'var(--color-signal-2)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'color-mix(in srgb, var(--color-signal) 35%, transparent)'
                  e.currentTarget.style.background = 'color-mix(in srgb, var(--color-signal) 8%, transparent)'
                  e.currentTarget.style.color = 'var(--color-signal)'
                }}
              >
                <span style={{ marginRight: '5px', fontSize: '11px', opacity: 0.6 }}>↗</span>
                {action}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
