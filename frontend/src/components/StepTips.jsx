/**
 * Contextual tip cards for the active step.
 */
export default function StepTips({ tips = [] }) {
  if (!tips.length) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
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
  )
}
