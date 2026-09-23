/**
 * Sun / moon toggle for light ↔ dark theme.
 */
export default function ThemeToggle({ theme, onToggle }) {
  const isLight = theme === 'light'
  return (
    <button
      type="button"
      onClick={onToggle}
      title={isLight ? 'Switch to dark mode' : 'Switch to light mode'}
      aria-label={isLight ? 'Switch to dark mode' : 'Switch to light mode'}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '36px',
        height: '36px',
        borderRadius: '8px',
        border: '1px solid var(--color-line-2)',
        background: 'var(--header-btn-bg)',
        color: 'var(--color-foam)',
        cursor: 'pointer',
        fontSize: '16px',
        lineHeight: 1,
        transition: 'background 150ms ease, border-color 150ms ease',
        flexShrink: 0,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'var(--header-btn-bg-hover)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'var(--header-btn-bg)'
      }}
    >
      {isLight ? (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      )}
    </button>
  )
}
