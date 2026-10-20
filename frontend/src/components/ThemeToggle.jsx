import { Moon, Sun } from 'lucide-react'

/**
 * Sun / moon toggle for light ↔ dark theme.
 */
export default function ThemeToggle({ theme, onToggle }) {
  const isLight = theme === 'light'
  const label = isLight ? 'Switch to dark mode' : 'Switch to light mode'
  return (
    <button type="button" className="app-header-btn app-icon-btn" onClick={onToggle} title={label} aria-label={label}>
      {isLight ? <Moon size={15} /> : <Sun size={15} />}
    </button>
  )
}
