/**
 * Light / dark theme with localStorage persistence.
 * Applies `data-theme` on <html> so CSS variables switch.
 */
import { useCallback, useEffect, useState } from 'react'

const STORAGE_KEY = 'sst_theme'
const DEFAULT = 'dark'

function readTheme() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    /* ignore */
  }
  return DEFAULT
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme)
}

// Apply before React paints to avoid flash
if (typeof document !== 'undefined') {
  applyTheme(readTheme())
}

export function useTheme() {
  const [theme, setThemeState] = useState(readTheme)

  useEffect(() => {
    applyTheme(theme)
    try {
      localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      /* ignore */
    }
  }, [theme])

  const setTheme = useCallback((next) => {
    setThemeState((prev) => (typeof next === 'function' ? next(prev) : next))
  }, [])

  const toggleTheme = useCallback(() => {
    setThemeState((prev) => (prev === 'dark' ? 'light' : 'dark'))
  }, [])

  return { theme, setTheme, toggleTheme, isLight: theme === 'light' }
}
