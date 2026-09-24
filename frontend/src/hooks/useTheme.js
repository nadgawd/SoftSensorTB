/**
 * Light / dark theme with localStorage persistence.
 * Applies `data-theme` on <html> so CSS variables switch.
 * Visitors start in light mode; only a theme picked with the toggle is saved.
 */
import { useCallback, useEffect, useState } from 'react'

// A new key: the old `sst_theme` was written on every visit, so it records the
// previous dark default rather than anyone's choice. Keep in sync with index.html.
const STORAGE_KEY = 'sst_theme_choice'
const DEFAULT = 'light'

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
  }, [theme])

  const setTheme = useCallback((next) => {
    setThemeState((prev) => {
      const value = typeof next === 'function' ? next(prev) : next
      try {
        localStorage.setItem(STORAGE_KEY, value)
      } catch {
        /* ignore */
      }
      return value
    })
  }, [])

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))
  }, [setTheme])

  return { theme, setTheme, toggleTheme, isLight: theme === 'light' }
}
