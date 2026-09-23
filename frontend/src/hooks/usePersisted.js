/**
 * usePersisted — useState that automatically syncs to localStorage.
 *
 * On first render it reads from localStorage, then writes on every change.
 * Works with any JSON-serializable value.
 *
 * @param {string}  key           localStorage key
 * @param {*}       defaultValue  initial value if nothing is stored
 * @param {object}  [opts]
 * @param {number}  [opts.maxAge] max age in ms — ignores stored values older than this
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { notifyProjectChange } from '../lib/projectSync'

const PREFIX = 'sst_v1_'  // prefix + version bump to auto-invalidate old format

export function usePersisted(key, defaultValue, { maxAge } = {}) {
  const storageKey = PREFIX + key

  function read() {
    try {
      const raw = localStorage.getItem(storageKey)
      if (!raw) return defaultValue
      const { value, ts } = JSON.parse(raw)
      if (maxAge && Date.now() - ts > maxAge) {
        localStorage.removeItem(storageKey)
        return defaultValue
      }
      return value
    } catch {
      return defaultValue
    }
  }

  const [state, setStateRaw] = useState(read)
  const latestState = useRef(state)
  latestState.current = state

  const setState = useCallback((update) => {
    setStateRaw((prev) => {
      const next = typeof update === 'function' ? update(prev) : update
      try {
        localStorage.setItem(storageKey, JSON.stringify({ value: next, ts: Date.now() }))
      } catch {
        // localStorage full or unavailable — still update React state
      }
      notifyProjectChange()
      return next
    })
  }, [storageKey])

  // Re-sync if key changes (e.g. dynamic keys)
  useEffect(() => {
    const stored = read()
    if (stored !== latestState.current) {
      setStateRaw(stored)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey])

  function clear() {
    localStorage.removeItem(storageKey)
    setStateRaw(defaultValue)
  }

  return [state, setState, clear]
}

/**
 * Clear all SST session keys from localStorage.
 * Call when the user explicitly starts a new project.
 */
export function clearPersistedSession() {
  for (const key of Object.keys(localStorage)) {
    if (key.startsWith(PREFIX)) {
      localStorage.removeItem(key)
    }
  }
}

/**
 * Returns true if there is a saved session in localStorage.
 */
export function hasPersistedSession() {
  return Object.keys(localStorage).some((k) => k.startsWith(PREFIX) && k.includes('datasetVersionId'))
}
