/**
 * usePlotHistory — every plot the assistant generated in this project, newest last.
 *
 * Persisted like usePersisted (same prefix, so clearPersistedSession wipes it),
 * but plots can be large: when localStorage runs out of room the oldest plots
 * are dropped from storage until the rest fit. They stay in memory until reload.
 */
import { useCallback, useState } from 'react'
import { notifyProjectChange } from '../lib/projectSync'

const KEY = 'sst_v1_plotHistory'
const LEGACY_KEY = 'sst_v1_plotData'
const MAX_PLOTS = 20
const EMPTY = { plots: [], activeId: null }

export function plotTitle(data) {
  const title = data?.layout?.title
  const text = typeof title === 'object' ? title?.text : title
  return (typeof text === 'string' && text.replace(/<[^>]+>/g, '').trim()) || 'Untitled plot'
}

function makeEntry(data, createdAt = Date.now()) {
  return { id: `${createdAt}-${Math.random().toString(36).slice(2, 8)}`, title: plotTitle(data), createdAt, data }
}

function readStored(key) {
  try {
    const { value } = JSON.parse(localStorage.getItem(key) || 'null') || {}
    return value ?? null
  } catch {
    return null
  }
}

function load() {
  const stored = readStored(KEY)
  if (stored && Array.isArray(stored.plots)) return stored
  // Sessions from before plot history kept only the plot on the canvas.
  const legacy = readStored(LEGACY_KEY)
  if (!legacy?.data) return EMPTY
  const entry = makeEntry(legacy)
  const migrated = { plots: [entry], activeId: entry.id }
  save(migrated)
  localStorage.removeItem(LEGACY_KEY)
  return migrated
}

function save(state) {
  let plots = state.plots
  for (;;) {
    try {
      localStorage.setItem(KEY, JSON.stringify({ value: { ...state, plots }, ts: Date.now() }))
      notifyProjectChange()
      return
    } catch {
      if (plots.length === 0) return
      plots = plots.slice(1)
    }
  }
}

export function usePlotHistory() {
  const [state, setStateRaw] = useState(load)

  const update = useCallback((fn) => {
    setStateRaw((prev) => {
      const next = fn(prev)
      save(next)
      return next
    })
  }, [])

  const addPlot = useCallback((data) => {
    update((prev) => {
      const entry = makeEntry(data)
      return { plots: [...prev.plots, entry].slice(-MAX_PLOTS), activeId: entry.id }
    })
  }, [update])

  const selectPlot = useCallback((id) => {
    update((prev) => ({ ...prev, activeId: id }))
  }, [update])

  const removePlot = useCallback((id) => {
    update((prev) => {
      const index = prev.plots.findIndex((p) => p.id === id)
      const plots = prev.plots.filter((p) => p.id !== id)
      if (prev.activeId !== id) return { ...prev, plots }
      // Show the neighbour that slides into the removed plot's place.
      const neighbour = plots[Math.min(index, plots.length - 1)]
      return { plots, activeId: neighbour?.id ?? null }
    })
  }, [update])

  const clearActive = useCallback(() => update((prev) => ({ ...prev, activeId: null })), [update])
  const clearAll = useCallback(() => update(() => EMPTY), [update])

  const activePlot = state.plots.find((p) => p.id === state.activeId) || null

  return {
    plots: state.plots,
    activePlot,
    addPlot,
    selectPlot,
    removePlot,
    clearActive,
    clearAll,
  }
}
