import { useCallback, useEffect, useState } from 'react'
import { errorDetail, fetchManualTools, fetchVersionSchema } from '../api/client'

let specRequest = null

function loadSpec() {
  if (!specRequest) {
    specRequest = fetchManualTools().catch((err) => {
      specRequest = null
      throw err
    })
  }
  return specRequest
}

/** Tool forms from the backend, fetched once per page load. */
export function useManualTools() {
  const [state, setState] = useState({ tools: null, error: null })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let cancelled = false
    loadSpec()
      .then((data) => !cancelled && setState({ tools: data.tools, error: null }))
      .catch((err) => !cancelled && setState({ tools: null, error: errorDetail(err) }))
    return () => {
      cancelled = true
    }
  }, [attempt])

  const retry = useCallback(() => {
    setState({ tools: null, error: null })
    setAttempt((n) => n + 1)
  }, [])

  return { ...state, retry }
}

/** Columns, dtypes and trained models of the active dataset version. */
export function useDatasetSchema(datasetVersionId, refreshKey) {
  const [state, setState] = useState({ schema: null, error: null, loading: false })

  useEffect(() => {
    if (!datasetVersionId) {
      setState({ schema: null, error: null, loading: false })
      return undefined
    }
    let cancelled = false
    setState((prev) => ({ ...prev, error: null, loading: true }))
    fetchVersionSchema(datasetVersionId)
      .then((schema) => !cancelled && setState({ schema, error: null, loading: false }))
      .catch((err) => !cancelled && setState({ schema: null, error: errorDetail(err), loading: false }))
    return () => {
      cancelled = true
    }
  }, [datasetVersionId, refreshKey])

  return state
}
