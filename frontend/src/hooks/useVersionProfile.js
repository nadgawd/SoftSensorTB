import { useEffect, useState } from 'react'
import { errorDetail, fetchVersionProfile } from '../api/client'

/** The quality profile of the active version (computed on the server over every row). */
export function useVersionProfile(datasetVersionId, { target, timeColumn } = {}) {
  const [state, setState] = useState({ data: null, loading: false, error: null })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    if (!datasetVersionId) {
      setState({ data: null, loading: false, error: null })
      return undefined
    }
    let active = true
    setState((prev) => ({ ...prev, loading: true, error: null }))
    fetchVersionProfile(datasetVersionId, { target, timeColumn })
      .then((data) => { if (active) setState({ data, loading: false, error: null }) })
      .catch((err) => { if (active) setState({ data: null, loading: false, error: errorDetail(err, 'Could not check the data.') }) })
    return () => { active = false }
  }, [datasetVersionId, target, timeColumn, attempt])

  return { ...state, retry: () => setAttempt((n) => n + 1) }
}
