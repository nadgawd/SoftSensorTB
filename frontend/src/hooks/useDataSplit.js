import { useEffect, useState } from 'react'
import { errorDetail, fetchDataSplit } from '../api/client'

const EMPTY = { split: null, outdated: null, outdatedReason: null, loading: false, error: null }

/**
 * The saved train / validation / test split that fits the active version (for `target`, when set).
 * Bump `revision` after a split is saved to fetch it again.
 */
export function useDataSplit(datasetVersionId, target, revision) {
  const [state, setState] = useState(EMPTY)

  useEffect(() => {
    if (!datasetVersionId) {
      setState(EMPTY)
      return undefined
    }
    let cancelled = false
    setState((prev) => ({ ...prev, loading: true, error: null }))
    fetchDataSplit(datasetVersionId, { target })
      .then((data) => {
        if (cancelled) return
        setState({
          split: data.split || null,
          outdated: data.outdated || null,
          outdatedReason: data.outdated_reason || null,
          loading: false,
          error: null,
        })
      })
      .catch((err) => {
        if (!cancelled) setState({ ...EMPTY, error: errorDetail(err, 'Could not load the split.') })
      })
    return () => {
      cancelled = true
    }
  }, [datasetVersionId, target, revision])

  return state
}
