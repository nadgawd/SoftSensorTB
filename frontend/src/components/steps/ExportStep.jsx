import { useEffect, useState } from 'react'
import { downloadExport, errorDetail, fetchVersionSchema } from '../../api/client'
import { formatBytes, formatCount } from '../../utils/format'

async function blobErrorDetail(err) {
  const data = err?.response?.data
  if (data instanceof Blob) {
    try {
      return JSON.parse(await data.text()).detail || 'Download failed.'
    } catch {
      return 'Download failed.'
    }
  }
  return errorDetail(err, 'Download failed.')
}

function modelLabel(m) {
  const score = m.r2_score != null ? `test R² ${Number(m.r2_score).toFixed(3)}`
    : m.cv?.r2 != null ? `CV R² ${Number(m.cv.r2).toFixed(3)}` : 'not tested'
  return `${m.algorithm} → ${m.target_column} · ${m.feature_columns?.length ?? 0} features · ${score} · ${m.model_id.slice(0, 8)}`
}

/**
 * Export step: the data, the model, the steps that produced them, and a report.
 */
export default function ExportStep({ datasetVersionId, modelMetrics, profile, targetVariable, timeColumn }) {
  const [models, setModels] = useState([])
  const [modelId, setModelId] = useState('')
  const [progress, setProgress] = useState({})
  const [error, setError] = useState(null)
  const data = profile?.data

  useEffect(() => {
    if (!datasetVersionId) return undefined
    let cancelled = false
    fetchVersionSchema(datasetVersionId)
      .then((schema) => {
        if (cancelled) return
        const list = schema.models || []
        setModels(list)
        setModelId((current) => (list.some((m) => m.model_id === current) ? current
          : list.find((m) => m.model_id === modelMetrics?.model_id)?.model_id || list[0]?.model_id || ''))
      })
      .catch(() => { if (!cancelled) setModels([]) })
    return () => { cancelled = true }
  }, [datasetVersionId, modelMetrics?.model_id])

  async function download(key, path, params) {
    setError(null)
    setProgress((p) => ({ ...p, [key]: 'Preparing…' }))
    try {
      await downloadExport(path, {
        params,
        onProgress: (fraction, loaded) => setProgress((p) => ({
          ...p,
          [key]: fraction != null ? `${Math.round(fraction * 100)}%` : formatBytes(loaded),
        })),
      })
    } catch (err) {
      setError(await blobErrorDetail(err))
    } finally {
      setProgress((p) => ({ ...p, [key]: null }))
    }
  }

  if (!datasetVersionId) {
    return <div className="panel export-panel"><p className="manual-hint">Upload a dataset in Step 1 to export it.</p></div>
  }

  const base = `/datasets/versions/${datasetVersionId}`
  const button = (key, label, path, params, disabled = false) => (
    <button type="button" className="btn-ghost export-btn" disabled={disabled || Boolean(progress[key])}
      onClick={() => download(key, path, params)}>
      {progress[key] || label}
    </button>
  )

  return (
    <div className="animate-rise panel export-panel">
      <p className="model-panel-title">Download from the active version</p>
      <div className="export-list">
        <div className="export-row">
          <div>
            <p className="export-name">Prepared dataset</p>
            <p className="export-desc">
              {data ? `All ${formatCount(data.row_count)} rows × ${data.column_count} columns` : 'Every row'} of the active version.
              Parquet keeps the column types and is much smaller.
            </p>
          </div>
          <div className="export-actions">
            {button('csv', 'CSV', `${base}/download`, { format: 'csv' })}
            {button('parquet', 'Parquet', `${base}/download`, { format: 'parquet' })}
          </div>
        </div>

        <div className="export-row">
          <div>
            <p className="export-name">Trained model</p>
            {models.length ? (
              <>
                <select className="manual-input export-select" value={modelId} onChange={(e) => setModelId(e.target.value)}>
                  {models.map((m) => <option key={m.model_id} value={m.model_id}>{modelLabel(m)}</option>)}
                </select>
                <p className="export-desc">
                  A joblib file. In Python: <code>a = joblib.load(path)</code>, then <code>a[&quot;model&quot;].predict(X[a[&quot;feature_columns&quot;]])</code>.
                </p>
              </>
            ) : (
              <p className="export-desc">No model trained on this dataset yet. Train one in the Model step.</p>
            )}
          </div>
          <div className="export-actions">
            {button('model', 'Model file', `/models/${modelId}/download`, undefined, !modelId)}
          </div>
        </div>

        <div className="export-row">
          <div>
            <p className="export-name">Pipeline recipe</p>
            <p className="export-desc">Every step from the upload to this version with its exact parameters, and the models trained along the way.</p>
          </div>
          <div className="export-actions">{button('recipe', 'JSON', `${base}/recipe`)}</div>
        </div>

        <div className="export-row">
          <div>
            <p className="export-name">Report</p>
            <p className="export-desc">One HTML page with the data summary, pipeline steps, model scores and quality findings.</p>
          </div>
          <div className="export-actions">
            {button('report', 'HTML', `${base}/report`, { target: targetVariable || undefined, time_column: timeColumn || undefined })}
          </div>
        </div>
      </div>
      {error && <p className="manual-error">{error}</p>}
    </div>
  )
}
