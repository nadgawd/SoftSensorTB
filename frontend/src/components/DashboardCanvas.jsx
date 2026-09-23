import { useCallback, useMemo, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import DynamicGraph from './DynamicGraph'
import {
  fetchDatasetHistory,
  rollbackVersion,
  uploadDataset,
} from '../api/client'

const TABS = [
  { id: 'preview', label: 'Data Preview' },
  { id: 'scorecard', label: 'Model Scorecard' },
  { id: 'history', label: 'Version History' },
]

function shortId(id) {
  if (!id) return '—'
  return `${id.slice(0, 8)}…`
}

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(4)
}

/**
 * Right-side workspace: dropzone, Plotly canvas, and bottom tabs.
 */
export default function DashboardCanvas({
  fileName,
  datasetId,
  datasetVersionId,
  plotData,
  tablePreview,
  modelMetrics,
  versionHistory,
  onUploadSuccess,
  onRollback,
  onHistoryRefresh,
}) {
  const [activeTab, setActiveTab] = useState('preview')
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [rollingBack, setRollingBack] = useState(false)

  const onDrop = useCallback(
    async (acceptedFiles) => {
      const file = acceptedFiles?.[0]
      if (!file) return
      setUploadError(null)
      setUploading(true)
      try {
        const result = await uploadDataset(file)
        onUploadSuccess?.(result)
        setActiveTab('preview')
      } catch (err) {
        const detail = err?.response?.data?.detail || err.message || 'Upload failed'
        setUploadError(typeof detail === 'string' ? detail : JSON.stringify(detail))
      } finally {
        setUploading(false)
      }
    },
    [onUploadSuccess],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      'text/csv': ['.csv'],
      'text/plain': ['.txt'],
      'application/vnd.apache.parquet': ['.parquet', '.pq'],
      'application/octet-stream': ['.parquet', '.pq', '.csv'],
    },
    disabled: uploading,
  })

  const columns = useMemo(() => {
    if (!tablePreview?.length) return []
    return Object.keys(tablePreview[0])
  }, [tablePreview])

  async function handleRollback(versionId) {
    if (!versionId || versionId === datasetVersionId || rollingBack) return
    setRollingBack(true)
    try {
      const result = await rollbackVersion(versionId)
      onRollback?.(result)
      if (result.dataset_id) {
        const history = await fetchDatasetHistory(result.dataset_id)
        onHistoryRefresh?.(history)
      }
      setActiveTab('preview')
    } catch (err) {
      const detail = err?.response?.data?.detail || err.message || 'Rollback failed'
      setUploadError(typeof detail === 'string' ? detail : JSON.stringify(detail))
    } finally {
      setRollingBack(false)
    }
  }

  return (
    <section className="flex h-full min-w-0 flex-1 flex-col gap-3">
      {/* Top panel: dropzone + active version badge */}
      <div className="panel-surface animate-rise grid gap-3 rounded-2xl p-4 md:grid-cols-[1fr_auto]">
        <div
          {...getRootProps()}
          className={`cursor-pointer rounded-xl border border-dashed px-4 py-5 transition ${
            isDragActive
              ? 'border-signal bg-signal/10'
              : 'border-line/90 bg-ink/35 hover:border-signal/50'
          }`}
        >
          <input {...getInputProps()} />
          <p className="font-display text-sm font-medium text-foam">
            {uploading
              ? 'Ingesting dataset…'
              : isDragActive
                ? 'Drop to upload'
                : 'Drop CSV / Parquet here'}
          </p>
          <p className="mt-1 text-xs text-mist">
            Creates a root dataset version under <span className="font-mono">/data_storage</span>
          </p>
          {fileName ? (
            <p className="mt-2 font-mono text-xs text-signal">{fileName}</p>
          ) : null}
        </div>

        <div className="flex min-w-[220px] flex-col justify-center rounded-xl bg-ink/45 px-4 py-3 ring-1 ring-line/80">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-mist">
            Active version
          </p>
          <p className="mt-1 font-mono text-sm text-foam" title={datasetVersionId || ''}>
            {datasetVersionId ? shortId(datasetVersionId) : 'No dataset loaded'}
          </p>
          {datasetId ? (
            <p className="mt-1 font-mono text-[11px] text-mist" title={datasetId}>
              dataset {shortId(datasetId)}
            </p>
          ) : null}
        </div>
      </div>

      {uploadError ? (
        <p className="rounded-xl bg-alert/10 px-3 py-2 text-xs text-alert ring-1 ring-alert/30">
          {uploadError}
        </p>
      ) : null}

      {/* Center visual canvas */}
      <div className="panel-surface min-h-[300px] flex-1 overflow-hidden rounded-2xl p-3">
        <div className="mb-2 flex items-center justify-between px-1">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-mist">
            Center visual canvas
          </p>
          {plotData ? (
            <span className="rounded-full bg-signal/15 px-2 py-0.5 font-mono text-[10px] text-signal ring-1 ring-signal/30">
              plot_data live
            </span>
          ) : null}
        </div>
        <div className="h-[calc(100%-1.5rem)] rounded-xl bg-ink/30 ring-1 ring-line/60">
          <DynamicGraph data={plotData} />
        </div>
      </div>

      {/* Bottom workspace tabs */}
      <div className="panel-surface flex min-h-[240px] flex-col rounded-2xl">
        <div className="flex gap-1 border-b border-line/80 px-2 pt-2">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`rounded-t-lg px-4 py-2 text-sm transition ${
                activeTab === tab.id
                  ? 'bg-panel-2 text-foam'
                  : 'text-mist hover:text-foam'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-auto p-4">
          {activeTab === 'preview' ? (
            <DataPreviewTable columns={columns} rows={tablePreview} />
          ) : null}

          {activeTab === 'scorecard' ? (
            <ModelScorecard metrics={modelMetrics} />
          ) : null}

          {activeTab === 'history' ? (
            <VersionTimeline
              history={versionHistory}
              activeVersionId={datasetVersionId}
              onSelect={handleRollback}
              busy={rollingBack}
            />
          ) : null}
        </div>
      </div>
    </section>
  )
}

function DataPreviewTable({ columns, rows }) {
  if (!rows?.length) {
    return (
      <p className="text-sm text-mist">
        No preview yet. Upload a file or run a chat tool that returns table rows.
      </p>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full border-collapse text-left text-xs">
        <thead>
          <tr className="border-b border-line">
            {columns.map((col) => (
              <th
                key={col}
                className="whitespace-nowrap px-3 py-2 font-mono font-medium text-signal"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 50).map((row, idx) => (
            <tr key={idx} className="border-b border-line/50 hover:bg-signal/5">
              {columns.map((col) => (
                <td key={col} className="whitespace-nowrap px-3 py-1.5 font-mono text-foam/90">
                  {formatCell(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 font-mono text-[11px] text-mist">
        Showing {Math.min(rows.length, 50)} row(s)
      </p>
    </div>
  )
}

function formatCell(value) {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4)
  return String(value)
}

function ModelScorecard({ metrics }) {
  if (!metrics) {
    return (
      <p className="text-sm text-mist">
        Train a soft sensor to populate R² and RMSE badges here.
      </p>
    )
  }

  return (
    <div className="animate-fade-in space-y-4">
      <div className="flex flex-wrap gap-3">
        <MetricBadge label="R²" value={formatMetric(metrics.r2_score)} accent="signal" />
        <MetricBadge label="RMSE" value={formatMetric(metrics.rmse)} accent="ember" />
        {metrics.algorithm ? (
          <MetricBadge label="Algorithm" value={metrics.algorithm} accent="mist" />
        ) : null}
      </div>
      {metrics.model_id ? (
        <p className="font-mono text-xs text-mist">
          model_id · {metrics.model_id}
        </p>
      ) : null}
      {metrics.feature_importances ? (
        <div>
          <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.16em] text-mist">
            Coefficients
          </p>
          <div className="grid gap-1 sm:grid-cols-2">
            {Object.entries(metrics.feature_importances).map(([name, value]) => (
              <div
                key={name}
                className="flex items-center justify-between rounded-lg bg-ink/40 px-3 py-2 ring-1 ring-line/70"
              >
                <span className="font-mono text-xs text-foam">{name}</span>
                <span className="font-mono text-xs text-signal">{formatMetric(value)}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  )
}

function MetricBadge({ label, value, accent }) {
  const color =
    accent === 'ember'
      ? 'text-ember ring-ember/30 bg-ember/10'
      : accent === 'mist'
        ? 'text-mist ring-line bg-ink/40'
        : 'text-signal ring-signal/30 bg-signal/10'

  return (
    <div className={`min-w-[120px] rounded-xl px-4 py-3 ring-1 ${color}`}>
      <p className="font-mono text-[10px] uppercase tracking-[0.16em] opacity-80">{label}</p>
      <p className="mt-1 font-display text-2xl font-semibold">{value}</p>
    </div>
  )
}

function VersionTimeline({ history, activeVersionId, onSelect, busy }) {
  if (!history?.length) {
    return (
      <p className="text-sm text-mist">
        Version lineage will appear here after upload and transformations.
      </p>
    )
  }

  return (
    <ol className="relative ml-2 space-y-0 border-l border-line pl-6">
      {[...history].reverse().map((node, index) => {
        const isActive = node.id === activeVersionId
        return (
          <li key={node.id} className="relative pb-6 last:pb-0">
            <button
              type="button"
              disabled={busy}
              onClick={() => onSelect(node.id)}
              className={`absolute -left-[1.55rem] top-1 h-3.5 w-3.5 rounded-full border-2 transition ${
                isActive
                  ? 'animate-pulse-node border-signal bg-signal'
                  : 'border-mist/50 bg-panel hover:border-ember hover:bg-ember'
              }`}
              title="Rollback to this version"
            />
            <button
              type="button"
              disabled={busy}
              onClick={() => onSelect(node.id)}
              className={`w-full rounded-xl px-3 py-2 text-left transition hover:bg-ink/40 ${
                isActive ? 'bg-signal/10 ring-1 ring-signal/30' : ''
              }`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs text-signal">{shortId(node.id)}</span>
                {isActive ? (
                  <span className="rounded-full bg-signal/20 px-2 py-0.5 font-mono text-[10px] text-signal">
                    active
                  </span>
                ) : (
                  <span className="font-mono text-[10px] text-mist">click to time-travel</span>
                )}
                {index === 0 ? (
                  <span className="font-mono text-[10px] text-ember">latest</span>
                ) : null}
              </div>
              <p className="mt-1 text-sm text-foam">{node.action_performed}</p>
              {node.created_at ? (
                <p className="mt-0.5 font-mono text-[11px] text-mist">
                  {new Date(node.created_at + (node.created_at.endsWith('Z') ? '' : 'Z')).toLocaleString()}
                </p>
              ) : null}
            </button>
          </li>
        )
      })}
    </ol>
  )
}
