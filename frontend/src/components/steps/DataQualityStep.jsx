import { useMemo, useState } from 'react'
import { formatCount, formatDuration, formatTime } from '../../utils/format'

const STATUS = {
  ok: { label: 'OK', cls: 'badge-success' },
  warning: { label: 'Check', cls: 'badge-ember' },
  problem: { label: 'Problem', cls: 'badge-alert' },
}
const FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'problem', label: 'Problems' },
  { id: 'warning', label: 'Checks' },
  { id: 'ok', label: 'OK' },
]
const ORDER = { problem: 0, warning: 1, ok: 2 }

/** The tool that addresses an issue: [step, tool, label]. */
function fixFor(issue) {
  if (issue.includes('missing')) return ['quality', 'remove_missing_data', 'Handle missing values']
  if (issue.includes('outlier')) return ['quality', 'remove_outliers', 'Handle outliers']
  if (issue.includes('stored as text')) return ['ingestion', 'convert_types', 'Convert types']
  if (issue.includes('constant') || issue.includes('stuck')) return ['feature-selection', 'drop_columns', 'Drop the column']
  return null
}

function formatNumber(v) {
  if (v === null || v === undefined) return '—'
  const a = Math.abs(v)
  return a !== 0 && (a >= 1e5 || a < 1e-3) ? v.toExponential(2) : Number(v.toPrecision(4)).toString()
}

/**
 * Step 2 — Data quality: checks over every row of the active version.
 */
export default function DataQualityStep({ profile, targetVariable, onOpenTool }) {
  const [filter, setFilter] = useState('all')
  const { data, loading, error, retry } = profile || {}

  const columns = useMemo(() => {
    const cols = [...(data?.columns || [])].sort((a, b) => ORDER[a.status] - ORDER[b.status])
    return filter === 'all' ? cols : cols.filter((c) => c.status === filter)
  }, [data, filter])

  if (!data) {
    return (
      <div className="panel quality-panel">
        {error ? (
          <p className="manual-error">{error} <button type="button" className="btn-ghost" onClick={retry}>Retry</button></p>
        ) : (
          <p className="manual-hint">{loading ? 'Checking every row of the dataset…' : 'Upload a dataset in Step 1 to check its quality.'}</p>
        )}
      </div>
    )
  }

  const { time, target, summary } = data
  return (
    <div className="animate-rise quality">
      <section className="panel quality-panel" aria-label="Quality overview">
        <div className="quality-head">
          <p className="model-panel-title">Quality check · every row of the active version{loading ? ' · updating…' : ''}</p>
          <div className="quality-counts">
            {['problem', 'warning', 'ok'].map((level) => (
              <button
                key={level}
                type="button"
                className={`quality-count quality-${level}${filter === level ? ' active' : ''}`}
                onClick={() => setFilter((f) => (f === level ? 'all' : level))}
              >
                <strong>{summary[level]}</strong> {STATUS[level].label === 'Check' ? 'to check' : STATUS[level].label === 'OK' ? 'OK' : 'problems'}
              </button>
            ))}
          </div>
        </div>

        <dl className="quality-facts">
          <div>
            <dt>Size</dt>
            <dd>{formatCount(data.row_count)} rows × {data.column_count} tags · {formatCount(data.complete_rows)} rows with no missing value</dd>
          </div>
          <div>
            <dt>Time</dt>
            <dd>
              {time ? (
                <>
                  <code>{data.time_column}</code> · {formatTime(time.start)} → {formatTime(time.end)}
                  {time.median_interval_s != null && <> · every {formatDuration(time.median_interval_s)}</>}
                  {time.n_gaps > 0 && <span className="quality-warn"> · {time.n_gaps} gap{time.n_gaps > 1 ? 's' : ''}</span>}
                  {!time.sorted && <span className="quality-warn"> · not in time order</span>}
                  {time.duplicates > 0 && <span className="quality-warn"> · {time.duplicates} repeated timestamps</span>}
                  {time.irregular_share > 0.05 && <span className="quality-warn"> · {Math.round(time.irregular_share * 100)}% irregular steps</span>}
                </>
              ) : 'No time column found. Pick one in the Variables table (Step 1) to check sampling and gaps.'}
            </dd>
          </div>
          <div>
            <dt>Target</dt>
            <dd>
              {target ? (
                <>
                  <code>{target.column}</code> · {formatCount(target.samples)} lab samples ({(target.share * 100).toFixed(1)}% of rows)
                  {target.median_interval_s != null && <> · about one every {formatDuration(target.median_interval_s)}</>}
                  {target.first && <> · {formatTime(target.first)} → {formatTime(target.last)}</>}
                </>
              ) : targetVariable ? 'Checking…' : 'No target yet. Pick one in the Variables table (Step 1) to see its lab-sample coverage.'}
            </dd>
          </div>
        </dl>

        {time?.gaps?.length > 0 && (
          <details className="quality-gaps">
            <summary>Largest gaps in the time column</summary>
            <ul>
              {time.gaps.map((g) => (
                <li key={g.start}>{formatTime(g.start)} → {formatTime(g.end)} · {formatDuration(g.duration_s)}</li>
              ))}
            </ul>
          </details>
        )}
      </section>

      <section className="panel quality-table-panel" aria-label="Column checks">
        <div className="quality-filter" role="tablist" aria-label="Show columns">
          {FILTERS.map((f) => (
            <button key={f.id} type="button" role="tab" aria-selected={filter === f.id}
              className={`split-tab${filter === f.id ? ' active' : ''}`} onClick={() => setFilter(f.id)}>
              {f.label}
            </button>
          ))}
        </div>
        <div className="manual-table-wrap quality-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Tag</th><th>Type</th><th>Missing</th><th>Range</th><th>Mean ± SD</th>
                <th title="Share of readings in runs of 20+ identical values">Flat runs</th><th>Outliers</th><th>Status</th><th>Findings</th>
              </tr>
            </thead>
            <tbody>
              {columns.map((c) => (
                <tr key={c.name}>
                  <td className="quality-name">
                    {c.name}
                    {c.role && <span className={`badge ${c.role === 'target' ? 'badge-signal' : 'badge-mist'} quality-role`}>{c.role}</span>}
                  </td>
                  <td>{c.kind}</td>
                  <td>{c.missing ? `${formatCount(c.missing)} (${(c.missing_share * 100).toFixed(1)}%)` : '—'}</td>
                  <td>{c.min != null ? `${formatNumber(c.min)} – ${formatNumber(c.max)}` : '—'}</td>
                  <td>{c.mean != null ? `${formatNumber(c.mean)} ± ${formatNumber(c.std)}` : '—'}</td>
                  <td>{c.flat_share != null ? `${Math.round(c.flat_share * 100)}%` : '—'}</td>
                  <td>{c.outliers != null ? formatCount(c.outliers) : '—'}</td>
                  <td><span className={`badge ${STATUS[c.status].cls}`}>{STATUS[c.status].label}</span></td>
                  <td className="quality-issues">
                    {c.issues.length === 0 && (c.role === 'target' ? 'Sparse lab values are expected' : '—')}
                    {c.issues.map((issue) => {
                      const fix = fixFor(issue)
                      return (
                        <span key={issue} className="quality-issue">
                          {issue}
                          {fix && onOpenTool && (
                            <button type="button" className="quality-fix" onClick={() => onOpenTool(fix[0], fix[1])}>
                              {fix[2]} →
                            </button>
                          )}
                        </span>
                      )
                    })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
