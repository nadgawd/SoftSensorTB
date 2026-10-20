import { Clock, Database, FlaskConical, TriangleAlert } from 'lucide-react'
import { formatCount, formatDuration, formatTime } from '../utils/format'

/** One line under the header: size, time span and sampling, lab-sample coverage, quality. */
export default function DataSummaryBar({ profile, rowCount, columnCount, targetVariable, onOpenQuality }) {
  const data = profile?.data
  const rows = data?.row_count ?? rowCount
  const cols = data?.column_count ?? columnCount
  const time = data?.time
  const target = data?.target
  const summary = data?.summary
  const issues = summary ? summary.problem + summary.warning : 0

  return (
    <div className="data-summary" aria-label="Dataset summary">
      <span className="data-summary-item">
        <Database size={12} aria-hidden="true" />
        {formatCount(rows)} rows × {cols ?? '?'} tags
      </span>
      {time && (
        <span className="data-summary-item" title={`Time column: ${data.time_column}`}>
          <Clock size={12} aria-hidden="true" />
          {formatTime(time.start)} → {formatTime(time.end)}
          {time.median_interval_s != null && <> · every {formatDuration(time.median_interval_s)}</>}
        </span>
      )}
      {target ? (
        <span className="data-summary-item" title="Rows where the target has a value (lab samples)">
          <FlaskConical size={12} aria-hidden="true" />
          {target.column}: {formatCount(target.samples)} samples ({(target.share * 100).toFixed(target.share < 0.1 ? 1 : 0)}%)
        </span>
      ) : !targetVariable && (
        <span className="data-summary-item data-summary-muted">
          <FlaskConical size={12} aria-hidden="true" /> No target yet
        </span>
      )}
      {issues > 0 && (
        <button type="button" className={`data-summary-issues${summary.problem ? ' problem' : ''}`} onClick={onOpenQuality}
          title="Open the data quality check">
          <TriangleAlert size={12} aria-hidden="true" />
          {summary.problem > 0 && `${summary.problem} problem${summary.problem > 1 ? 's' : ''}`}
          {summary.problem > 0 && summary.warning > 0 && ' · '}
          {summary.warning > 0 && `${summary.warning} to check`}
        </button>
      )}
      {profile?.loading && <span className="data-summary-muted">updating…</span>}
    </div>
  )
}
