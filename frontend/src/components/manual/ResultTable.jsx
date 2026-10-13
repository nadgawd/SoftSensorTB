function cell(value) {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toPrecision(5).replace(/\.?0+$/, '')
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

/** Tabular output of a tool (statistics, SQL results, model list). */
export default function ResultTable({ table, onClose }) {
  const rows = table?.rows || []
  const columns = [...new Set(rows.flatMap((r) => Object.keys(r)))]
  return (
    <section className="panel manual-panel" aria-label={table.title}>
      <div className="manual-head">
        <p className="manual-title">{table.title}</p>
        <span className="manual-hint">{rows.length} row{rows.length === 1 ? '' : 's'}</span>
        <button type="button" className="manual-link manual-close" onClick={onClose} aria-label="Close result">✕</button>
      </div>
      {rows.length === 0 ? (
        <p className="manual-hint">No rows returned.</p>
      ) : (
        <div className="manual-table-wrap">
          <table className="data-table">
            <thead>
              <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i}>
                  {columns.map((c) => <td key={c}>{cell(row[c])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
