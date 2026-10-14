import { useMemo, useState } from 'react'
import { errorDetail, uploadDataset } from '../../api/client'
import { columnChoices, isVisible } from '../../lib/manualForm'

const OPTION_WORDS = {
  acf: 'ACF', adasyn: 'ADASYN', adf: 'ADF', anova: 'ANOVA', ecdf: 'ECDF', f: 'F', iqr: 'IQR',
  kde: 'KDE', kfold: 'K-fold', knn: 'KNN', lda: 'LDA', lsa: 'LSA', mi: 'MI', pacf: 'PACF',
  pca: 'PCA', pchip: 'PCHIP', qq: 'Q-Q', r2: 'R²', rmse: 'RMSE', shap: 'SHAP', smote: 'SMOTE',
  svd: 'SVD', 't-test': 't-test', tfidf: 'TF-IDF', tsne: 't-SNE', woe: 'WoE', '2d': '2D', '3d': '3D',
}

function optionLabel(value) {
  const text = String(value)
  if (text === text.toUpperCase()) return text
  const words = text.replace(/^t_test/, 't-test').split('_')
  const first = OPTION_WORDS[words[0]]
  const label = words.map((w) => OPTION_WORDS[w] ?? w).join(' ')
  return first ? label : label.charAt(0).toUpperCase() + label.slice(1)
}

function rangeHint(field) {
  const lo = field.min ?? field.exclusive_min
  const hi = field.max ?? field.exclusive_max
  if (lo === undefined && hi === undefined) return null
  const loText = lo === undefined ? '' : `${field.min === undefined ? '>' : '≥'} ${lo}`
  const hiText = hi === undefined ? '' : `${field.max === undefined ? '<' : '≤'} ${hi}`
  return [loText, hiText].filter(Boolean).join(', ')
}

function SelectInput({ field, value, onChange, id }) {
  const hasDefault = field.default !== undefined && field.default !== null
  return (
    <select id={id} className="input-base manual-input" value={value} onChange={(e) => onChange(e.target.value)}>
      {!field.required && !hasDefault && <option value="">Default</option>}
      {field.required && !value && <option value="" disabled>Choose…</option>}
      {field.options.map((opt) => (
        <option key={String(opt)} value={opt}>{optionLabel(opt)}</option>
      ))}
    </select>
  )
}

function NumberInput({ field, value, onChange, id }) {
  const placeholder = field.nullable && (field.default === null || field.default === undefined) ? 'Auto' : ''
  return (
    <input
      id={id}
      type="number"
      className="input-base manual-input"
      value={value}
      step={field.widget === 'integer' ? 1 : 'any'}
      min={field.min}
      max={field.max}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
    />
  )
}

function ColumnSelect({ field, value, onChange, choices, id }) {
  const missing = value && !choices.includes(value)
  return (
    <select id={id} className="input-base manual-input" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">{field.required ? 'Choose a column…' : 'None'}</option>
      {missing && <option value={value}>{value}</option>}
      {choices.map((c) => <option key={c} value={c}>{c}</option>)}
    </select>
  )
}

function ColumnPicker({ field, value, onChange, choices }) {
  const [filter, setFilter] = useState('')
  const selected = useMemo(() => new Set(value || []), [value])
  const shown = filter ? choices.filter((c) => c.toLowerCase().includes(filter.toLowerCase())) : choices

  function toggle(col) {
    const next = new Set(selected)
    if (next.has(col)) next.delete(col)
    else next.add(col)
    onChange(choices.filter((c) => next.has(c)))
  }

  if (!choices.length) {
    return <p className="manual-hint">{field.numeric_only ? 'No numeric columns in this dataset.' : 'No columns.'}</p>
  }
  return (
    <div className="manual-picker">
      <div className="manual-picker-bar">
        <span className="manual-hint">
          {selected.size ? `${selected.size} of ${choices.length} selected` : (field.required ? 'None selected' : 'None selected (all applicable)')}
        </span>
        {choices.length > 10 && (
          <input
            className="input-base manual-input manual-picker-filter"
            placeholder="Filter…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        )}
        <button type="button" className="manual-link" onClick={() => onChange([...choices])}>All</button>
        <button type="button" className="manual-link" onClick={() => onChange([])}>None</button>
      </div>
      <div className="manual-chips">
        {shown.map((col) => (
          <button
            key={col}
            type="button"
            className={`manual-chip${selected.has(col) ? ' selected' : ''}`}
            aria-pressed={selected.has(col)}
            onClick={() => toggle(col)}
          >
            {col}
          </button>
        ))}
      </div>
    </div>
  )
}

/** Rows of from → to: a column → its new name, or (`freeKeys`) a typed value → its replacement. */
function MappingInput({ value, onChange, choices, freeKeys = false }) {
  const rows = value?.length ? value : [{ from: '', to: '' }]
  function update(i, patch) {
    onChange(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  }
  return (
    <div className="manual-mapping">
      {rows.map((row, i) => (
        <div key={i} className="manual-mapping-row">
          {freeKeys ? (
            <input
              className="input-base manual-input"
              placeholder="Value"
              value={row.from}
              onChange={(e) => update(i, { from: e.target.value })}
            />
          ) : (
            <select className="input-base manual-input" value={row.from} onChange={(e) => update(i, { from: e.target.value })}>
              <option value="">Column…</option>
              {choices.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          )}
          <span className="manual-hint">→</span>
          <input
            className="input-base manual-input"
            placeholder={freeKeys ? 'Becomes' : 'New name'}
            value={row.to}
            onChange={(e) => update(i, { to: e.target.value })}
          />
          <button
            type="button"
            className="manual-link"
            aria-label="Remove row"
            onClick={() => onChange(rows.length > 1 ? rows.filter((_, j) => j !== i) : [{ from: '', to: '' }])}
          >
            ✕
          </button>
        </div>
      ))}
      <button type="button" className="manual-link" onClick={() => onChange([...rows, { from: '', to: '' }])}>
        {freeKeys ? '+ Add value' : '+ Add column'}
      </button>
    </div>
  )
}

/** Another of the user's datasets, by id; a file can be uploaded here without opening it. */
function DatasetSelect({ value, onChange, schema, id }) {
  const [uploaded, setUploaded] = useState([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const known = [...(schema?.datasets || []).filter((d) => !d.current), ...uploaded]
  const options = known.filter((d, i) => known.findIndex((o) => o.dataset_id === d.dataset_id) === i)
  const missing = value && !options.some((d) => d.dataset_id === value)

  async function upload(file) {
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      const data = await uploadDataset(file, { keepVersionId: schema?.dataset_version_id })
      setUploaded((prev) => [...prev, { dataset_id: data.dataset_id, file_name: data.file_name }])
      onChange(data.dataset_id)
    } catch (err) {
      setError(errorDetail(err, 'Upload failed.'))
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="manual-dataset">
      <select id={id} className="input-base manual-input" value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">{options.length ? 'Choose a dataset…' : 'No other datasets yet'}</option>
        {missing && <option value={value}>{value}</option>}
        {options.map((d) => <option key={d.dataset_id} value={d.dataset_id}>{d.file_name}</option>)}
      </select>
      <label className="manual-link manual-upload">
        {uploading ? 'Uploading…' : 'Upload another file…'}
        <input
          type="file"
          accept=".csv,.txt,.parquet,.pq"
          hidden
          disabled={uploading}
          onChange={(e) => {
            upload(e.target.files?.[0])
            e.target.value = ''
          }}
        />
      </label>
      {error && <p className="manual-error">{error}</p>}
    </div>
  )
}

function ModelSelect({ value, onChange, models, id }) {
  return (
    <select id={id} className="input-base manual-input" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">Latest model</option>
      {(models || []).map((m) => (
        <option key={m.model_id} value={m.model_id}>
          {m.algorithm} → {m.target_column}
          {typeof m.r2_score === 'number' ? ` · R² ${m.r2_score.toFixed(3)}` : ''} · {String(m.model_id).slice(0, 8)}
        </option>
      ))}
    </select>
  )
}

function FieldControl({ field, value, onChange, schema, values, id }) {
  if (!schema && ['column', 'columns', 'mapping', 'dataset'].includes(field.widget)) {
    return <p className="manual-hint">Columns appear once a dataset is loaded.</p>
  }
  switch (field.widget) {
    case 'select':
      return <SelectInput field={field} value={value} onChange={onChange} id={id} />
    case 'integer':
    case 'number':
      return <NumberInput field={field} value={value} onChange={onChange} id={id} />
    case 'checkbox':
      return (
        <label className="manual-check">
          <input id={id} type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
          <span>{field.label}</span>
        </label>
      )
    case 'column':
      return <ColumnSelect field={field} value={value} onChange={onChange} choices={columnChoices(field, schema)} id={id} />
    case 'columns':
      return <ColumnPicker field={field} value={value} onChange={onChange} choices={columnChoices(field, schema, values)} />
    case 'mapping':
      return <MappingInput value={value} onChange={onChange} choices={columnChoices(field, schema)} />
    case 'value_mapping':
      return <MappingInput value={value} onChange={onChange} choices={[]} freeKeys />
    case 'dataset':
      return <DatasetSelect value={value} onChange={onChange} schema={schema} id={id} />
    case 'model':
      return <ModelSelect value={value} onChange={onChange} models={schema?.models} id={id} />
    case 'int_list':
      return (
        <input id={id} className="input-base manual-input" placeholder="e.g. 1, 2, 3" value={value} onChange={(e) => onChange(e.target.value)} />
      )
    case 'text_list':
      return (
        <input
          id={id}
          className="input-base manual-input"
          placeholder="Comma-separated"
          value={(value || []).join(', ')}
          onChange={(e) => onChange(e.target.value.split(',').map((s) => s.trim()))}
        />
      )
    case 'code':
      return (
        <textarea
          id={id}
          className="input-base manual-input manual-code"
          rows={field.name === 'sql_query' ? 4 : 2}
          spellCheck={false}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      )
    default:
      return <input id={id} className="input-base manual-input" value={value} onChange={(e) => onChange(e.target.value)} />
  }
}

function Field({ tool, field, value, values, onChange, schema, error }) {
  const id = `manual-${tool.name}-${field.name}`
  const wide = ['columns', 'mapping', 'value_mapping', 'code'].includes(field.widget)
  const hint = rangeHint(field)
  return (
    <div className={`manual-field${wide ? ' wide' : ''}`}>
      {field.widget !== 'checkbox' && (
        <label htmlFor={id} className="manual-label">
          {field.label}
          {field.required && <span className="manual-required"> *</span>}
          {hint && <span className="manual-range">{hint}</span>}
        </label>
      )}
      <FieldControl field={field} value={value} onChange={onChange} schema={schema} values={values} id={id} />
      {field.description && <p className="manual-hint manual-desc" title={field.description}>{field.description}</p>}
      {error && <p className="manual-error">{error}</p>}
    </div>
  )
}

/** The controls for one tool's parameters; hidden fields depend on the options chosen. */
export default function ToolForm({ tool, values, onChange, schema, errors }) {
  const visible = tool.fields.filter((f) => isVisible(f, values))
  const basic = visible.filter((f) => !f.advanced)
  const advanced = visible.filter((f) => f.advanced)
  const advancedError = advanced.some((f) => errors[f.name])

  const render = (field) => (
    <Field
      key={field.name}
      tool={tool}
      field={field}
      value={values[field.name]}
      values={values}
      onChange={(v) => onChange(field.name, v)}
      schema={schema}
      error={errors[field.name]}
    />
  )

  if (!visible.length) {
    return <p className="manual-hint">This tool has no parameters.</p>
  }
  return (
    <>
      {basic.length > 0 && <div className="manual-grid">{basic.map(render)}</div>}
      {advanced.length > 0 && (
        <details className="manual-advanced" open={advancedError || undefined}>
          <summary>Advanced settings ({advanced.length})</summary>
          <div className="manual-grid">{advanced.map(render)}</div>
        </details>
      )}
    </>
  )
}
