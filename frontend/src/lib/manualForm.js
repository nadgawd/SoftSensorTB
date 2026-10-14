/**
 * Form state for manual tool runs.
 *
 * A form's values are its defaults (from the tool schema, or the workbench's
 * target / selected features) overlaid with what the user edited. Only the
 * edits are stored, so defaults keep following the workbench until the user
 * overrides them, and "Reset" just drops the edits.
 */

const COLUMN_WIDGETS = new Set(['column', 'columns'])

/** `show_when` is one set of conditions that must all hold, or a list of such sets of which any may hold. */
export function isVisible(field, values) {
  const when = field.show_when
  if (!when) return true
  const alternatives = Array.isArray(when) ? when : [when]
  return alternatives.some((conditions) =>
    Object.entries(conditions).every(([other, allowed]) => allowed.includes(values[other])),
  )
}

/** Columns a field may pick; a feature list never offers the form's target. */
function columnChoices(field, schema, values) {
  const cols = schema?.columns || []
  const names = (field.numeric_only ? cols.filter((c) => c.numeric) : cols).map((c) => c.name)
  const target = field.name === 'feature_columns' ? values?.target_column : null
  return target ? names.filter((c) => c !== target) : names
}

function defaultFeatures(schema, target) {
  return (schema?.columns || [])
    .filter((c) => c.numeric && c.name !== target && !c.name.startsWith('Predicted_'))
    .map((c) => c.name)
}

function appDefault(field, schema, app) {
  const choices = new Set(columnChoices(field, schema))
  if (field.app_default === 'target') {
    return app.targetVariable && choices.has(app.targetVariable) ? app.targetVariable : undefined
  }
  if (field.app_default === 'features') {
    const target = app.targetVariable
    const picked = (app.selectedFeatures || []).filter((c) => choices.has(c) && c !== target)
    return picked.length ? picked : defaultFeatures(schema, target).filter((c) => choices.has(c))
  }
  return undefined
}

/** The value a field shows before the user touches it. */
export function fieldDefault(field, schema, app) {
  const fromApp = appDefault(field, schema, app)
  if (fromApp !== undefined) return fromApp
  const d = field.default
  switch (field.widget) {
    case 'checkbox':
      return Boolean(d)
    case 'columns':
    case 'text_list':
      return Array.isArray(d) ? [...d] : []
    case 'int_list':
      return Array.isArray(d) ? d.join(', ') : ''
    case 'mapping':
    case 'value_mapping':
      return [{ from: '', to: '' }]
    case 'integer':
    case 'number':
      return d === null || d === undefined ? '' : String(d)
    default:
      return d === null || d === undefined ? '' : String(d)
  }
}

/** Drop edits that point at columns the active version no longer has. */
function pruneColumns(field, value, schema) {
  if (!schema || (!COLUMN_WIDGETS.has(field.widget) && field.widget !== 'mapping')) return value
  const names = new Set((schema.columns || []).map((c) => c.name))
  if (field.widget === 'column') return names.has(value) ? value : ''
  if (field.widget === 'columns') return (value || []).filter((c) => names.has(c))
  return (value || []).map((row) => (names.has(row.from) ? row : { ...row, from: '' }))
}

export function formValues(tool, edits, schema, app) {
  const values = {}
  for (const field of tool.fields) {
    values[field.name] = Object.prototype.hasOwnProperty.call(edits || {}, field.name)
      ? pruneColumns(field, edits[field.name], schema)
      : fieldDefault(field, schema, app)
  }
  return values
}

function checkRange(field, n) {
  if (field.min !== undefined && n < field.min) return `Must be at least ${field.min}`
  if (field.max !== undefined && n > field.max) return `Must be at most ${field.max}`
  if (field.exclusive_min !== undefined && n <= field.exclusive_min) return `Must be greater than ${field.exclusive_min}`
  if (field.exclusive_max !== undefined && n >= field.exclusive_max) return `Must be less than ${field.exclusive_max}`
  return null
}

function parseNumber(field, raw) {
  const text = String(raw).trim()
  if (text === '') return { empty: true }
  const n = Number(text)
  if (!Number.isFinite(n)) return { error: 'Enter a number' }
  if (field.widget === 'integer' && !Number.isInteger(n)) return { error: 'Enter a whole number' }
  const rangeError = checkRange(field, n)
  return rangeError ? { error: rangeError } : { value: n }
}

function parseIntList(raw) {
  const parts = String(raw).split(/[\s,;]+/).filter(Boolean)
  const out = []
  for (const p of parts) {
    const n = Number(p)
    if (!Number.isInteger(n) || n <= 0) return { error: `"${p}" is not a positive whole number` }
    out.push(n)
  }
  return out.length ? { value: out } : { empty: true }
}

/**
 * Turn form values into tool arguments.
 * Hidden fields and blank optional fields are left out so the tool's own
 * defaults apply. Returns `{ args, errors }`, errors keyed by field name.
 */
export function buildArguments(tool, values) {
  const args = {}
  const errors = {}
  for (const field of tool.fields) {
    if (!isVisible(field, values)) continue
    const raw = values[field.name]
    let parsed
    switch (field.widget) {
      case 'checkbox':
        parsed = { value: Boolean(raw) }
        break
      case 'integer':
      case 'number':
        parsed = parseNumber(field, raw)
        break
      case 'int_list':
        parsed = parseIntList(raw)
        break
      case 'columns':
      case 'text_list': {
        const target = field.name === 'feature_columns' ? values.target_column : null
        const list = (raw || []).filter((c) => c && c !== target)
        if (field.min_items && list.length && list.length < field.min_items) {
          parsed = { error: `Pick at least ${field.min_items}` }
        } else {
          parsed = list.length ? { value: list } : { empty: true }
        }
        break
      }
      case 'mapping': {
        const rows = (raw || []).filter((r) => r.from && r.to.trim())
        parsed = rows.length ? { value: Object.fromEntries(rows.map((r) => [r.from, r.to.trim()])) } : { empty: true }
        break
      }
      case 'value_mapping': {
        const rows = (raw || []).filter((r) => r.from.trim() && r.to.trim())
        parsed = rows.length
          ? { value: Object.fromEntries(rows.map((r) => [r.from.trim(), r.to.trim()])) }
          : { empty: true }
        break
      }
      default: {
        const text = String(raw ?? '').trim()
        parsed = text ? { value: text } : { empty: true }
      }
    }
    if (parsed.error) {
      errors[field.name] = parsed.error
    } else if (parsed.empty) {
      if (field.required) errors[field.name] = field.widget === 'columns' ? 'Pick at least one column' : 'Required'
    } else {
      args[field.name] = parsed.value
    }
  }
  return { args, errors }
}

function editValue(field, value) {
  switch (field.widget) {
    case 'checkbox':
      return Boolean(value)
    case 'int_list':
      return Array.isArray(value) ? value.join(', ') : String(value)
    case 'columns':
    case 'text_list':
      return (Array.isArray(value) ? value : [value]).map(String)
    case 'mapping':
    case 'value_mapping':
      return value && typeof value === 'object'
        ? Object.entries(value).map(([from, to]) => ({ from, to: String(to) }))
        : [{ from: '', to: '' }]
    case 'select': {
      const text = String(value)
      return field.options.find((o) => String(o) === text)
        ?? field.options.find((o) => String(o).toLowerCase() === text.toLowerCase())
        ?? text
    }
    default:
      return String(value)
  }
}

/**
 * Form edits that reproduce a tool call (e.g. one the assistant made).
 * Arguments the call left out stay at their defaults, which is what the tool used.
 */
export function editsFromArguments(tool, args) {
  const edits = {}
  for (const field of tool.fields) {
    const value = args?.[field.name]
    if (value !== undefined && value !== null) {
      edits[field.name] = editValue(field, value)
    } else if (field.app_default) {
      // The workbench default would not be what the call used.
      edits[field.name] = field.widget === 'columns' ? [] : ''
    }
  }
  return edits
}

export { columnChoices }
