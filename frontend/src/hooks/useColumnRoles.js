import { useCallback, useMemo } from 'react'

export const ROLES = [
  { id: 'time', label: 'Time', hint: 'The timestamp of each row: used for gaps, resampling and time-ordered splits' },
  { id: 'target', label: 'Target', hint: 'The lab value the soft sensor predicts (y)' },
  { id: 'feature', label: 'Feature', hint: 'A model input (x)' },
  { id: 'ignore', label: 'Ignore', hint: 'Kept in the data but not used by the model' },
]

/** Whether `role` can be given to a column of this kind (kind is unknown until the profile loads). */
export function roleAllowed(role, kind) {
  if (!kind) return true
  if (role === 'feature') return kind === 'numeric' || kind === 'boolean'
  if (role === 'target') return kind !== 'time'
  if (role === 'time') return kind === 'time' || kind === 'text'
  return true
}

/**
 * Column roles: one time column, one target, the features, and the rest ignored.
 * No features chosen means every numeric column except the target and the time column
 * (the same default the tool forms use); changing a role makes the list explicit.
 */
export function useColumnRoles({
  columns, timeColumn, targetVariable, selectedFeatures,
  onTimeColumnChange, onTargetSelect, onFeaturesChange,
}) {
  const explicit = (selectedFeatures || []).length > 0
  const features = useMemo(() => {
    if (explicit) return selectedFeatures.filter((f) => f !== targetVariable && f !== timeColumn)
    return (columns || [])
      .filter((c) => c.kind === 'numeric' && c.name !== targetVariable && c.name !== timeColumn)
      .map((c) => c.name)
  }, [explicit, selectedFeatures, columns, targetVariable, timeColumn])

  const roleOf = useCallback((name) => {
    if (name === timeColumn) return 'time'
    if (name === targetVariable) return 'target'
    return features.includes(name) ? 'feature' : 'ignore'
  }, [timeColumn, targetVariable, features])

  const setRole = useCallback((name, role) => {
    const current = roleOf(name)
    if (role === current || current === 'time') return
    if (current === 'target') onTargetSelect(null)
    const rest = features.filter((f) => f !== name)
    if (role === 'time') onTimeColumnChange(name)
    if (role === 'target') onTargetSelect(name)
    if (role === 'feature') onFeaturesChange([...rest, name])
    else if (role === 'ignore' || (explicit && current === 'feature')) onFeaturesChange(rest)
  }, [roleOf, features, explicit, onTargetSelect, onTimeColumnChange, onFeaturesChange])

  return { features, explicit, roleOf, setRole }
}
