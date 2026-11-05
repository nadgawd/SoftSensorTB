export const SPLIT_LABELS = {
  random: 'Random split',
  stratified: 'Stratified split',
  sequential: 'Chronological split',
  group: 'Group split',
  custom: 'Custom split',
}

/** The sets of a split, in order, with their plot colours (CSS variables). */
export const SPLIT_SETS = [
  { id: 'train', code: 1, label: 'Train', color: '--color-signal' },
  { id: 'validation', code: 2, label: 'Validation', color: '--color-ember' },
  { id: 'test', code: 3, label: 'Test', color: '--color-alert' },
]

export const CV_LABELS = {
  kfold: 'K-Fold',
  stratified_kfold: 'Stratified K-Fold',
  leave_one_out: 'Leave-one-out',
  time_series: 'Time-series CV',
  group_kfold: 'Group K-Fold',
  leave_one_group_out: 'Leave-one-group-out',
  repeated_kfold: 'Repeated K-Fold',
  repeated_stratified_kfold: 'Repeated Stratified K-Fold',
  shuffle_split: 'Shuffle split',
  stratified_shuffle_split: 'Stratified shuffle split',
  walk_forward: 'Walk-forward',
  bootstrap: 'Bootstrap (out-of-bag)',
}

export function splitLabel(strategy) {
  return SPLIT_LABELS[strategy] ?? strategy
}

/** "train 800 · validation 100 · test 200 rows" (validation only when there is one). */
export function splitCountsText(counts) {
  if (!counts) return ''
  const parts = SPLIT_SETS
    .filter((s) => s.id !== 'validation' || counts.validation)
    .map((s) => `${s.label.toLowerCase()} ${counts[s.id] ?? 0}`)
  return `${parts.join(' · ')} rows`
}

/** What one CV round is called: a fold, a random split, or a bootstrap resample. */
export function cvUnit(strategy) {
  if (strategy === 'bootstrap') return 'resamples'
  if (strategy === 'shuffle_split' || strategy === 'stratified_shuffle_split') return 'splits'
  return 'folds'
}
