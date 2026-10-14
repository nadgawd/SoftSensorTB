export const SPLIT_LABELS = {
  random: 'Random split',
  stratified: 'Stratified split',
  sequential: 'Chronological split',
  group: 'Group split',
}

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

/** What one CV round is called: a fold, a random split, or a bootstrap resample. */
export function cvUnit(strategy) {
  if (strategy === 'bootstrap') return 'resamples'
  if (strategy === 'shuffle_split' || strategy === 'stratified_shuffle_split') return 'splits'
  return 'folds'
}
