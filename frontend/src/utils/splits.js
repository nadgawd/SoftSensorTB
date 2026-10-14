export const SPLIT_LABELS = {
  random: 'Random split',
  stratified: 'Stratified split',
  sequential: 'Chronological split',
}

export const CV_LABELS = {
  kfold: 'K-Fold',
  stratified_kfold: 'Stratified K-Fold',
  leave_one_out: 'Leave-one-out',
  time_series: 'Time-series CV',
}

export function splitLabel(strategy) {
  return SPLIT_LABELS[strategy] ?? strategy
}
