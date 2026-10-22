/** 600000 → "600,000"; anything not a number → "—". */
export function formatCount(n) {
  return Number.isFinite(n) ? n.toLocaleString('en-US') : '—'
}

/** Seconds → "10 s", "5 min", "1.5 h", "2 d". */
export function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return '—'
  const units = [[86400, 'd'], [3600, 'h'], [60, 'min'], [1, 's']]
  for (const [size, unit] of units) {
    if (Math.abs(seconds) >= size || unit === 's') {
      const value = seconds / size
      return `${Number.isInteger(value) ? value : value.toFixed(1)} ${unit}`
    }
  }
  return `${seconds} s`
}

/** ISO timestamp → "2024-01-01 06:30:00" (no fraction, no zone); empty → "—". */
export function formatTime(iso) {
  return iso ? iso.replace('T', ' ').replace(/(:\d\d)(\.\d+)?(\+00:00|Z)?$/, '$1') : '—'
}

/**
 * Server timestamp → Date, or null. SQLite returns naive UTC ("2024-01-01T06:30:00"),
 * Postgres an offset ("…+00:00"); only the naive form needs a "Z".
 */
export function parseServerTime(ts) {
  if (!ts) return null
  const date = new Date(/(Z|[+-]\d\d:?\d\d)$/.test(ts) ? ts : `${ts}Z`)
  return Number.isNaN(date.getTime()) ? null : date
}

/** Bytes → "12.3 MB". */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let i = 0
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024
    i += 1
  }
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}
