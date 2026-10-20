import { useEffect, useState } from 'react'

function format(seconds) {
  if (seconds < 60) return `${seconds} s`
  return `${Math.floor(seconds / 60)} min ${String(seconds % 60).padStart(2, '0')} s`
}

/** Seconds since it was mounted, for long-running work. Shows nothing for the first second. */
export default function Elapsed() {
  const [start] = useState(() => Date.now())
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setSeconds(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => clearInterval(id)
  }, [start])
  return seconds ? <span className="elapsed">{format(seconds)}</span> : null
}
