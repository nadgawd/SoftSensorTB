import { useCallback, useEffect, useState } from 'react'
import { fetchLlmStatus } from '../api/client'

const POLL_MS = 30_000

function describe(status, error) {
  if (error) {
    return {
      tone: 'var(--color-alert)',
      label: 'Backend offline',
      detail: 'The API server is not responding, so chat is unavailable.',
    }
  }
  if (!status) {
    return { tone: 'var(--color-mist)', label: 'Checking LLM…', detail: 'Contacting the backend.' }
  }

  const { active, local, cloud_providers: cloud = [] } = status
  const cloudList = cloud.join(', ')

  if (active === 'local') {
    return {
      tone: 'var(--color-success)',
      label: `HPC · ${local.model}`,
      detail: `Chat runs on the HPC GPU via ${local.base_url}.` +
        (cloud.length ? ` Cloud fallback ready: ${cloudList}.` : ''),
    }
  }

  const localProblem = local
    ? local.online
      ? `The HPC server is up but does not serve "${local.model}" (serving: ${local.served.join(', ') || 'nothing'}).`
      : `The HPC server at ${local.base_url} is unreachable — is the tunnel up and the PBS job running?`
    : null

  if (active === 'cloud') {
    return {
      tone: local ? 'var(--color-warning)' : 'var(--color-signal)',
      label: local ? 'Cloud fallback' : 'Cloud',
      detail: [localProblem, `Chat is using: ${cloudList}.`].filter(Boolean).join(' '),
    }
  }

  return {
    tone: 'var(--color-alert)',
    label: 'No LLM available',
    detail: localProblem || 'No cloud API keys are set and no local model is configured.',
  }
}

/**
 * Sidebar chip showing which LLM backend chat will use. Click to re-check.
 */
export default function LlmStatus() {
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(false)
  const [checking, setChecking] = useState(false)

  const refresh = useCallback(async () => {
    setChecking(true)
    try {
      setStatus(await fetchLlmStatus())
      setError(false)
    } catch {
      setError(true)
    } finally {
      setChecking(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, POLL_MS)
    return () => clearInterval(timer)
  }, [refresh])

  const { tone, label, detail } = describe(status, error)

  return (
    <button
      type="button"
      onClick={refresh}
      disabled={checking}
      title={`${detail}\n\nClick to re-check.`}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '8px',
        width: '100%',
        padding: '8px 12px',
        borderRadius: '8px',
        background: `color-mix(in srgb, ${tone} 10%, transparent)`,
        border: `1px solid color-mix(in srgb, ${tone} 35%, transparent)`,
        fontSize: '11px',
        fontFamily: 'var(--font-mono)',
        color: tone,
        cursor: checking ? 'progress' : 'pointer',
        transition: 'all 150ms ease',
      }}
    >
      <span
        style={{
          width: '7px',
          height: '7px',
          borderRadius: '50%',
          background: tone,
          flexShrink: 0,
          opacity: checking ? 0.4 : 1,
          transition: 'opacity 150ms ease',
        }}
      />
      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
    </button>
  )
}
