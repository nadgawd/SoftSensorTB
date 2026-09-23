import { useEffect, useRef, useState } from 'react'
import { postChat } from '../api/client'

/**
 * Left-side chat drawer bound to POST /chat.
 */
export default function ChatSidebar({
  datasetVersionId,
  onChatResponse,
  disabled,
}) {
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      intent: null,
      content:
        'Upload a CSV/Parquet dataset, then ask me to explore, clean, plot, or train a soft sensor.',
    },
  ])
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  async function handleSubmit(event) {
    event.preventDefault()
    const prompt = input.trim()
    if (!prompt || busy) return

    setError(null)
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: prompt }])
    setBusy(true)

    try {
      const response = await postChat({
        message: prompt,
        datasetVersionId,
      })

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          intent: response.intent,
          content: response.message || 'Done.',
          uiUpdate: Boolean(response.ui_update_required),
        },
      ])

      // Push rich payload into App state when the backend signals a UI refresh.
      onChatResponse?.(response)
    } catch (err) {
      const detail =
        err?.response?.data?.detail ||
        err?.message ||
        'Chat request failed.'
      setError(typeof detail === 'string' ? detail : JSON.stringify(detail))
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          intent: 'ERROR',
          content: 'I could not complete that request. Check the error below and try again.',
        },
      ])
    } finally {
      setBusy(false)
    }
  }

  return (
    <aside className="panel-surface flex h-full w-full flex-col overflow-hidden rounded-2xl lg:w-[360px] xl:w-[400px]">
      <header className="border-b border-line/80 px-5 py-4">
        <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-signal">
          Agent console
        </p>
        <h2 className="mt-1 font-display text-xl font-semibold text-foam">Chat</h2>
        <p className="mt-1 text-xs text-mist">
          Routed to EXECUTE or RAG · synced to the live canvas
        </p>
      </header>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {messages.map((msg, index) => (
          <div
            key={`${msg.role}-${index}`}
            className={`animate-rise max-w-[95%] rounded-xl px-3.5 py-2.5 text-sm leading-relaxed ${
              msg.role === 'user'
                ? 'ml-auto bg-signal/15 text-foam ring-1 ring-signal/30'
                : 'mr-auto bg-ink/50 text-foam/95 ring-1 ring-line/70'
            }`}
          >
            {msg.intent ? (
              <span
                className={`mb-1 inline-block font-mono text-[10px] uppercase tracking-wider ${
                  msg.intent === 'EXECUTE'
                    ? 'text-ember'
                    : msg.intent === 'ERROR'
                      ? 'text-alert'
                      : 'text-signal'
                }`}
              >
                {msg.intent}
                {msg.uiUpdate ? ' · UI sync' : ''}
              </span>
            ) : null}
            <p className="whitespace-pre-wrap">{msg.content}</p>
          </div>
        ))}
        {busy ? (
          <div className="animate-fade-in mr-auto rounded-xl bg-ink/40 px-3.5 py-2 font-mono text-xs text-mist ring-1 ring-line/60">
            Thinking / calling tools…
          </div>
        ) : null}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="border-t border-line/80 p-4">
        {error ? (
          <p className="mb-2 rounded-lg bg-alert/10 px-3 py-2 text-xs text-alert ring-1 ring-alert/30">
            {error}
          </p>
        ) : null}
        {!datasetVersionId ? (
          <p className="mb-2 text-xs text-mist">
            Upload a dataset first for EXECUTE tools. Conceptual questions still work.
          </p>
        ) : null}
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={2}
            placeholder="e.g. Plot a parity chart for the PLS model…"
            disabled={busy || disabled}
            className="min-h-[64px] flex-1 resize-none rounded-xl border border-line bg-ink/60 px-3 py-2 text-sm text-foam outline-none placeholder:text-mist/60 focus:border-signal/60"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSubmit(e)
              }
            }}
          />
          <button
            type="submit"
            disabled={busy || !input.trim()}
            className="self-end rounded-xl bg-signal px-4 py-2 text-sm font-semibold text-ink transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Send
          </button>
        </div>
      </form>
    </aside>
  )
}
