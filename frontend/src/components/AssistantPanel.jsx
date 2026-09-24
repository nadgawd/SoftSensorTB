import { fetchLlmStatus, postChat } from '../api/client'
import ChatMarkdown from './ChatMarkdown'
import { useEffect, useRef, useState } from 'react'
import { usePersisted } from '../hooks/usePersisted'
import { ActivitySummary, LegacyToolChips, LiveActivity } from './ChatActivity'

const SESSION_MAX_AGE = 24 * 60 * 60 * 1000
const HISTORY_TURNS = 12
const EMPTY_LIVE = { intent: null, status: 'Sending', steps: [], draft: '', reasoningActive: false }

/** Earlier turns sent with each request so the assistant can resolve follow-ups. */
function toHistory(messages) {
  return messages
    .filter((m) => m.role === 'user' || m.intent === 'EXECUTE' || m.intent === 'RAG')
    .filter((m) => m.content && !m.content.startsWith('*Generation stopped'))
    .slice(-HISTORY_TURNS)
    .map((m) => ({
      role: m.role,
      content: m.content,
      tools: m.toolCalls || [],
      intent: m.role === 'assistant' ? m.intent : null,
    }))
}

// The routing label (EXECUTE / RAG) stays internal; only errors and canvas updates are shown.
function IntentBadge({ intent, uiUpdate }) {
  const isError = intent === 'ERROR'
  if (!isError && !uiUpdate) return null
  return (
    <div style={{ paddingLeft: '4px' }}>
      <span className={`badge ${isError ? 'badge-alert' : 'badge-ember'}`}>
        {isError ? 'ERROR' : 'canvas updated'}
      </span>
    </div>
  )
}


/**
 * Collapsible AI assistant panel that slides in from the right.
 */
export default function AssistantPanel({
  open,
  onClose,
  datasetVersionId,
  stepHint,
  uiContext,
  onChatResponse,
  pendingAction,
  onPendingActionConsumed,
  width,
  onWidthPreview,
  onWidthCommit,
  onWidthReset,
}) {
  const [resizing, setResizing] = useState(false)
  const resizingRef = useRef(false)
  const lastWidthRef = useRef(width)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [live, setLive] = useState(EMPTY_LIVE)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [error, setError] = useState(null)
  const [thinkingAvailable, setThinkingAvailable] = useState(false)
  const [think, setThink] = usePersisted('thinkMode', false)
  const [messages, setMessages] = usePersisted('chatMessages', [
    {
      role: 'assistant',
      intent: null,
      content:
        'Hello! I\'m your soft-sensor assistant. Upload a dataset to get started, or ask me a conceptual question about process analytics, PLS, linear regression (OLS), Ridge, Lasso, PCR, k-NN, or EDA.',
      toolCalls: [],
    },
  ], { maxAge: SESSION_MAX_AGE })
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)
  const abortControllerRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy, live.steps.length])

  useEffect(() => {
    if (!busy) return
    const started = Date.now()
    setElapsedMs(0)
    const timer = setInterval(() => setElapsedMs(Date.now() - started), 1000)
    return () => clearInterval(timer)
  }, [busy])

  // Keep the drag cursor and suppress text selection even when the pointer
  // leaves the handle mid-drag.
  useEffect(() => {
    if (!resizing) return
    const { cursor, userSelect } = document.body.style
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    return () => {
      document.body.style.cursor = cursor
      document.body.style.userSelect = userSelect
    }
  }, [resizing])

  function handleResizeStart(e) {
    if (e.button !== 0) return
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    lastWidthRef.current = width
    resizingRef.current = true
    setResizing(true)
  }

  function handleResizeMove(e) {
    if (!resizingRef.current) return
    lastWidthRef.current = window.innerWidth - e.clientX
    onWidthPreview?.(lastWidthRef.current)
  }

  function handleResizeEnd(e) {
    if (!resizingRef.current) return
    e.currentTarget.releasePointerCapture?.(e.pointerId)
    resizingRef.current = false
    setResizing(false)
    onWidthCommit?.(lastWidthRef.current)
  }

  function handleResizeKey(e) {
    const step = e.shiftKey ? 80 : 24
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      onWidthCommit?.(width + step)
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      onWidthCommit?.(width - step)
    }
  }

  // Focus textarea when panel opens
  useEffect(() => {
    if (open) {
      setTimeout(() => textareaRef.current?.focus(), 350)
    }
  }, [open])

  // Only offer the Think toggle when the model chat will actually use has one.
  useEffect(() => {
    if (!open) return
    let cancelled = false
    fetchLlmStatus()
      .then((s) => { if (!cancelled) setThinkingAvailable(Boolean(s?.thinking)) })
      .catch(() => { if (!cancelled) setThinkingAvailable(false) })
    return () => { cancelled = true }
  }, [open])

  // Auto-submit a pending action from clicking a chip
  useEffect(() => {
    if (!pendingAction || busy) return
    // slight delay so the panel animation completes first
    const timer = setTimeout(() => {
      handleSubmitText(pendingAction)
      onPendingActionConsumed?.()
    }, 400)
    return () => clearTimeout(timer)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingAction])

  // Core send function (used by both submit button and pendingAction)
  async function handleSubmitText(prompt) {
    if (!prompt?.trim() || busy) return

    setError(null)
    setInput('')
    const history = toHistory(messages)
    setMessages((prev) => [...prev, { role: 'user', content: prompt.trim(), toolCalls: [] }])
    setBusy(true)
    setLive(EMPTY_LIVE)

    abortControllerRef.current = new AbortController()
    const startedAt = Date.now()
    // Plain locals: SSE callbacks fire faster than React re-renders.
    let intent = null
    let status = 'Sending'
    let steps = []
    let draft = ''
    let reasoningActive = false
    const publish = () => setLive({ intent, status, steps, draft, reasoningActive })

    try {
      let finalResponse = {}

      await postChat({
        message: prompt.trim(),
        history,
        datasetVersionId,
        stepHint,
        uiContext,
        think: thinkingAvailable && think,
        signal: abortControllerRef.current.signal,
      }, (event, data) => {
        const continuesReasoning = event === 'reasoning' && reasoningActive
        reasoningActive = event === 'reasoning'
        if (event === 'intent') {
          intent = data
        } else if (event === 'status') {
          status = data
        } else if (event === 'reasoning') {
          steps = continuesReasoning
            ? [...steps.slice(0, -1), { ...steps[steps.length - 1], text: steps[steps.length - 1].text + data }]
            : [...steps, { type: 'reasoning', text: data.trimStart() }]
        } else if (event === 'token') {
          draft += data
        } else if (event === 'thought') {
          // Text streamed before a tool call is narration, not the answer.
          const text = (data || draft).trim()
          if (text) steps = [...steps, { type: 'thought', text }]
          draft = ''
        } else if (event === 'retract') {
          // The backend rejected the draft (it claimed work no tool did).
          draft = ''
        } else if (event === 'tool_start') {
          steps = [...steps, { type: 'tool', id: data.id, name: data.name, args: data.args }]
        } else if (event === 'tool_end') {
          steps = steps.map((s) =>
            s.type === 'tool' && s.id === data.id
              ? { ...s, ok: data.ok, ms: data.ms, summary: data.summary }
              : s,
          )
        } else if (event === 'final_state') {
          finalResponse = data
          return
        } else {
          return
        }
        publish()
      })

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          intent,
          content: draft.trim() || 'Done.',
          uiUpdate: Boolean(finalResponse.ui_update_required),
          toolCalls: finalResponse.tool_calls_made || steps.filter((s) => s.type === 'tool').map((s) => s.name),
          activity: steps,
          durationMs: Date.now() - startedAt,
        },
      ])

      if (finalResponse.ui_update_required) {
        onChatResponse?.(finalResponse)
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            intent: null,
            content: `${draft.trim() ? `${draft.trim()}\n\n` : ''}*Generation stopped by user.*`,
            toolCalls: [],
            activity: steps.map((s) => (s.type === 'tool' && s.ok === undefined ? { ...s, ok: false, summary: 'Stopped' } : s)),
            durationMs: Date.now() - startedAt,
          },
        ])
        return
      }

      const detail =
        err?.response?.data?.detail ||
        err?.message ||
        'Chat request failed.'
      const msg = typeof detail === 'string' ? detail : JSON.stringify(detail)
      setError(msg)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          intent: 'ERROR',
          content: `I couldn't complete that request.\n\n**Error:** ${msg}`,
          toolCalls: [],
        },
      ])
    } finally {
      setBusy(false)
      setLive(EMPTY_LIVE)
    }
  }

  function handleSubmit(e) {
    e?.preventDefault()
    handleSubmitText(input.trim())
    setInput('')
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmitText(input.trim())
      setInput('')
    }
  }

  return (
    <div
      className={`assistant-panel panel-elevated${open ? ' open' : ''}`}
      style={{ borderLeft: '1px solid var(--color-line-2)', width: `${width}px` }}
    >
      {/* Resize handle */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize assistant panel"
        aria-valuenow={Math.round(width)}
        title="Drag to resize · double-click to reset"
        tabIndex={open ? 0 : -1}
        className={`assistant-resize-handle${resizing ? ' dragging' : ''}`}
        onPointerDown={handleResizeStart}
        onPointerMove={handleResizeMove}
        onPointerUp={handleResizeEnd}
        onPointerCancel={handleResizeEnd}
        onDoubleClick={onWidthReset}
        onKeyDown={handleResizeKey}
      />

      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '16px 20px',
          borderBottom: '1px solid var(--color-line)',
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: 'var(--color-ember)', fontSize: '18px' }}>✦</span>
          <span style={{ fontWeight: 600, fontSize: '1rem', color: 'var(--color-foam)' }}>
            Assistant
          </span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          {/* Clear button */}
          <button
            type="button"
            title="Clear conversation"
            onClick={() =>
              setMessages([
                {
                  role: 'assistant',
                  intent: null,
                  content: 'Conversation cleared. How can I help?',
                  toolCalls: [],
                },
              ])
            }
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--color-mist)',
              cursor: 'pointer',
              padding: '4px 8px',
              borderRadius: '6px',
              fontSize: '16px',
              lineHeight: 1,
              transition: 'color 150ms ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--color-foam)')}
            onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--color-mist)')}
          >
            🗑
          </button>
          <button
            type="button"
            title="Close assistant"
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--color-mist)',
              cursor: 'pointer',
              padding: '4px 8px',
              borderRadius: '6px',
              fontSize: '18px',
              lineHeight: 1,
              transition: 'color 150ms ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--color-foam)')}
            onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--color-mist)')}
          >
            ×
          </button>
        </div>
      </div>

      {/* Step context badge */}
      {stepHint && (
        <div style={{ padding: '8px 20px', borderBottom: '1px solid var(--color-line)' }}>
          <span className="badge badge-mist">
            Step: {stepHint}
          </span>
          {!datasetVersionId && (
            <span className="badge badge-alert" style={{ marginLeft: '6px' }}>
              No dataset
            </span>
          )}
        </div>
      )}

      {/* Messages */}
      <div
        className="scroll-y"
        style={{ flex: 1, padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className="animate-slide-up"
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '4px',
            }}
          >
            {msg.role === 'assistant' && msg.intent && (
              <IntentBadge intent={msg.intent} uiUpdate={msg.uiUpdate} />
            )}
            {msg.activity ? (
              <ActivitySummary steps={msg.activity} durationMs={msg.durationMs} />
            ) : (
              <LegacyToolChips names={msg.toolCalls} />
            )}
            {/* Bubble */}
            <div className={msg.role === 'user' ? 'bubble-user' : 'bubble-assistant'}>
              {msg.role === 'assistant' ? (
                <div className="prose-chat">
                  <ChatMarkdown>{msg.content}</ChatMarkdown>
                </div>
              ) : (
                <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{msg.content}</p>
              )}
            </div>
          </div>
        ))}

        {busy && (
          <div className="animate-slide-up" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {live.intent && <IntentBadge intent={live.intent} />}
            <LiveActivity
              steps={live.steps}
              status={live.reasoningActive ? 'Thinking' : live.draft ? 'Writing' : live.status}
              elapsedMs={elapsedMs}
              reasoningActive={live.reasoningActive}
            />
            {live.draft && (
              <div className="bubble-assistant">
                <div className="prose-chat">
                  <ChatMarkdown>{live.draft + ' ▋'}</ChatMarkdown>
                </div>
              </div>
            )}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input form */}
      <div
        style={{
          padding: '14px 16px',
          borderTop: '1px solid var(--color-line)',
          flexShrink: 0,
        }}
      >
        {error && (
          <div
            className="badge badge-alert"
            style={{
              display: 'block',
              marginBottom: '8px',
              padding: '8px 12px',
              borderRadius: '8px',
              fontSize: '11px',
              lineHeight: 1.5,
              whiteSpace: 'pre-wrap',
            }}
          >
            {error}
          </div>
        )}
        {!datasetVersionId && (
          <p
            style={{
              margin: '0 0 8px',
              fontSize: '11px',
              color: 'var(--color-mist)',
            }}
          >
            Upload a dataset to enable EXECUTE tools. Conceptual questions work without a dataset.
          </p>
        )}
        {thinkingAvailable && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <button
              type="button"
              className="think-toggle"
              aria-pressed={think}
              onClick={() => setThink((t) => !t)}
              title="Let the model reason step by step before answering. Slower, better on multi-step requests."
            >
              ✦ Think {think ? 'on' : 'off'}
            </button>
            {think && (
              <span style={{ fontSize: '11px', color: 'var(--color-mist)' }}>
                Slower; reasoning appears in the activity log
              </span>
            )}
          </div>
        )}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
          <textarea
            ref={textareaRef}
            className="input-base"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
            placeholder="Ask a question…"
            disabled={busy}
            style={{
              resize: 'none',
              minHeight: '60px',
              flex: 1,
            }}
          />
          {busy ? (
            <button
              type="button"
              onClick={() => abortControllerRef.current?.abort()}
              title="Stop generating"
              style={{
                padding: '10px 14px',
                borderRadius: '10px',
                flexShrink: 0,
                fontSize: '16px',
                background: 'rgba(255,100,100,0.2)',
                color: '#ff6666',
                border: '1px solid rgba(255,100,100,0.3)',
                cursor: 'pointer',
              }}
            >
              ⏹
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!input.trim()}
              className="btn-signal"
              style={{
                padding: '10px 14px',
                borderRadius: '10px',
                flexShrink: 0,
                fontSize: '16px',
              }}
            >
              →
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
