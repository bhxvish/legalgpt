import { useCallback, useEffect, useRef, useState } from 'react'
import { streamChat } from '../api.js'
import SourceModal from './SourceModal.jsx'

const HISTORY_MESSAGES = 6

const EXAMPLES = [
  'What is the punishment for causing death by negligence?',
  'What is the difference between culpable homicide and murder?',
  'Explain Section 378 IPC with an example.',
]

let nextId = 1

/** Renders inline [n] citation markers as buttons, plus **bold** / *italic*, line by line. */
function AnswerText({ text, sources, onCite }) {
  const byMarker = new Map((sources || []).map((s) => [s.marker, s]))
  return text.split('\n').map((line, li) => {
    if (line.trim() === '---') return <hr key={li} className="my-2 border-slate-200" />
    const parts = line.split(/(\[\d+\]|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g)
    return (
      <p key={li} className="min-h-[1em] break-words">
        {parts.map((part, pi) => {
          const cite = /^\[(\d+)\]$/.exec(part)
          if (cite) {
            const source = byMarker.get(Number(cite[1]))
            return source ? (
              <button
                key={pi}
                onClick={() => onCite(source)}
                className="mx-0.5 rounded bg-indigo-50 px-1 text-xs font-semibold text-indigo-700 ring-1 ring-indigo-200 hover:bg-indigo-100"
                title={source.citation_path}
                aria-label={`Source ${source.marker}: ${source.citation_path}`}
                data-testid="citation-marker"
              >
                {part}
              </button>
            ) : (
              // A marker the model produced that matches no retrieved source.
              <span key={pi} className="mx-0.5 rounded bg-red-50 px-1 text-xs text-red-700 ring-1 ring-red-200" title="No retrieved source has this number">
                {part}
              </span>
            )
          }
          if (part.startsWith('**') && part.endsWith('**') && part.length > 4) return <strong key={pi}>{part.slice(2, -2)}</strong>
          if (part.startsWith('*') && part.endsWith('*') && part.length > 2) return <em key={pi}>{part.slice(1, -1)}</em>
          return part
        })}
      </p>
    )
  })
}

function ModeToggle({ mode, setMode, disabled }) {
  const options = [
    { value: 'legal', label: 'Legal', hint: 'Answers only from the IPC, with citations' },
    { value: 'general', label: 'General', hint: 'Ungrounded general answer, no retrieval' },
  ]
  return (
    <div role="radiogroup" aria-label="Answer mode" className="inline-flex rounded-lg bg-slate-100 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={mode === o.value}
          title={o.hint}
          disabled={disabled}
          onClick={() => setMode(o.value)}
          className={`rounded-md px-3 py-1 text-sm font-medium transition disabled:opacity-50 ${
            mode === o.value ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

function AssistantMessage({ msg, onCite }) {
  const tone = msg.error
    ? 'bg-red-50 ring-red-200'
    : msg.refused
      ? 'bg-amber-50 ring-amber-200'
      : 'bg-white ring-slate-200'
  return (
    <div className={`max-w-[85%] rounded-2xl rounded-bl-sm px-4 py-3 text-sm leading-relaxed text-slate-800 shadow-sm ring-1 ${tone}`} data-testid="assistant-message">
      <div className="mb-1 flex items-center gap-2 text-xs text-slate-500">
        <span className="font-medium uppercase tracking-wide">{msg.mode === 'general' ? 'General' : 'Legal'}</span>
        {msg.refused && <span className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-800">Out of scope</span>}
      </div>
      {msg.content ? (
        <AnswerText text={msg.content} sources={msg.sources} onCite={onCite} />
      ) : (
        !msg.error && <p className="animate-pulse text-slate-400">{msg.mode === 'legal' ? 'Searching the IPC…' : 'Thinking…'}</p>
      )}
      {msg.error && <p className="mt-1 text-red-700">Error: {msg.error}</p>}
      {msg.sources?.length > 0 && (
        <div className="mt-3 border-t border-slate-100 pt-2">
          <p className="mb-1 text-xs font-medium text-slate-500">Sources</p>
          <ul className="flex flex-wrap gap-1.5">
            {msg.sources.map((s) => (
              <li key={s.chunk_id}>
                <button
                  onClick={() => onCite(s)}
                  className="rounded-md bg-slate-50 px-2 py-1 text-left text-xs text-slate-700 ring-1 ring-slate-200 hover:bg-slate-100"
                  data-testid="source-chip"
                >
                  <span className="font-semibold">[{s.marker}]</span>{' '}
                  {s.section ? `§${s.section}` : s.citation_path}
                  {s.title ? ` ${s.title}` : ''}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default function ChatWindow() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [mode, setMode] = useState('legal')
  const [streaming, setStreaming] = useState(false)
  const [openSource, setOpenSource] = useState(null)
  const abortRef = useRef(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  const update = (id, fn) => setMessages((ms) => ms.map((m) => (m.id === id ? fn(m) : m)))

  const send = useCallback(
    async (question) => {
      question = question.trim()
      if (!question || streaming) return
      const history = messages
        .filter((m) => m.content && !m.error)
        .slice(-HISTORY_MESSAGES)
        .map((m) => ({ role: m.role, content: m.content }))
      const userMsg = { id: nextId++, role: 'user', content: question, mode }
      const botId = nextId++
      setMessages((ms) => [...ms, userMsg, { id: botId, role: 'assistant', content: '', mode, sources: [] }])
      setInput('')
      setStreaming(true)
      const controller = new AbortController()
      abortRef.current = controller
      try {
        await streamChat(
          { question, mode, history },
          (event, data) => {
            if (event === 'meta') update(botId, (m) => ({ ...m, refused: data.refused, model: data.model }))
            else if (event === 'sources') update(botId, (m) => ({ ...m, sources: data }))
            else if (event === 'token') update(botId, (m) => ({ ...m, content: m.content + data.text }))
            else if (event === 'error') update(botId, (m) => ({ ...m, error: data.message }))
          },
          controller.signal,
        )
      } catch (err) {
        if (err.name !== 'AbortError') update(botId, (m) => ({ ...m, error: err.message }))
      } finally {
        setStreaming(false)
        abortRef.current = null
      }
    },
    [messages, mode, streaming],
  )

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send(input)
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6" data-testid="message-list">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.length === 0 && (
            <div className="mt-10 text-center">
              <h2 className="text-lg font-semibold text-slate-800">Ask about the Indian Penal Code</h2>
              <p className="mt-1 text-sm text-slate-500">
                Legal mode answers only from IPC text and cites each source. General mode skips retrieval.
              </p>
              <div className="mt-5 flex flex-col items-center gap-2">
                {EXAMPLES.map((q) => (
                  <button
                    key={q}
                    onClick={() => send(q)}
                    className="rounded-lg bg-white px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200 hover:bg-slate-50"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m) =>
            m.role === 'user' ? (
              <div key={m.id} className="flex justify-end">
                <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2.5 text-sm text-white" data-testid="user-message">
                  {m.content}
                </div>
              </div>
            ) : (
              <div key={m.id} className="flex justify-start">
                <AssistantMessage msg={m} onCite={setOpenSource} />
              </div>
            ),
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="border-t border-slate-200 bg-white px-4 py-3">
        <div className="mx-auto max-w-3xl">
          <div className="mb-2 flex items-center justify-between gap-2">
            <ModeToggle mode={mode} setMode={setMode} disabled={streaming} />
            <span className="hidden text-xs text-slate-400 sm:inline">Enter to send · Shift+Enter for a new line</span>
          </div>
          <form
            className="flex items-end gap-2"
            onSubmit={(e) => {
              e.preventDefault()
              send(input)
            }}
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={2}
              maxLength={4000}
              placeholder={mode === 'legal' ? 'Ask a question about the IPC…' : 'Ask anything (no sources)…'}
              aria-label="Your question"
              className="min-h-[2.75rem] flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
            />
            {streaming ? (
              <button
                type="button"
                onClick={() => abortRef.current?.abort()}
                className="rounded-lg bg-slate-200 px-4 py-2 text-sm font-medium text-slate-800 hover:bg-slate-300"
              >
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-40"
              >
                Send
              </button>
            )}
          </form>
        </div>
      </div>

      <SourceModal source={openSource} onClose={() => setOpenSource(null)} />
    </div>
  )
}
