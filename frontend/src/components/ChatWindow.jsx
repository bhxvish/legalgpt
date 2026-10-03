import { useCallback, useEffect, useRef, useState } from 'react'
import { API_BASE_URL, streamChat, verifyApi } from '../api.js'
import SourceModal from './SourceModal.jsx'
import { VerificationResultView } from './VerifyPanel.jsx'
import WhyPanel from './WhyPanel.jsx'

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
          className={`rounded-md px-2.5 py-1 text-xs font-medium transition disabled:opacity-50 sm:px-3 sm:text-sm ${
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
  const cited = new Set([...(msg.content || '').matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])))
  const tone = msg.error
    ? 'bg-red-50 ring-red-200'
    : msg.refused
      ? 'bg-amber-50 ring-amber-200'
      : 'bg-white ring-slate-200'
  return (
    <div className={`max-w-[85%] rounded-2xl rounded-bl-sm px-4 py-3 text-sm leading-relaxed text-slate-800 shadow-sm ring-1 ${tone}`} data-testid="assistant-message">
      <div className="mb-1 flex items-center gap-2 text-xs text-slate-500">
        <span className="font-medium uppercase tracking-wide">{msg.mode === 'general' ? 'General' : 'Legal'}</span>
        {msg.model && <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono" data-testid="answer-model">{msg.model}</span>}
        {msg.refused && <span className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-800">Out of scope</span>}
      </div>
      {msg.content ? (
        <AnswerText text={msg.content} sources={msg.sources} onCite={onCite} />
      ) : (
        !msg.error && (
          <p className="animate-pulse text-slate-400">
            {msg.mode === 'legal' && !msg.sources?.length && !msg.model ? 'Searching the IPC…' : 'Writing the answer…'}
            {msg.model?.includes('+lora') && (
              <span className="mt-1 block text-xs">
                The tuned model runs on this server. Without a GPU, the first answer also loads it and can take a few
                minutes; later answers about a minute.
              </span>
            )}
          </p>
        )
      )}
      {msg.error && <p className="mt-1 text-red-700">Error: {msg.error}</p>}
      {msg.sources?.length > 0 && (
        <div className="mt-3 border-t border-slate-100 pt-2">
          <p className="mb-1 text-xs font-medium text-slate-500">
            Sources
            {cited.size > 0 && <span className="font-normal text-slate-400"> · highlighted = cited in the answer</span>}
          </p>
          <ul className="flex flex-wrap gap-1.5">
            {msg.sources.map((s) => (
              <li key={s.chunk_id}>
                <button
                  onClick={() => onCite(s)}
                  title={cited.has(s.marker) ? 'Cited in the answer: open the IPC text' : 'Retrieved but not cited: open the IPC text'}
                  className={`rounded-md px-2 py-1 text-left text-xs ring-1 hover:brightness-95 ${
                    cited.has(s.marker)
                      ? 'bg-indigo-50 font-medium text-indigo-900 ring-indigo-200'
                      : 'bg-white text-slate-500 ring-slate-200'
                  }`}
                  data-testid="source-chip"
                  data-cited={cited.has(s.marker)}
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
      {msg.explanation && <WhyPanel explanation={msg.explanation} sources={msg.sources} onOpenSource={onCite} />}
      {msg.verifySection && !msg.verification && !msg.error && (
        <p className="mt-3 animate-pulse text-xs text-slate-500">Checking the facts against s.{msg.verifySection}…</p>
      )}
      {msg.verification &&
        (msg.verification.error ? (
          <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800 ring-1 ring-amber-200">
            Verification against s.{msg.verifySection} failed: {msg.verification.error}
          </p>
        ) : (
          <div className="mt-3">
            <p className="mb-1 text-xs font-medium text-slate-500">Rule check of the facts in your question</p>
            <VerificationResultView result={msg.verification} className="space-y-3 rounded-lg bg-slate-50 p-3 text-xs ring-1 ring-slate-200" />
          </div>
        ))}
    </div>
  )
}

export default function ChatWindow() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [mode, setMode] = useState('legal')
  const [models, setModels] = useState([])
  const [model, setModel] = useState(null)
  const [sections, setSections] = useState([])
  const [verifySection, setVerifySection] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [openSource, setOpenSource] = useState(null)
  const abortRef = useRef(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/models`)
      .then((r) => (r.ok ? r.json() : []))
      .then((list) => {
        setModels(list)
        setModel((m) => m || list.find((x) => x.default)?.id || null)
      })
      .catch(() => setModels([]))
    verifyApi
      .sections()
      .then((m) => setSections(m.sections))
      .catch(() => setSections([])) // verification not available on this server
  }, [])

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
      const verify = mode === 'legal' && verifySection ? verifySection : null
      setMessages((ms) => [...ms, userMsg, { id: botId, role: 'assistant', content: '', mode, sources: [], verifySection: verify }])
      setInput('')
      setStreaming(true)
      const controller = new AbortController()
      abortRef.current = controller
      try {
        await streamChat(
          { question, mode, history, model, verify_section: verify },
          (event, data) => {
            if (event === 'meta') update(botId, (m) => ({ ...m, refused: data.refused, model: data.model }))
            else if (event === 'sources') update(botId, (m) => ({ ...m, sources: data }))
            else if (event === 'token') update(botId, (m) => ({ ...m, content: m.content + data.text }))
            else if (event === 'explanation') update(botId, (m) => ({ ...m, explanation: data }))
            else if (event === 'verification') update(botId, (m) => ({ ...m, verification: data }))
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
    [messages, mode, streaming, model, verifySection],
  )

  const newChat = () => {
    abortRef.current?.abort()
    setMessages([])
    setInput('')
  }

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
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <ModeToggle mode={mode} setMode={setMode} disabled={streaming} />
            {models.length > 1 && (
              <div role="radiogroup" aria-label="Answer model" className="inline-flex rounded-lg bg-slate-100 p-1" data-testid="model-toggle">
                {models.map((m) => (
                  <button
                    key={m.id}
                    role="radio"
                    aria-checked={model === m.id}
                    disabled={streaming || !m.available}
                    title={`${m.model_id || m.id}${m.note ? ` — ${m.note}` : ''}`}
                    onClick={() => setModel(m.id)}
                    className={`rounded-md px-2.5 py-1 text-xs font-medium transition disabled:opacity-40 sm:px-3 sm:text-sm ${
                      model === m.id ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {m.id === 'adapter' ? 'Tuned (LoRA)' : m.id === 'groq' ? 'Base (Groq)' : m.id}
                  </button>
                ))}
              </div>
            )}
            {mode === 'legal' && sections.length > 0 && (
              <label className="flex items-center gap-1 text-xs text-slate-600" title="After the answer, check the facts in your question against an encoded IPC section">
                <span className="hidden sm:inline">Also verify facts against</span>
                <span className="sm:hidden">Verify facts</span>
                <select
                  value={verifySection}
                  onChange={(e) => setVerifySection(e.target.value)}
                  disabled={streaming}
                  aria-label="Also verify facts against section"
                  className="rounded-md border border-slate-300 px-1 py-0.5 text-xs"
                  data-testid="verify-select"
                >
                  <option value="">—</option>
                  {sections.map((s) => (
                    <option key={s.section} value={s.section}>
                      s.{s.section}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <span className="ml-auto hidden text-xs text-slate-400 lg:inline">Enter to send · Shift+Enter for a new line</span>
            {messages.length > 0 && (
              <button
                type="button"
                onClick={newChat}
                className="ml-auto rounded-md px-2 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 hover:bg-slate-50 lg:ml-0"
                data-testid="new-chat"
                title="Clear this conversation and start again"
              >
                New chat
              </button>
            )}
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
