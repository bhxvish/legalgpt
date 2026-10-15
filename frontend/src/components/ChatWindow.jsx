import { useCallback, useEffect, useRef, useState } from 'react'
import { API_BASE_URL, streamChat, verifyApi } from '../api.js'
import { chatTitle, loadChats, newId, relativeTime, saveChats } from '../chatStore.js'
import SourceModal from './SourceModal.jsx'
import { VerificationResultView } from './VerifyPanel.jsx'
import WhyPanel from './WhyPanel.jsx'
import { AnswerProof } from './ZkProof.jsx'

const HISTORY_MESSAGES = 6

const PROVIDER_NAMES = { gemini: 'Gemini', openai: 'OpenAI', openrouter: 'OpenRouter' }

/** Short name for the model switch: "gemini:gemini-2.5-flash" -> "Gemini 2.5 flash". */
function modelLabel(id) {
  if (id === 'adapter') return 'Local (LoRA)'
  if (id === 'local-heavy') return 'Local heavy (7B)'
  if (id === 'groq') return 'Base (Groq)'
  const [provider, model = ''] = id.split(/:(.*)/s)
  const name = model.split('/').pop().replace(/^gemini-/, '').replace(/-/g, ' ')
  return `${PROVIDER_NAMES[provider] || provider} ${name}`.trim()
}

const EXAMPLES = [
  'What is the punishment for causing death by negligence?',
  'What is the difference between culpable homicide and murder?',
  'Explain Section 378 IPC with an example.',
  'My neighbour broke into my shed at night and took my bicycle. What offence is that?',
]

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
      {msg.explanation && <WhyPanel explanation={msg.explanation} sources={msg.sources} retrieval={msg.retrieval} onOpenSource={onCite} />}
      <AnswerProof msg={msg} />
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

function TrashIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M8.75 1A2.75 2.75 0 006 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 10.23 1.482l.149-.022.841 10.518A2.75 2.75 0 007.596 19h4.807a2.75 2.75 0 002.742-2.53l.841-10.52.149.023a.75.75 0 00.23-1.482A41.03 41.03 0 0014 4.193V3.75A2.75 2.75 0 0011.25 1h-2.5zM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4zM8.58 7.72a.75.75 0 00-1.5.06l.3 7.5a.75.75 0 101.5-.06l-.3-7.5zm4.34.06a.75.75 0 10-1.5-.06l-.3 7.5a.75.75 0 101.5.06l.3-7.5z"
        clipRule="evenodd"
      />
    </svg>
  )
}

/** Saved conversations: new chat, switch, delete one, delete all. */
function ChatHistory({ chats, activeId, streamingChatId, onSelect, onNew, onDelete, onClearAll, className = '' }) {
  return (
    <aside className={`flex min-h-0 flex-col border-r border-slate-200 bg-white ${className}`} aria-label="Chat history" data-testid="chat-history">
      <div className="p-3">
        <button
          type="button"
          onClick={onNew}
          className="w-full rounded-lg bg-accent px-3 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover"
          data-testid="new-chat"
        >
          + New chat
        </button>
      </div>
      <p className="px-4 pb-1 text-xs font-medium uppercase tracking-wide text-slate-400">History</p>
      <ul className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {chats.length === 0 && <li className="px-2 py-3 text-xs text-slate-400">Your conversations will appear here.</li>}
        {chats.map((c) => (
          <li key={c.id} className="group relative">
            <button
              type="button"
              onClick={() => onSelect(c.id)}
              aria-current={c.id === activeId ? 'true' : undefined}
              className={`w-full rounded-md py-2 pl-2 pr-8 text-left text-sm ${
                c.id === activeId ? 'bg-indigo-50 text-indigo-900' : 'text-slate-700 hover:bg-slate-100'
              }`}
              title={c.title}
              data-testid="chat-history-item"
            >
              <span className="block truncate">{c.title}</span>
              <span className="block text-xs text-slate-400">{c.id === streamingChatId ? 'answering…' : relativeTime(c.updatedAt)}</span>
            </button>
            <button
              type="button"
              onClick={() => onDelete(c)}
              className="absolute right-1 top-1/2 -translate-y-1/2 rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-600 md:opacity-0 md:focus:opacity-100 md:group-hover:opacity-100"
              aria-label={`Delete chat: ${c.title}`}
              title="Delete this chat"
              data-testid="delete-chat"
            >
              <TrashIcon />
            </button>
          </li>
        ))}
      </ul>
      {chats.length > 0 && (
        <div className="border-t border-slate-100 p-2 text-center">
          <button type="button" onClick={onClearAll} className="text-xs text-slate-400 hover:text-red-600" data-testid="clear-history">
            Delete all chats
          </button>
        </div>
      )}
      <p className="border-t border-slate-100 px-3 py-2 text-[11px] leading-snug text-slate-400">Saved in this browser only.</p>
    </aside>
  )
}

export default function ChatWindow() {
  const [chats, setChats] = useState(loadChats)
  const [activeId, setActiveId] = useState(null) // null = a new chat, saved once its first question is sent
  const [streamingChatId, setStreamingChatId] = useState(null)
  const [historyOpen, setHistoryOpen] = useState(false) // drawer on small screens
  const [input, setInput] = useState('')
  const [mode, setMode] = useState('legal')
  const [models, setModels] = useState([])
  const [model, setModel] = useState(null)
  const [sections, setSections] = useState([])
  const [verifySection, setVerifySection] = useState('')
  const [openSource, setOpenSource] = useState(null)
  const abortRef = useRef(null)
  const bottomRef = useRef(null)

  const streaming = streamingChatId !== null
  const messages = chats.find((c) => c.id === activeId)?.messages || []

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  // Save after each finished answer, not on every streamed token.
  useEffect(() => {
    if (!streaming) saveChats(chats)
  }, [chats, streaming])

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

  /** Update one message of one chat. The chat is found by id, so an answer keeps streaming into its own
   * chat even if the user switches to another one meanwhile. */
  const update = (chatId, msgId, fn) =>
    setChats((cs) => cs.map((c) => (c.id === chatId ? { ...c, messages: c.messages.map((m) => (m.id === msgId ? fn(m) : m)) } : c)))

  const send = useCallback(
    async (question) => {
      question = question.trim()
      if (!question || streaming) return
      const history = messages
        .filter((m) => m.content && !m.error)
        .slice(-HISTORY_MESSAGES)
        .map((m) => ({ role: m.role, content: m.content }))
      const botId = newId()
      const verify = mode === 'legal' && verifySection ? verifySection : null
      const added = [
        { id: newId(), role: 'user', content: question, mode },
        { id: botId, role: 'assistant', content: '', mode, question, sources: [], verifySection: verify },
      ]
      const chatId = activeId || newId()
      const now = Date.now()
      setChats((cs) => {
        const existing = cs.find((c) => c.id === chatId)
        const msgs = [...(existing?.messages || []), ...added]
        const chat = { id: chatId, createdAt: existing?.createdAt || now, updatedAt: now, title: chatTitle(msgs), messages: msgs }
        return [chat, ...cs.filter((c) => c.id !== chatId)] // most recent first
      })
      setActiveId(chatId)
      setInput('')
      setStreamingChatId(chatId)
      const upd = (fn) => update(chatId, botId, fn)
      const controller = new AbortController()
      abortRef.current = controller
      try {
        await streamChat(
          { question, mode, history, model, verify_section: verify },
          (event, data) => {
            if (event === 'meta') upd((m) => ({ ...m, refused: data.refused, model: data.model }))
            else if (event === 'sources') upd((m) => ({ ...m, sources: data }))
            else if (event === 'token') upd((m) => ({ ...m, content: m.content + data.text }))
            else if (event === 'explanation') upd((m) => ({ ...m, explanation: data }))
            else if (event === 'verification') upd((m) => ({ ...m, verification: data }))
            else if (event === 'attestation') upd((m) => ({ ...m, attestation: data }))
            else if (event === 'retrieval') upd((m) => ({ ...m, retrieval: data }))
            else if (event === 'error') upd((m) => ({ ...m, error: data.message }))
          },
          controller.signal,
        )
      } catch (err) {
        if (err.name === 'AbortError') upd((m) => (m.content ? m : { ...m, error: 'stopped before the answer started' }))
        else upd((m) => ({ ...m, error: err.message }))
      } finally {
        setStreamingChatId(null)
        abortRef.current = null
      }
    },
    [messages, mode, streaming, model, verifySection, activeId],
  )

  const newChat = () => {
    setActiveId(null)
    setInput('')
    setHistoryOpen(false)
  }

  const selectChat = (id) => {
    setActiveId(id)
    setHistoryOpen(false)
  }

  const deleteChat = (chat) => {
    if (!window.confirm(`Delete this chat?\n\n"${chat.title}"`)) return
    if (chat.id === streamingChatId) abortRef.current?.abort()
    setChats((cs) => cs.filter((c) => c.id !== chat.id))
    if (chat.id === activeId) setActiveId(null)
  }

  const clearAll = () => {
    if (!window.confirm(`Delete all ${chats.length} saved chats? This cannot be undone.`)) return
    abortRef.current?.abort()
    setChats([])
    setActiveId(null)
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send(input)
    }
  }

  const historyPanel = (className) => (
    <ChatHistory
      chats={chats}
      activeId={activeId}
      streamingChatId={streamingChatId}
      onSelect={selectChat}
      onNew={newChat}
      onDelete={deleteChat}
      onClearAll={clearAll}
      className={className}
    />
  )

  return (
    <div className="flex min-h-0 flex-1">
      {historyPanel('hidden w-64 shrink-0 md:flex')}
      {historyOpen && (
        <div className="fixed inset-0 z-40 flex md:hidden" role="dialog" aria-modal="true" aria-label="Chat history">
          {historyPanel('w-72 max-w-[85%] shadow-xl')}
          <button type="button" className="flex-1 bg-scrim/30" aria-label="Close chat history" onClick={() => setHistoryOpen(false)} />
        </div>
      )}

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6" data-testid="message-list">
          <div className="mx-auto flex max-w-3xl flex-col gap-4">
            {messages.length === 0 && (
              <div className="mt-6 sm:mt-14">
                <h2 className="text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">Ask about the Indian Penal Code</h2>
                <p className="mt-2 max-w-xl text-sm leading-relaxed text-slate-500">
                  Legal mode answers only from IPC text and cites each source, so you can open the section it relied on.
                  General mode skips retrieval. Describe what happened in plain words; you do not need section numbers.
                </p>
                <p className="mt-8 text-xs font-medium text-slate-500">Try one of these</p>
                <div className="mt-2 grid gap-2 sm:grid-cols-2">
                  {EXAMPLES.map((q) => (
                    <button
                      key={q}
                      onClick={() => send(q)}
                      disabled={streaming}
                      className="group flex items-start justify-between gap-3 rounded-xl bg-white px-4 py-3 text-left text-sm text-slate-700 ring-1 ring-slate-200 hover:ring-indigo-300 hover:bg-indigo-50/40 disabled:opacity-50"
                    >
                      <span>{q}</span>
                      <span aria-hidden="true" className="text-slate-400 group-hover:translate-x-0.5 group-hover:text-indigo-600 transition-transform">→</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m) =>
              m.role === 'user' ? (
                <div key={m.id} className="flex justify-end">
                  <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-sm bg-accent px-4 py-2.5 text-sm text-on-accent" data-testid="user-message">
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
              <button
                type="button"
                onClick={() => setHistoryOpen(true)}
                className="rounded-md px-2 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 hover:bg-slate-50 md:hidden"
                data-testid="open-history"
              >
                Chats{chats.length > 0 ? ` (${chats.length})` : ''}
              </button>
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
                      {modelLabel(m.id)}
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
                    className="rounded-md border border-slate-300 bg-white px-1 py-0.5 text-xs text-slate-800"
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
                  className="ml-auto rounded-md px-2 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 hover:bg-slate-50 md:hidden"
                  title="Start a new conversation (this one stays in the history)"
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
                className="min-h-[2.75rem] flex-1 resize-none rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
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
                  className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover disabled:opacity-40"
                >
                  Send
                </button>
              )}
            </form>
          </div>
        </div>

        <SourceModal source={openSource} onClose={() => setOpenSource(null)} />
      </div>
    </div>
  )
}
