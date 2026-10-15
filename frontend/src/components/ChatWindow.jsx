import { useCallback, useEffect, useRef, useState } from 'react'
import { API_BASE_URL, streamChat, verifyApi } from '../api.js'
import { chatTitle, loadChats, newId, relativeTime, saveChats } from '../chatStore.js'
import { NewChatIcon, SearchIcon, SendIcon, SidebarIcon, StopIcon, TrashIcon } from './icons.jsx'
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

const SIDEBAR_KEY = 'legalgpt.sidebar'

function readSidebar() {
  try {
    return localStorage.getItem(SIDEBAR_KEY) !== '0' // open unless the user closed it
  } catch {
    return true
  }
}

/** Renders inline [n] citation markers as buttons, plus **bold** / *italic* and "* " / "- " bullets, line by line. */
function AnswerText({ text, sources, onCite }) {
  const byMarker = new Map((sources || []).map((s) => [s.marker, s]))
  return text.split('\n').map((line, li) => {
    if (line.trim() === '---') return <hr key={li} className="my-2 border-slate-200" />
    const bullet = /^\s*[*-]\s+/.exec(line)
    if (bullet) line = line.slice(bullet[0].length)
    const parts = line.split(/(\[\d+\]|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g)
    return (
      <p key={li} className={`min-h-[1em] break-words ${bullet ? 'relative my-1 pl-5 before:absolute before:left-1.5 before:top-[0.7em] before:size-1.5 before:rounded-full before:bg-slate-400' : ''}`}>
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
    <div role="radiogroup" aria-label="Answer mode" className="inline-flex rounded-lg bg-slate-100 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={mode === o.value}
          title={o.hint}
          disabled={disabled}
          onClick={() => setMode(o.value)}
          className={`rounded-md px-2.5 py-1 text-xs font-medium disabled:opacity-50 ${
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
    <div className={`w-full min-w-0 rounded-2xl px-4 py-3.5 text-[0.9375rem] leading-relaxed text-slate-800 shadow-[0_1px_2px_rgb(15_23_42/0.04)] ring-1 sm:px-5 sm:py-4 ${tone}`} data-testid="assistant-message">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <span aria-hidden="true" className="grid size-6 place-items-center rounded-md bg-accent font-serif text-sm font-bold text-on-accent">§</span>
        <span className="font-medium text-slate-700">{msg.mode === 'general' ? 'General answer' : 'Legal answer'}</span>
        {msg.model && <span className="max-w-full truncate rounded-md bg-slate-100 px-1.5 py-0.5 font-mono text-[11px]" data-testid="answer-model">{msg.model}</span>}
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

/** "Today", "Yesterday", "Previous 7 days", "Older": groups for the chat list, newest first. */
function dayGroup(ts, now = new Date()) {
  const startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  if (ts >= startOfDay) return 'Today'
  if (ts >= startOfDay - 864e5) return 'Yesterday'
  if (ts >= startOfDay - 7 * 864e5) return 'Previous 7 days'
  return 'Older'
}

/** Saved conversations: new chat, search, switch, delete one, delete all. */
function ChatHistory({ chats, activeId, streamingChatId, onSelect, onNew, onDelete, onClearAll, onClose, closeLabel, className = '' }) {
  const [query, setQuery] = useState('')
  const q = query.trim().toLowerCase()
  const shown = q
    ? chats.filter((c) => c.title.toLowerCase().includes(q) || c.messages.some((m) => m.content?.toLowerCase().includes(q)))
    : chats
  const groups = []
  for (const c of shown) {
    const g = dayGroup(c.updatedAt)
    if (groups.at(-1)?.label !== g) groups.push({ label: g, chats: [] })
    groups.at(-1).chats.push(c)
  }
  return (
    <aside className={`flex h-full min-h-0 flex-col bg-white ${className}`} aria-label="Chat history" data-testid="chat-history">
      <div className="flex items-center gap-1 px-3 pb-2 pt-3">
        <button
          type="button"
          onClick={onClose}
          className="grid size-9 place-items-center rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900"
          aria-label={closeLabel}
          title={`${closeLabel} (Ctrl+B)`}
        >
          <SidebarIcon />
        </button>
        <span className="flex-1 pl-1 text-sm font-semibold text-slate-900">Chats</span>
        <button
          type="button"
          onClick={onNew}
          className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-on-accent shadow-sm hover:bg-accent-hover"
          data-testid="new-chat"
        >
          <NewChatIcon className="size-4" />
          New chat
        </button>
      </div>
      {chats.length > 0 && (
        <label className="relative mx-3 mb-2 block">
          <span className="sr-only">Search chats</span>
          <SearchIcon className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-slate-400" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search chats"
            className="w-full rounded-lg border border-slate-200 bg-slate-50 py-1.5 pl-8 pr-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-400 focus:bg-white focus:outline-none"
            data-testid="chat-search"
          />
        </label>
      )}
      <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-2" aria-label="Saved chats">
        {chats.length === 0 && (
          <p className="px-3 py-6 text-sm leading-relaxed text-slate-500">
            No chats yet. Your questions and answers are saved here, in this browser only.
          </p>
        )}
        {chats.length > 0 && shown.length === 0 && <p className="px-3 py-4 text-sm text-slate-500">No chat matches "{query}".</p>}
        {groups.map((g) => (
          <section key={g.label} className="mt-2 first:mt-0">
            <h3 className="px-2.5 pb-1 pt-2 text-xs font-medium text-slate-400">{g.label}</h3>
            <ul className="space-y-0.5">
              {g.chats.map((c) => (
                <li key={c.id} className="group relative">
                  <button
                    type="button"
                    onClick={() => onSelect(c.id)}
                    aria-current={c.id === activeId ? 'true' : undefined}
                    className={`w-full rounded-lg py-2 pl-2.5 pr-9 text-left text-sm ${
                      c.id === activeId ? 'bg-indigo-50 text-indigo-900' : 'text-slate-700 hover:bg-slate-100'
                    }`}
                    title={c.title}
                    data-testid="chat-history-item"
                  >
                    <span className="block truncate">{c.title}</span>
                    <span className="mt-0.5 flex items-center gap-1.5 text-xs text-slate-400">
                      {c.id === streamingChatId && <span aria-hidden="true" className="size-1.5 animate-pulse rounded-full bg-indigo-500" />}
                      {c.id === streamingChatId ? 'answering…' : relativeTime(c.updatedAt)}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onDelete(c)}
                    className="absolute right-1.5 top-1/2 grid size-7 -translate-y-1/2 place-items-center rounded-md text-slate-400 hover:bg-red-50 hover:text-red-600 md:opacity-0 md:focus:opacity-100 md:group-hover:opacity-100"
                    aria-label={`Delete chat: ${c.title}`}
                    title="Delete this chat"
                    data-testid="delete-chat"
                  >
                    <TrashIcon />
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </nav>
      <div className="flex items-center justify-between gap-2 border-t border-slate-200 px-3 py-2.5 text-xs text-slate-400">
        <span>Saved in this browser only</span>
        {chats.length > 0 && (
          <button type="button" onClick={onClearAll} className="rounded px-1 hover:text-red-600" data-testid="clear-history">
            Delete all
          </button>
        )}
      </div>
    </aside>
  )
}

export default function ChatWindow() {
  const [chats, setChats] = useState(loadChats)
  const [activeId, setActiveId] = useState(null) // null = a new chat, saved once its first question is sent
  const [streamingChatId, setStreamingChatId] = useState(null)
  const [historyOpen, setHistoryOpen] = useState(false) // drawer on small screens
  const [sidebarOpen, setSidebarOpen] = useState(readSidebar) // side panel on wide screens
  const [input, setInput] = useState('')
  const [mode, setMode] = useState('legal')
  const [models, setModels] = useState([])
  const [model, setModel] = useState(null)
  const [sections, setSections] = useState([])
  const [verifySection, setVerifySection] = useState('')
  const [openSource, setOpenSource] = useState(null)
  const abortRef = useRef(null)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  const streaming = streamingChatId !== null
  const messages = chats.find((c) => c.id === activeId)?.messages || []

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  // the question box grows with its text (up to max-h-52, then scrolls)
  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [input])

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

  useEffect(() => {
    try {
      localStorage.setItem(SIDEBAR_KEY, sidebarOpen ? '1' : '0')
    } catch {
      // private window: not remembered
    }
  }, [sidebarOpen])

  const toggleHistory = useCallback(() => {
    if (window.matchMedia('(min-width: 768px)').matches) setSidebarOpen((o) => !o)
    else setHistoryOpen((o) => !o)
  }, [])

  // Ctrl+B (Cmd+B) opens and closes the chat list; Escape closes the drawer on small screens
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'b') {
        e.preventDefault()
        toggleHistory()
      } else if (e.key === 'Escape') setHistoryOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [toggleHistory])

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

  const historyPanel = (className, onClose, closeLabel) => (
    <ChatHistory
      onClose={onClose}
      closeLabel={closeLabel}
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

  const activeTitle = chats.find((c) => c.id === activeId)?.title

  return (
    <div className="flex min-h-0 flex-1">
      {/* wide screens: a side panel that slides open and closed, and pushes the chat aside */}
      <div
        className={`hidden shrink-0 overflow-hidden border-r border-slate-200 transition-[width,border-color] duration-300 ease-[cubic-bezier(0.2,0,0,1)] md:block ${
          sidebarOpen ? 'w-72' : 'w-0 border-transparent'
        }`}
        inert={!sidebarOpen}
        data-testid="sidebar"
        data-open={sidebarOpen}
      >
        <div className={`h-full w-72 transition-transform duration-300 ease-[cubic-bezier(0.2,0,0,1)] ${sidebarOpen ? '' : '-translate-x-6'}`}>
          {historyPanel('', () => setSidebarOpen(false), 'Close chat list')}
        </div>
      </div>
      {/* small screens: a drawer that slides in over the chat */}
      <div className={`fixed inset-0 z-40 md:hidden ${historyOpen ? '' : 'pointer-events-none'}`} inert={!historyOpen}>
        <button
          type="button"
          tabIndex={-1}
          className={`absolute inset-0 bg-scrim/50 transition-opacity duration-300 ${historyOpen ? 'opacity-100' : 'opacity-0'}`}
          aria-label="Close chat list"
          onClick={() => setHistoryOpen(false)}
        />
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Chat history"
          className={`absolute inset-y-0 left-0 w-80 max-w-[85%] shadow-2xl transition-transform duration-300 ease-[cubic-bezier(0.2,0,0,1)] ${
            historyOpen ? 'translate-x-0' : '-translate-x-full'
          }`}
        >
          {historyPanel('', () => setHistoryOpen(false), 'Close chat list')}
        </div>
      </div>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="flex h-12 shrink-0 items-center gap-1 border-b border-slate-200/70 px-2 sm:px-3">
          <button
            type="button"
            onClick={toggleHistory}
            className={`grid size-9 place-items-center rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900 ${sidebarOpen ? 'md:hidden' : ''}`}
            aria-label="Open chat list"
            aria-expanded={sidebarOpen}
            title="Open chat list (Ctrl+B)"
            data-testid="open-history"
          >
            <SidebarIcon />
          </button>
          {!sidebarOpen && (
            <button
              type="button"
              onClick={newChat}
              className="hidden size-9 place-items-center rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900 md:grid"
              aria-label="New chat"
              title="New chat"
            >
              <NewChatIcon />
            </button>
          )}
          <h2 className="min-w-0 flex-1 truncate px-1 text-sm font-medium text-slate-700" data-testid="chat-title">
            {activeTitle || 'New chat'}
          </h2>
          {messages.length > 0 && (
            <button
              type="button"
              onClick={newChat}
              className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-900 md:hidden"
              title="Start a new conversation (this one stays in the history)"
            >
              <NewChatIcon className="size-4" />
              New
            </button>
          )}
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-6 sm:px-4 sm:py-8" data-testid="message-list">
          <div className="mx-auto flex max-w-3xl flex-col gap-5">
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
                  <div className="max-w-[85%] animate-rise whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-[0.9375rem] leading-relaxed text-on-accent shadow-sm" data-testid="user-message">
                    {m.content}
                  </div>
                </div>
              ) : (
                <div key={m.id} className="flex animate-rise justify-start">
                  <AssistantMessage msg={m} onCite={setOpenSource} />
                </div>
              ),
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="px-3 pb-3 pt-2 sm:px-4 sm:pb-4">
          <div className="mx-auto max-w-3xl">
            <form
              className="rounded-2xl bg-white shadow-[0_1px_2px_rgb(15_23_42/0.04),0_8px_24px_-12px_rgb(15_23_42/0.18)] ring-1 ring-slate-200 transition-shadow focus-within:ring-2 focus-within:ring-indigo-300"
              onSubmit={(e) => {
                e.preventDefault()
                send(input)
              }}
            >
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                rows={1}
                maxLength={4000}
                placeholder={mode === 'legal' ? 'Describe what happened, or ask about an IPC section…' : 'Ask anything (no sources)…'}
                aria-label="Your question"
                className="block max-h-52 min-h-[3.25rem] w-full resize-none rounded-t-2xl border-0 bg-transparent px-4 pb-1 pt-3.5 text-[0.9375rem] leading-relaxed text-slate-900 placeholder:text-slate-400 focus:outline-none focus-visible:outline-none"
              />
              <div className="flex flex-wrap items-center gap-2 px-2.5 pb-2.5 pt-1">
                <ModeToggle mode={mode} setMode={setMode} disabled={streaming} />
              {models.length > 1 && (
                <div role="radiogroup" aria-label="Answer model" className="inline-flex max-w-full overflow-x-auto rounded-lg bg-slate-100 p-0.5" data-testid="model-toggle">
                  {models.map((m) => (
                    <button
                      key={m.id}
                      role="radio"
                      aria-checked={model === m.id}
                      disabled={streaming || !m.available}
                      title={`${m.model_id || m.id}${m.note ? ` — ${m.note}` : ''}`}
                      onClick={() => setModel(m.id)}
                      className={`shrink-0 whitespace-nowrap rounded-md px-2.5 py-1 text-xs font-medium disabled:opacity-40 ${
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
                <div className="ml-auto">
                  {streaming ? (
                    <button
                      type="button"
                      onClick={() => abortRef.current?.abort()}
                      className="inline-flex h-9 items-center gap-1.5 rounded-xl bg-slate-200 px-3 text-sm font-medium text-slate-800 hover:bg-slate-300"
                      aria-label="Stop the answer"
                    >
                      <StopIcon />
                      Stop
                    </button>
                  ) : (
                    <button
                      type="submit"
                      disabled={!input.trim()}
                      className="grid size-9 place-items-center rounded-xl bg-accent text-on-accent shadow-sm hover:bg-accent-hover disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
                      aria-label="Send"
                      title="Send (Enter)"
                    >
                      <SendIcon />
                    </button>
                  )}
                </div>
              </div>
            </form>
            <p className="mt-2 px-2 text-center text-[11px] leading-snug text-slate-400">
              Legal information, not legal advice. Check answers against the cited sections.
              <span className="hidden lg:inline"> · Enter to send, Shift+Enter for a new line, Ctrl+B for the chat list</span>
            </p>
          </div>
        </div>

        <SourceModal source={openSource} onClose={() => setOpenSource(null)} />
      </div>
    </div>
  )
}
