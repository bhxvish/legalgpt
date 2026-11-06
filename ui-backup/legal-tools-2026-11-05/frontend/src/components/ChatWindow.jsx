import { useCallback, useEffect, useRef, useState } from 'react'
import { API_BASE_URL, streamChat, verifyApi } from '../api.js'
import { chatTitle, loadChats, newId, relativeTime, saveChats } from '../chatStore.js'
import { CloseIcon, NewChatIcon, SearchIcon, SendIcon, SidebarIcon, StopIcon, TrashIcon } from './icons.jsx'
import { LogoMark } from './LawArt.jsx'
import { SlidingPill, stagger } from './motion.jsx'
import { Lines } from './Chapter.jsx'
import { Scales, Stamp } from './LegalArt.jsx'
import Select from './Select.jsx'
import SourceModal from './SourceModal.jsx'
import { VerificationResultView } from './VerifyPanel.jsx'
import WhyPanel from './WhyPanel.jsx'
import { AnswerProof } from './ZkProof.jsx'
import AnswerTrace from './AnswerTrace.jsx'
import ExplainPanel from './ExplainPanel.jsx'

const HISTORY_MESSAGES = 6

const PROVIDER_NAMES = { gemini: 'Gemini', openai: 'OpenAI', openrouter: 'OpenRouter' }

/** Short name for the model switch: "gemini:gemini-2.5-flash" -> "Gemini 2.5 flash". */
function modelLabel(id) {
  if (id === 'adapter') return 'Local (LoRA)'
  if (id === 'local-heavy') return 'Local heavy (7B)'
  if (id === 'groq') return 'Base (Groq)'
  const [provider, model = ''] = id.split(/:(.*)/s)
  if (provider === 'openai') return model.replace(/^gpt/, 'GPT').replace(/-(mini|nano|pro)$/, ' $1') // "GPT-5.4 mini"
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
    return localStorage.getItem(SIDEBAR_KEY) === '1' // closed unless the user opened it
  } catch {
    return false
  }
}

/** Inline markdown: [n] citation markers as buttons, plus **bold** and *italic*. */
function Inline({ text, byMarker, onCite }) {
  return text.split(/(\[\d+\]|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g).map((part, pi) => {
    const cite = /^\[(\d+)\]$/.exec(part)
    if (cite) {
      const source = byMarker.get(Number(cite[1]))
      return source ? (
        <button
          key={pi}
          onClick={() => onCite(source)}
          className="relative -top-[0.42em] mx-px rounded-sm px-px font-sans text-[0.68em] font-semibold leading-none text-indigo-600 hover:bg-indigo-100 hover:text-indigo-800"
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
  })
}

const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line)
const isTableRule = (line) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line)
const cells = (line) => line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim())

/** The answer text, line by line: paragraphs, "# " headings, "* " / "- " bullets, "---" rules and
 * markdown tables (a header row, a |---| rule, then body rows; a table still streaming in shows
 * its rows as they arrive). */
export function AnswerText({ text, sources, onCite, caret = false }) {
  const byMarker = new Map((sources || []).map((s) => [s.marker, s]))
  const lines = text.split('\n')
  const blocks = []
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (isTableRow(line)) {
      const rows = []
      while (i < lines.length && (isTableRow(lines[i]) || isTableRule(lines[i]))) rows.push(lines[i++])
      i--
      const hasHeader = rows.length > 1 && isTableRule(rows[1])
      const body = rows.filter((r) => !isTableRule(r)).map(cells)
      const head = hasHeader ? body.shift() : null
      blocks.push(
        <div key={i} className="my-4 overflow-x-auto border-y border-rule" data-testid="answer-table">
          <table className="w-full border-collapse text-left font-sans text-sm">
            {head && (
              <thead className="text-slate-900">
                <tr>
                  {head.map((c, ci) => (
                    <th key={ci} scope="col" className="border-b border-slate-200 px-3 py-2 font-semibold">
                      <Inline text={c} byMarker={byMarker} onCite={onCite} />
                    </th>
                  ))}
                </tr>
              </thead>
            )}
            <tbody>
              {body.map((r, ri) => (
                <tr key={ri} className="align-top [&:not(:last-child)>td]:border-b [&>td]:border-slate-100">
                  {r.map((c, ci) => (
                    <td key={ci} className={`px-3 py-2 ${ci === 0 ? 'font-medium text-slate-800' : 'text-slate-700'}`}>
                      <Inline text={c} byMarker={byMarker} onCite={onCite} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      )
      continue
    }
    if (line.trim() === '---') {
      blocks.push(<hr key={i} className="my-3 border-slate-200" />)
      continue
    }
    const heading = /^\s*#{1,6}\s+(.*)$/.exec(line)
    if (heading) {
      blocks.push(
        <p key={i} className="mb-1 mt-6 font-sans text-[1.3rem] font-semibold leading-snug tracking-[-0.025em] text-slate-900 first:mt-0" role="heading" aria-level={3}>
          <Inline text={heading[1]} byMarker={byMarker} onCite={onCite} />
        </p>,
      )
      continue
    }
    const bullet = /^\s*[*-]\s+/.exec(line)
    const body = bullet ? line.slice(bullet[0].length) : line
    blocks.push(
      <p key={i} className={`min-h-[0.75em] break-words ${bullet ? 'relative my-1.5 pl-5 before:absolute before:left-0.5 before:top-[0.8em] before:h-px before:w-2.5 before:bg-indigo-500' : ''}`}>
        <Inline text={body} byMarker={byMarker} onCite={onCite} />
        {caret && i === lines.length - 1 && (
          <span aria-hidden="true" className="ml-0.5 inline-block h-[1.05em] w-[2px] translate-y-[0.18em] animate-caret rounded-full bg-indigo-500" />
        )}
      </p>,
    )
  }
  return blocks
}

function ModeToggle({ mode, setMode, disabled }) {
  const options = [
    { value: 'legal', label: 'Legal', hint: 'Answers only from the IPC, with citations' },
    { value: 'general', label: 'General', hint: 'Ungrounded general answer, no retrieval' },
  ]
  const ref = useRef(null)
  return (
    <div ref={ref} role="radiogroup" aria-label="Answer mode" className="relative inline-flex">
      <SlidingPill containerRef={ref} selector='[aria-checked="true"]' deps={[mode]} className="border-b-2 border-indigo-500" />
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={mode === o.value}
          title={o.hint}
          disabled={disabled}
          onClick={() => setMode(o.value)}
          className={`relative px-2 py-1 text-sm font-medium disabled:opacity-50 ${
            mode === o.value ? 'text-slate-900' : 'text-slate-500 hover:text-slate-900'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

/** Which code to answer under. Auto: the BNS when the question names it or dates the events on or
 * after 1 July 2024, otherwise the IPC (backend app/core/bns.py: choose_law). */
function LawToggle({ law, setLaw, disabled }) {
  const options = [
    { value: 'auto', label: 'Auto', hint: 'IPC, or the BNS when the events are on or after 1 July 2024 or you name the BNS' },
    { value: 'ipc', label: 'IPC', hint: 'Indian Penal Code, 1860: offences before 1 July 2024' },
    { value: 'bns', label: 'BNS', hint: 'Bharatiya Nyaya Sanhita, 2023: offences from 1 July 2024' },
  ]
  const ref = useRef(null)
  return (
    <div ref={ref} role="radiogroup" aria-label="Law" className="relative inline-flex" data-testid="law-toggle">
      <SlidingPill containerRef={ref} selector='[aria-checked="true"]' deps={[law]} className="border-b-2 border-indigo-500" />
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={law === o.value}
          title={o.hint}
          disabled={disabled}
          onClick={() => setLaw(o.value)}
          className={`relative px-2 py-1 text-sm font-medium disabled:opacity-50 ${
            law === o.value ? 'text-slate-900' : 'text-slate-500 hover:text-slate-900'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

const ScalesMini = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="size-3.5 shrink-0">
    <path d="M12 3.5v16M8 20.5h8M4.5 6.5h15M4.5 6.5 2 13M4.5 6.5 7 13M1.8 13a2.7 2.7 0 0 0 5.4 0ZM19.5 6.5 17 13M19.5 6.5 22 13M16.8 13a2.7 2.7 0 0 0 5.4 0Z" />
  </svg>
)

function AssistantMessage({ msg, onCite, live = false, onUpdate }) {
  const box = useRef(null)
  const cited = new Set([...(msg.content || '').matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])))
  const tone = msg.error
    ? 'rounded-2xl bg-red-50 px-4 py-3.5 ring-1 ring-red-200 sm:px-5 sm:py-4'
    : msg.refused
      ? 'rounded-2xl bg-amber-50 px-4 py-3.5 ring-1 ring-amber-200 sm:px-5 sm:py-4'
      : ''
  return (
    <div ref={box} className={`w-full min-w-0 text-[0.9375rem] leading-relaxed text-slate-800 ${tone}`} data-testid="assistant-message">
      <div className="mb-3 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[13px] text-slate-500">
        <LogoMark className="size-5" />
        <span className="font-medium text-slate-800">{msg.mode === 'general' ? 'General answer' : 'Legal answer'}</span>
        {msg.retrieval?.law && (
          <span className="font-medium text-indigo-600" title={`Answered under the ${msg.retrieval.law === 'bns' ? 'Bharatiya Nyaya Sanhita, 2023' : 'Indian Penal Code, 1860'}: ${msg.retrieval.law_reason}`}>
            {msg.retrieval.law.toUpperCase()}
          </span>
        )}
        {msg.model && <span className="max-w-full truncate font-mono text-[11px] text-slate-400" data-testid="answer-model">{msg.model}</span>}
        {msg.refused && <span className="font-medium text-amber-800">Out of scope</span>}
        {!live && !msg.error && !msg.refused && cited.size > 0 && <Stamp className="ml-auto text-indigo-600">Cited</Stamp>}
      </div>
      {msg.content ? (
        <div className="reading">
          <AnswerText text={msg.content} sources={msg.sources} onCite={onCite} caret={live} />
        </div>
      ) : (
        !msg.error && (
          <p className="text-shimmer text-lg font-medium">
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
      {(() => {
        // IPC sections cited in the answer, with their BNS equivalent (BPRD correspondence table)
        const cited = new Set([...(msg.content || '').matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])))
        const seen = new Set()
        const bnsRows = (msg.sources || []).filter((s) => s.doc_id === 'BNS' && s.ipc?.length && cited.has(s.marker) && !seen.has(s.section) && seen.add(s.section))
        if (bnsRows.length > 0 && !msg.refused)
          return (
            <p className="mt-4 flex flex-wrap items-center gap-x-2 gap-y-1 border-l-2 border-indigo-500 pl-3 text-[13px] text-slate-600" data-testid="ipc-line">
              <span className="font-medium text-slate-700">Answered under the BNS; before 1 July 2024:</span>
              {bnsRows.map((s) => (
                <span key={s.section} className="whitespace-nowrap">
                  s.{s.section} BNS ← {s.ipc[0] === 'new' ? <i>new provision</i> : <b className="font-semibold text-slate-800">s.{s.ipc.slice(0, 4).join(', ')} IPC</b>}
                </span>
              ))}
            </p>
          )
        const rows = (msg.sources || []).filter((s) => s.kind !== 'judgment' && s.section && s.bns?.length && cited.has(s.marker) && !seen.has(s.section) && seen.add(s.section))
        return rows.length > 0 && !msg.refused ? (
          <p className="mt-4 flex flex-wrap items-center gap-x-2 gap-y-1 border-l-2 border-indigo-500 pl-3 text-[13px] text-slate-600" data-testid="bns-line">
            <span className="font-medium text-slate-700">Under the BNS (since 1 July 2024):</span>
            {rows.map((s) => (
              <span key={s.section} className="whitespace-nowrap">
                s.{s.section} IPC → {s.bns[0] === 'none' ? <i>no direct equivalent</i> : <b className="font-semibold text-slate-800">s.{s.bns.join(', ')} BNS</b>}
              </span>
            ))}
          </p>
        ) : null
      })()}
      {msg.sources?.length > 0 && (
        <div className="mt-6 border-t border-rule pt-2.5">
          <p className="mb-1.5 flex items-baseline justify-between gap-3 text-[13px] font-medium text-slate-900">
            Sources
            {cited.size > 0 && <span className="font-normal text-slate-500">dark = cited in the answer</span>}
          </p>
          <ul className="grid gap-x-8 sm:grid-cols-2">
            {msg.sources.map((s, si) => (
              <li key={s.chunk_id} className="min-w-0 animate-rise border-b border-slate-200" style={stagger(si, 35)}>
                <button
                  onClick={() => onCite(s)}
                  title={`${cited.has(s.marker) ? 'Cited in the answer' : 'Retrieved but not cited'}: open the ${s.kind === 'judgment' ? 'judgment passage' : 'IPC text'}`}
                  className={`group flex w-full min-w-0 items-baseline gap-2 py-1.5 text-left text-[13px] leading-snug ${cited.has(s.marker) ? 'text-slate-900' : 'text-slate-500'}`}
                  data-testid="source-chip"
                  data-kind={s.kind || 'statute'}
                  data-cited={cited.has(s.marker)}
                >
                  <span className={`shrink-0 font-semibold ${cited.has(s.marker) ? 'text-indigo-600' : ''}`}>{s.marker}</span>
                  {s.kind === 'judgment' ? (
                    <span className="flex min-w-0 items-baseline gap-1.5 group-hover:underline">
                      <span className="translate-y-0.5 self-center"><ScalesMini /></span>
                      <span className="truncate">{s.title}</span>
                      <span className="shrink-0 opacity-70">({[s.court?.includes('Supreme') ? 'SC' : '', s.year].filter(Boolean).join(', ')})</span>
                    </span>
                  ) : (
                    <span className="min-w-0 group-hover:underline">
                      {s.doc_id === 'BNS' && <span className="mr-1 font-semibold">BNS</span>}
                      {s.section ? `§${s.section}` : s.citation_path}
                      {s.title ? ` ${s.title}` : ''}
                      {s.ipc?.length > 0 && (
                        <span className="ml-1.5 font-mono text-[10px] text-slate-500" title="The IPC section this BNS section replaced">
                          {s.ipc[0] === 'new' ? 'new in BNS' : `IPC ${s.ipc.slice(0, 4).join(', ')}`}
                        </span>
                      )}
                      {s.bns?.length > 0 && (
                        <span className="ml-1.5 font-mono text-[10px] text-slate-500" title="Bharatiya Nyaya Sanhita, 2023 equivalent">
                          {s.bns[0] === 'none' ? 'no BNS' : `BNS ${s.bns.join(', ')}`}
                        </span>
                      )}
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      <AnswerTrace msg={msg} containerRef={box} onCite={onCite} />
      {msg.explanation && <WhyPanel explanation={msg.explanation} sources={msg.sources} retrieval={msg.retrieval} onOpenSource={onCite} />}
      {!live && onUpdate && msg.mode === 'legal' && msg.content && !msg.error && msg.question && (
        <ExplainPanel msg={msg} onUpdate={onUpdate} onCite={onCite} />
      )}
      <AnswerProof msg={msg} />
      {msg.verifySection && !msg.verification && !msg.error && (
        <p className="text-shimmer mt-3 text-xs font-medium">Checking the facts against s.{msg.verifySection}…</p>
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
  const [confirmClear, setConfirmClear] = useState(false)
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
    <aside className={`flex h-full min-h-0 flex-col bg-slate-50 ${className}`} aria-label="Chat history" data-testid="chat-history">
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
        <span className="flex-1 pl-1 text-lg font-semibold tracking-[-0.02em] text-slate-900">Chats</span>
        <button
          type="button"
          onClick={onNew}
          className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[13px] font-medium text-slate-800 ring-1 ring-slate-300 hover:text-slate-950 hover:ring-slate-900"
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
            className="w-full rounded-lg border border-slate-200 bg-transparent py-1.5 pl-8 pr-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-rule focus:outline-none"
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
            <h3 className="px-2.5 pb-1 pt-2 text-[13px] font-medium text-slate-500">{g.label}</h3>
            <ul className="space-y-0.5">
              {g.chats.map((c) => (
                <li key={c.id} className="group relative">
                  <button
                    type="button"
                    onClick={() => onSelect(c.id)}
                    aria-current={c.id === activeId ? 'true' : undefined}
                    className={`w-full rounded-lg py-2 pl-2.5 pr-9 text-left text-sm ${
                      c.id === activeId ? 'bg-slate-100 text-slate-950 shadow-[inset_2px_0_0_var(--color-indigo-600)]' : 'text-slate-700 hover:bg-slate-100'
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
        {chats.length > 0 &&
          (confirmClear ? (
            <span className="flex items-center gap-1" onMouseLeave={() => setConfirmClear(false)}>
              <span className="text-slate-500">Delete all?</span>
              <button
                type="button"
                onClick={() => {
                  setConfirmClear(false)
                  onClearAll()
                }}
                className="rounded-md bg-red-50 px-1.5 py-0.5 font-medium text-red-700 ring-1 ring-red-200 hover:bg-red-100"
                data-testid="confirm-clear-history"
              >
                Yes
              </button>
              <button type="button" onClick={() => setConfirmClear(false)} className="rounded-md px-1.5 py-0.5 hover:bg-slate-100 hover:text-slate-700">
                No
              </button>
            </span>
          ) : (
            <button type="button" onClick={() => setConfirmClear(true)} className="rounded px-1 hover:text-red-600" data-testid="clear-history">
              Delete all
            </button>
          ))}
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
  const [law, setLaw] = useState('auto')
  const [models, setModels] = useState([])
  const [model, setModel] = useState(null)
  const [sections, setSections] = useState([])
  const [verifySection, setVerifySection] = useState('')
  const [openSource, setOpenSource] = useState(null)
  const [undo, setUndo] = useState(null) // { label, removed: [chats], activeId } after a delete
  const undoTimer = useRef(null)
  const abortRef = useRef(null)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  const streaming = streamingChatId !== null
  const messages = chats.find((c) => c.id === activeId)?.messages || []

  useEffect(() => {
    if (messages.length) bottomRef.current?.scrollIntoView({ block: 'end' }) // an empty chat stays at its title
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
        { id: botId, role: 'assistant', content: '', mode, question, sources: [], verifySection: verify, modelKey: model, law },
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
          { question, mode, history, model, verify_section: verify, law },
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
    [messages, mode, streaming, model, verifySection, activeId, law],
  )

  // the question box on the Overview hands its question to the chat
  const sendRef = useRef(send)
  sendRef.current = send
  useEffect(() => {
    const onAsk = (e) => sendRef.current(e.detail)
    window.addEventListener('legalgpt:ask', onAsk)
    return () => window.removeEventListener('legalgpt:ask', onAsk)
  }, [])

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

  // Deleting happens at once, with an Undo notice for a few seconds (no browser confirm box).
  const offerUndo = (label, removed) => {
    clearTimeout(undoTimer.current)
    setUndo({ label, removed, activeId })
    undoTimer.current = setTimeout(() => setUndo(null), 6000)
  }

  const undoDelete = () => {
    if (!undo) return
    clearTimeout(undoTimer.current)
    setChats((cs) =>
      [...cs, ...undo.removed.filter((r) => !cs.some((c) => c.id === r.id))].sort((a, b) => b.updatedAt - a.updatedAt),
    )
    if (undo.removed.some((c) => c.id === undo.activeId)) setActiveId(undo.activeId)
    setUndo(null)
  }

  useEffect(() => () => clearTimeout(undoTimer.current), [])

  const deleteChat = (chat) => {
    if (chat.id === streamingChatId) abortRef.current?.abort()
    setChats((cs) => cs.filter((c) => c.id !== chat.id))
    if (chat.id === activeId) setActiveId(null)
    offerUndo('Chat deleted', [chat])
  }

  const clearAll = () => {
    abortRef.current?.abort()
    offerUndo(chats.length === 1 ? 'Chat deleted' : `${chats.length} chats deleted`, chats)
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

      <div className={`relative isolate flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden transition-colors duration-500 ${messages.length ? '' : 'bg-field-gold text-ink'}`}>
        <div className="flex h-12 shrink-0 items-center gap-1 px-2 sm:px-3">
          <button
            type="button"
            onClick={toggleHistory}
            className={`press grid size-9 place-items-center rounded-lg ${messages.length ? 'text-slate-500 hover:bg-slate-100 hover:text-slate-900' : 'text-ink hover:bg-ink/10'} ${sidebarOpen ? 'md:hidden' : ''}`}
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
              className={`press hidden size-9 place-items-center rounded-lg md:grid ${messages.length ? 'text-slate-500 hover:bg-slate-100 hover:text-slate-900' : 'text-ink hover:bg-ink/10'}`}
              aria-label="New chat"
              title="New chat"
            >
              <NewChatIcon />
            </button>
          )}
          <h2 className={`min-w-0 flex-1 truncate px-1 text-[15px] font-medium text-slate-500 ${messages.length ? '' : 'invisible'}`} data-testid="chat-title">
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
        <div className={`min-h-0 flex-1 overflow-y-auto ${messages.length ? 'fade-edges px-3 py-6 sm:px-4 sm:py-8' : 'px-4 pb-4 sm:px-8'}`} data-testid="message-list">
          <div className={`mx-auto flex flex-col ${messages.length ? 'max-w-[44rem]' : 'h-full max-w-none'}`}>
            {messages.length === 0 && (
              <div className="flex min-h-full flex-col justify-between gap-8">
                <div className="relative">
                  <Scales tilt={-4} className="pointer-events-none absolute -top-2 right-0 hidden h-[17rem] text-[#e0a22b] lg:block" />
                  <Lines lines={['Ask the', 'Penal Code']} className="chapter-title relative pt-2" />
                </div>
                <div>
                  <p className="standfirst animate-rise font-medium" style={{ animationDelay: '300ms' }}>
                    Describe what happened in plain words. Legal mode answers only from IPC text and cites each source. General mode skips retrieval.
                  </p>
                  <ul className="mt-5 grid gap-x-8 lg:grid-cols-2" aria-label="Example questions">
                    {EXAMPLES.map((q, qi) => (
                      <li key={q} className="animate-rise" style={stagger(qi, 70, 360)}>
                        <button
                          onClick={() => send(q)}
                          disabled={streaming}
                          className="press group flex w-full items-baseline justify-between gap-4 py-2 text-left text-[clamp(1.0625rem,1rem+0.3vw,1.25rem)] font-medium leading-snug disabled:opacity-50"
                        >
                          <span className="underline decoration-transparent decoration-2 underline-offset-4 transition-[text-decoration-color] duration-200 group-hover:decoration-current">{q}</span>
                          <span aria-hidden="true" className="transition-transform duration-200 group-hover:translate-x-1">→</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
            {messages.map((m) =>
              m.role === 'user' ? (
                <div key={m.id} className="mt-10 animate-rise border-t border-slate-300 pt-8 first:mt-2 first:border-0 first:pt-0">
                  <div
                    className={`whitespace-pre-wrap break-words font-semibold text-slate-900 ${m.content.length > 200 ? 'text-[1.125rem] leading-relaxed' : 'text-[clamp(1.75rem,1.3rem+1.6vw,2.5rem)] leading-[1.05] tracking-[-0.035em]'}`}
                    data-testid="user-message"
                  >
                    {m.content}
                  </div>
                </div>
              ) : (
                <div key={m.id} className="mt-5 flex animate-rise justify-start">
                  <AssistantMessage msg={m} onCite={setOpenSource} live={streamingChatId === activeId && m === messages.at(-1)} onUpdate={(fn) => update(activeId, m.id, fn)} />
                </div>
              ),
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className={messages.length ? 'px-3 pb-3 pt-2 sm:px-4 sm:pb-4' : 'px-4 pb-4 pt-2 sm:px-8 sm:pb-6'}>
          <div className={`mx-auto ${messages.length ? 'max-w-[44rem]' : 'max-w-none'}`}>
            <form
              className={`bg-white text-slate-900 transition-shadow ${messages.length ? 'ring-1 ring-slate-200 focus-within:ring-2 focus-within:ring-slate-400' : ''}`}
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
                className="block max-h-52 min-h-[4.5rem] w-full sm:min-h-[3.25rem] resize-none border-0 bg-transparent px-4 pb-1 pt-3.5 text-[1.0625rem] leading-relaxed text-slate-900 placeholder:text-slate-400 focus:outline-none focus-visible:outline-none"
              />
              <div className="flex flex-wrap items-center gap-2 px-2.5 pb-2.5 pt-1">
                <ModeToggle mode={mode} setMode={setMode} disabled={streaming} />
                {mode === 'legal' && <LawToggle law={law} setLaw={setLaw} disabled={streaming} />}
              {models.length > 1 && (
                <Select
                  up
                  label="Answer model"
                  value={model}
                  onChange={setModel}
                  disabled={streaming}
                  menuClass="w-72"
                  testId="model-toggle"
                  options={models.map((m) => ({ value: m.id, label: modelLabel(m.id), hint: m.available ? m.model_id : m.note || 'Not available', disabled: !m.available }))}
                />
              )}
              {mode === 'legal' && sections.length > 0 && (
                <Select
                  up
                  label="Also verify facts against section"
                  value={verifySection}
                  onChange={setVerifySection}
                  disabled={streaming}
                  menuClass="w-72"
                  testId="verify-select"
                  options={[{ value: '', label: 'Verify facts: off', hint: 'No rule check after the answer' }, ...sections.map((s) => ({ value: s.section, label: `Verify against s.${s.section}`, hint: s.title }))]}
                />
              )}
                <div className="ml-auto">
                  {streaming ? (
                    <button
                      type="button"
                      onClick={() => abortRef.current?.abort()}
                      className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-slate-200 px-3 text-sm font-medium text-slate-800 hover:bg-slate-300"
                      aria-label="Stop the answer"
                    >
                      <StopIcon />
                      Stop
                    </button>
                  ) : (
                    <button
                      type="submit"
                      disabled={!input.trim()}
                      className="press grid size-10 place-items-center bg-accent text-on-accent hover:bg-accent-hover disabled:bg-slate-200 disabled:text-slate-400"
                      aria-label="Send"
                      title="Send (Enter)"
                    >
                      <SendIcon />
                    </button>
                  )}
                </div>
              </div>
            </form>
            <p className={`mt-2 text-[13px] leading-snug ${messages.length ? 'px-2 text-center text-slate-500' : 'text-ink'}`}>
              Legal information, not legal advice. Check answers against the cited sections.
              <span className="hidden lg:inline"> · Enter to send, Shift+Enter for a new line, Ctrl+B for the chat list</span>
            </p>
          </div>
        </div>

        <SourceModal source={openSource} onClose={() => setOpenSource(null)} />
        {undo && (
          <div
            key={undo.label + undo.removed.length}
            role="status"
            className="fixed bottom-4 left-1/2 z-50 flex -translate-x-1/2 animate-rise items-center md:left-4 md:translate-x-0 gap-1 rounded-xl bg-slate-900 py-1.5 pl-4 pr-1.5 text-sm text-slate-50 shadow-2xl ring-1 ring-slate-900/10"
            data-testid="undo-toast"
          >
            <span className="pr-2">{undo.label}</span>
            <button
              type="button"
              onClick={undoDelete}
              className="rounded-lg px-2.5 py-1 font-semibold text-indigo-300 hover:bg-white/10"
              data-testid="undo-delete"
            >
              Undo
            </button>
            <button
              type="button"
              onClick={() => setUndo(null)}
              className="grid size-7 place-items-center rounded-lg text-slate-400 hover:bg-white/10 hover:text-slate-50"
              aria-label="Dismiss"
            >
              <CloseIcon />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
