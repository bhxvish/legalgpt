import { useEffect, useRef, useState } from 'react'
import { API_BASE_URL } from './api.js'
import AnnotateView from './components/AnnotateView.jsx'
import ChatWindow from './components/ChatWindow.jsx'
import ReviewView from './components/ReviewView.jsx'
import VerifyPanel from './components/VerifyPanel.jsx'
import { SlidingPill } from './components/motion.jsx'
import ThemeMenu from './components/ThemeMenu.jsx'

const STYLES = {
  loading: { badge: 'text-slate-500', dot: 'bg-slate-400 animate-pulse' },
  ok: { badge: 'text-slate-600', dot: 'bg-emerald-500' },
  error: { badge: 'text-red-700', dot: 'bg-red-500' },
}

function HealthBadge() {
  const [health, setHealth] = useState({ state: 'loading', text: 'checking…' })

  // checked again every few seconds while unreachable (e.g. the page opened before the backend
  // finished starting), and every 30 s once up, so the badge also notices the backend stopping
  useEffect(() => {
    const controller = new AbortController()
    let timer
    const check = () =>
      fetch(`${API_BASE_URL}/health`, { signal: controller.signal })
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          return res.json()
        })
        .then((data) => {
          const ok = data.status === 'ok'
          setHealth({ state: ok ? 'ok' : 'error', text: `backend ${data.status}` })
          timer = setTimeout(check, ok ? 30000 : 4000)
        })
        .catch((err) => {
          if (err.name === 'AbortError') return
          setHealth({ state: 'error', text: 'backend unreachable' })
          timer = setTimeout(check, 4000)
        })
    check()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [])

  return (
    <span data-testid="health-status" title={`${API_BASE_URL}/health`} role="status"
      className={`inline-flex items-center gap-1.5 whitespace-nowrap text-xs font-medium ${STYLES[health.state].badge}`}>
      <span aria-hidden="true" className="relative inline-flex size-2">
        {health.state === 'ok' && <span className="absolute inset-0 animate-ping-soft rounded-full bg-emerald-500" />}
        <span className={`relative size-2 rounded-full ${STYLES[health.state].dot}`} />
      </span>
      <span className="hidden sm:inline">{health.text}</span>
      <span className="sr-only sm:hidden">{health.text}</span>
    </span>
  )
}

// Review comes before Annotate: new judgments are labelled by checking the classifier's
// suggestions; full manual annotation is for the cases listed in ANNOTATE_NOTE.
const VIEWS = [
  { id: 'chat', label: 'Chat' },
  { id: 'review', label: 'Review' },
  { id: 'annotate', label: 'Annotate' },
  { id: 'verify', label: 'Verify' },
]

function AnnotateNote() {
  return (
    <p className="border-b border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-900" data-testid="annotate-note">
      <b>When to annotate by hand:</b> new judgments normally go through the <a href="#review" className="font-medium underline">Review</a> tab,
      where you only check the computer's suggestions. Use this tab for (1) cases the Review tab sent back because too many
      spot checks were wrong, (2) a small "gold" sample — about 1 case in 10, labelled without seeing suggestions, so we can
      keep measuring how accurate the computer is — and (3) the same case labelled by two people, to check that the
      label guideline is applied consistently.
    </p>
  )
}

function Nav({ view, className }) {
  const ref = useRef(null)
  return (
    <nav ref={ref} className={`relative gap-1 ${className}`} aria-label="Sections">
      <SlidingPill containerRef={ref} selector='[aria-current="page"]' deps={[view]} className="rounded-md bg-indigo-50" />
      {VIEWS.map((v) => (
        <a
          key={v.id}
          href={`#${v.id}`}
          aria-current={view === v.id ? 'page' : undefined}
          className={`relative shrink-0 rounded-md px-3 py-1.5 text-sm font-medium ${
            view === v.id ? 'text-indigo-700' : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          {v.label}
        </a>
      ))}
    </nav>
  )
}

function viewFromHash() {
  const id = window.location.hash.replace('#', '')
  return VIEWS.some((v) => v.id === id) ? id : 'chat'
}

export default function App() {
  const [view, setView] = useState(viewFromHash)

  useEffect(() => {
    const onHash = () => setView(viewFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  return (
    <div className="flex h-dvh flex-col bg-slate-50 text-slate-900">
      <a href="#main" onClick={(e) => { e.preventDefault(); document.getElementById('main')?.focus() }}
        className="sr-only z-50 rounded-md bg-accent px-3 py-2 text-sm font-medium text-on-accent focus:not-sr-only focus:absolute focus:left-3 focus:top-3">
        Skip to content
      </a>
      <header className="relative z-10 border-b border-slate-200 bg-white/85 px-4 py-2 backdrop-blur-md sm:py-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-4">
            <div className="flex min-w-0 items-center gap-2.5">
              <span aria-hidden="true" className="grid size-8 shrink-0 place-items-center rounded-lg bg-accent font-serif text-lg font-bold text-on-accent">§</span>
              <div className="min-w-0">
                <h1 className="text-lg font-semibold leading-tight tracking-tight text-slate-900">LegalGPT</h1>
                <p className="text-xs text-slate-500">
                  <span className="hidden sm:inline">Indian criminal law · legal information, not legal advice</span>
                  <span className="sm:hidden">Not legal advice</span>
                </p>
              </div>
            </div>
            <Nav view={view} className="hidden sm:flex" />
          </div>
          <div className="flex shrink-0 items-center gap-3">
            <HealthBadge />
            <ThemeMenu />
          </div>
        </div>
        <Nav view={view} className="-mx-1 mt-2 flex overflow-x-auto sm:hidden" />
      </header>
      {/* The chat stays mounted on other tabs (only hidden), so the conversation and an answer that
          is still streaming survive switching to Review or Verify and back. */}
      <main id="main" tabIndex={-1} className="flex min-h-0 flex-1 flex-col outline-none">
      <div className={view === 'chat' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
        <ChatWindow />
      </div>
      {view === 'annotate' ? (
        <div key="annotate" className="flex min-h-0 flex-1 animate-rise flex-col">
          <AnnotateNote />
          <AnnotateView />
        </div>
      ) : view === 'review' ? (
        <div key="review" className="flex min-h-0 flex-1 animate-rise flex-col">
          <ReviewView />
        </div>
      ) : view === 'verify' ? (
        <div key="verify" className="flex min-h-0 flex-1 animate-rise flex-col">
          <VerifyPanel />
        </div>
      ) : null}
      </main>
    </div>
  )
}
