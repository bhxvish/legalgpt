import { useEffect, useRef, useState } from 'react'
import { API_BASE_URL } from './api.js'
import AnnotateView from './components/AnnotateView.jsx'
import CaseAnalysisView from './components/CaseAnalysisView.jsx'
import ChatWindow from './components/ChatWindow.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import ReviewView from './components/ReviewView.jsx'
import SystemView from './components/SystemView.jsx'
import VerifyPanel from './components/VerifyPanel.jsx'
import { LogoMark, TabIcons } from './components/LawArt.jsx'
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

  const label = { loading: 'Connecting', ok: 'Online', error: 'Offline' }[health.state]
  return (
    <span
      data-testid="health-status"
      title={`${health.text} (${API_BASE_URL}/health)`}
      role="status"
      className={`inline-flex h-8 items-center gap-2 whitespace-nowrap rounded-full bg-white/70 px-2.5 text-xs font-medium ring-1 ring-slate-200 ${STYLES[health.state].badge}`}
    >
      <span aria-hidden="true" className="relative inline-flex size-2">
        {health.state === 'ok' && <span className="absolute inset-0 animate-ping-soft rounded-full bg-emerald-500" />}
        <span className={`relative size-2 rounded-full ${STYLES[health.state].dot}`} />
      </span>
      <span className="hidden sm:inline">{label}</span>
      <span className="sr-only">{health.text}</span>
    </span>
  )
}

// In pipeline order, in the groups of the System overview: building the knowledge (offline), then
// using it (every question). `module`: the module number in the paper.
const VIEWS = [
  { id: 'system', label: 'Overview', short: 'Home', group: 'overview' },
  { id: 'annotate', label: 'Annotate', group: 'build', module: 'M1' },
  { id: 'review', label: 'Review', group: 'build', module: 'M2' },
  { id: 'chat', label: 'Chat', group: 'use', module: 'M0' },
  { id: 'case', label: 'Case analysis', short: 'Case', group: 'use' },
  { id: 'verify', label: 'Verify', group: 'use', module: 'M5' },
]
const GROUP_TITLE = { build: 'Build the knowledge (offline)', use: 'Use it (every question)' }

function Nav({ view, className }) {
  const ref = useRef(null)
  return (
    <nav ref={ref} className={`relative gap-0.5 rounded-full bg-slate-100/80 p-1 ring-1 ring-slate-200/70 ${className}`} aria-label="Sections">
      <SlidingPill
        containerRef={ref}
        selector='[aria-current="page"]'
        deps={[view]}
        className="rounded-full bg-white shadow-[0_1px_2px_rgb(15_23_42/0.08),0_2px_8px_-2px_rgb(15_23_42/0.12)] ring-1 ring-slate-900/5"
      />
      {VIEWS.map((v, i) => (
        <a
          key={v.id}
          href={`#${v.id}`}
          title={GROUP_TITLE[v.group] ? `${GROUP_TITLE[v.group]}${v.module ? ` · module ${v.module.slice(1)}` : ''}` : 'How the parts fit together'}
          aria-current={view === v.id ? 'page' : undefined}
          data-group-start={i > 0 && VIEWS[i - 1].group !== v.group ? '' : undefined}
          className={`group relative inline-flex shrink-0 items-center justify-center gap-1.5 rounded-full px-3.5 py-1.5 text-sm font-medium max-md:min-w-0 max-md:flex-1 max-md:px-1 max-md:text-[13px] ${
            view === v.id ? 'text-slate-900' : 'text-slate-500 hover:text-slate-900'
          }`}
        >
          <span className={`transition-[color,transform] duration-200 group-hover:scale-110 max-md:hidden ${view === v.id ? 'text-indigo-600' : 'text-slate-400 group-hover:text-slate-600'}`}>
            {TabIcons[v.id]}
          </span>
          {v.short ? (
            <>
              <span className="md:hidden">{v.short}</span>
              <span className="hidden md:inline">{v.label}</span>
            </>
          ) : (
            v.label
          )}
          {v.module && (
            <span className={`hidden font-mono text-[9.5px] font-semibold xl:inline ${view === v.id ? 'text-indigo-500' : 'text-slate-400'}`}>{v.module}</span>
          )}
        </a>
      ))}
    </nav>
  )
}

function viewFromHash() {
  const id = window.location.hash.replace('#', '').split('?')[0]
  return VIEWS.some((v) => v.id === id) ? id : 'system'
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
      <header className="relative z-10 bg-white/75 px-3 backdrop-blur-xl backdrop-saturate-150 sm:px-5">
        <div className="flex h-14 items-center justify-between gap-3 md:grid md:grid-cols-[1fr_auto_1fr]">
          <a href="#system" className="group flex min-w-0 items-center gap-2.5 justify-self-start rounded-xl pr-2" aria-label="LegalGPT, go to the overview">
            <LogoMark className="size-9 transition-transform duration-300 group-hover:-rotate-6 group-hover:scale-105" />
            <span className="min-w-0">
              <h1 className="text-[1.05rem] font-semibold leading-none tracking-tight text-slate-900">
                Legal<span className="text-indigo-600">GPT</span>
              </h1>
              <span className="mt-1 hidden truncate text-[11px] font-medium leading-none tracking-wide text-slate-500 sm:block md:hidden xl:block">
                Indian Penal Code assistant
              </span>
            </span>
          </a>
          <Nav view={view} className="hidden md:flex" />
          <div className="flex shrink-0 items-center gap-2 justify-self-end">
            <HealthBadge />
            <ThemeMenu />
          </div>
        </div>
        <Nav view={view} className="mb-2.5 flex w-full md:hidden" />
        {/* hairline with a slow travelling highlight */}
        <div aria-hidden="true" className="absolute inset-x-0 bottom-0 h-px bg-slate-200/80">
          <div className="h-full w-full animate-[sheen_9s_linear_infinite] bg-[linear-gradient(90deg,transparent_0%,transparent_35%,var(--color-indigo-400)_50%,transparent_65%,transparent_100%)] bg-[length:200%_100%] opacity-70" />
        </div>
      </header>
      {/* The chat stays mounted on other tabs (only hidden), so the conversation and an answer that
          is still streaming survive switching to Review or Verify and back. */}
      <main id="main" tabIndex={-1} className="flex min-h-0 flex-1 flex-col outline-none">
      <div className={view === 'chat' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
        <ErrorBoundary>
          <ChatWindow />
        </ErrorBoundary>
      </div>
      {view === 'system' ? (
        <div key="system" className="flex min-h-0 flex-1 animate-rise flex-col">
          <ErrorBoundary>
            <SystemView />
          </ErrorBoundary>
        </div>
      ) : view === 'annotate' ? (
        <div key="annotate" className="flex min-h-0 flex-1 animate-rise flex-col">
          <AnnotateView />
        </div>
      ) : view === 'review' ? (
        <div key="review" className="flex min-h-0 flex-1 animate-rise flex-col">
          <ReviewView />
        </div>
      ) : view === 'case' ? (
        <div key="case" className="flex min-h-0 flex-1 animate-rise flex-col">
          <ErrorBoundary>
            <CaseAnalysisView />
          </ErrorBoundary>
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
