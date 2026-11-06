import { useEffect, useRef, useState } from 'react'
import { API_BASE_URL } from './api.js'
import CaseAnalysisView from './components/CaseAnalysisView.jsx'
import ChatWindow from './components/ChatWindow.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import ExplainabilityView from './components/ExplainabilityView.jsx'
import LabelView from './components/LabelView.jsx'
import SystemView from './components/SystemView.jsx'
import VerifyPanel from './components/VerifyPanel.jsx'
import { LogoMark } from './components/LawArt.jsx'
import { SlidingPill } from './components/motion.jsx'
import ThemeMenu from './components/ThemeMenu.jsx'

const STYLES = {
  loading: { badge: 'text-slate-500', dot: 'bg-slate-400 animate-pulse' },
  ok: { badge: 'text-slate-500', dot: 'bg-emerald-500' },
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
      className={`inline-flex h-8 items-center gap-1.5 whitespace-nowrap text-[13px] font-medium ${STYLES[health.state].badge}`}
    >
      <span aria-hidden="true" className={`size-1.5 rounded-full ${STYLES[health.state].dot}`} />
      <span className="hidden sm:inline">{label}</span>
      <span className="sr-only">{health.text}</span>
    </span>
  )
}

// In pipeline order, in the groups of the System overview: building the knowledge (offline), then
// using it (every question). `module`: the module number in the paper.
const VIEWS = [
  { id: 'system', label: 'Overview', short: 'Home', group: 'overview' },
  { id: 'label', label: 'Label', group: 'build', module: 'M1·M2' },
  { id: 'chat', label: 'Chat', group: 'use', module: 'M0' },
  { id: 'case', label: 'Case analysis', short: 'Case', group: 'use' },
  { id: 'verify', label: 'Verify', group: 'use', module: 'M5' },
  { id: 'explain', label: 'Explainability', short: 'Explain', group: 'trust', module: 'M4' },
]
const GROUP_TITLE = { build: 'Build the knowledge (offline)', use: 'Use it (every question)', trust: 'Open the black box: explanations and bias' }

function Nav({ view, className }) {
  const ref = useRef(null)
  return (
    <nav ref={ref} className={`relative ${className}`} aria-label="Sections">
      <SlidingPill containerRef={ref} selector='[aria-current="page"]' deps={[view]} className="border-b-2 border-indigo-600" />
      {VIEWS.map((v) => (
        <a
          key={v.id}
          href={`#${v.id}`}
          title={GROUP_TITLE[v.group] ? `${GROUP_TITLE[v.group]}${v.module ? ` · module ${v.module.slice(1)}` : ''}` : 'How the parts fit together'}
          aria-current={view === v.id ? 'page' : undefined}
          className={`relative inline-flex h-full shrink-0 items-center justify-center px-3 text-sm font-medium max-md:min-w-0 max-md:flex-1 max-md:px-1 max-md:text-[13px] ${
            view === v.id ? 'text-slate-900' : 'text-slate-500 hover:text-slate-900'
          }`}
        >
          {v.short ? (
            <>
              <span className="md:hidden">{v.short}</span>
              <span className="hidden md:inline">{v.label}</span>
            </>
          ) : (
            v.label
          )}
        </a>
      ))}
    </nav>
  )
}

function viewFromHash() {
  const id = window.location.hash.replace('#', '').split('?')[0]
  const alias = { annotate: 'label', review: 'label' }[id] // the former tabs, merged into Label
  return alias || (VIEWS.some((v) => v.id === id) ? id : 'system')
}

export default function App() {
  const [view, setView] = useState(viewFromHash)

  useEffect(() => {
    const onHash = () => setView(viewFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-slate-50 text-slate-900">
      <a href="#main" onClick={(e) => { e.preventDefault(); document.getElementById('main')?.focus() }}
        className="sr-only z-50 rounded-md bg-accent px-3 py-2 text-sm font-medium text-on-accent focus:not-sr-only focus:absolute focus:left-3 focus:top-3">
        Skip to content
      </a>
      <header className="rule-double relative z-10 mb-[3px] bg-slate-50 px-3 sm:px-6">
        <div className="flex h-14 items-stretch justify-between gap-6">
          <a href="#system" className="flex min-w-0 items-center gap-2.5" aria-label="LegalGPT, go to the overview">
            <LogoMark className="size-7" />
            <h1 className="font-serif text-[1.4rem] font-medium leading-none tracking-[-0.015em] text-slate-900">LegalGPT</h1>
          </a>
          <Nav view={view} className="hidden flex-1 md:flex" />
          <div className="flex shrink-0 items-center gap-3">
            <HealthBadge />
            <ThemeMenu />
          </div>
        </div>
        <Nav view={view} className="flex h-10 w-full border-t border-slate-200 md:hidden" />
      </header>
      {/* The chat and the case analysis stay mounted on other tabs (only hidden), so a conversation,
          an analysis and its follow-up questions, and anything still streaming, survive switching
          tabs and back. */}
      <main id="main" tabIndex={-1} className="flex min-h-0 flex-1 flex-col outline-none">
      <div className={view === 'chat' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
        <ErrorBoundary>
          <ChatWindow />
        </ErrorBoundary>
      </div>
      <div className={view === 'case' ? 'flex min-h-0 flex-1 animate-rise flex-col' : 'hidden'}>
        <ErrorBoundary>
          <CaseAnalysisView />
        </ErrorBoundary>
      </div>
      {view === 'system' ? (
        <div key="system" className="flex min-h-0 flex-1 animate-rise flex-col">
          <ErrorBoundary>
            <SystemView />
          </ErrorBoundary>
        </div>
      ) : view === 'label' ? (
        <div key="label" className="flex min-h-0 flex-1 animate-rise flex-col">
          <ErrorBoundary>
            <LabelView />
          </ErrorBoundary>
        </div>
      ) : view === 'explain' ? (
        <div key="explain" className="flex min-h-0 flex-1 animate-rise flex-col">
          <ErrorBoundary>
            <ExplainabilityView />
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
