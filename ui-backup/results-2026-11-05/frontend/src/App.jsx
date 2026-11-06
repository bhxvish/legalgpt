import { useEffect, useRef, useState } from 'react'
import { API_BASE_URL } from './api.js'
import CaseAnalysisView from './components/CaseAnalysisView.jsx'
import ChatWindow from './components/ChatWindow.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import ExplainabilityView from './components/ExplainabilityView.jsx'
import LabelView from './components/LabelView.jsx'
import SystemView from './components/SystemView.jsx'
import VerifyPanel from './components/VerifyPanel.jsx'
import { LogoMark, TabIcons } from './components/LawArt.jsx'
import { SlidingPill } from './components/motion.jsx'
import ThemeMenu from './components/ThemeMenu.jsx'

const DOT = { loading: 'bg-paper/40 animate-pulse', ok: 'bg-emerald-400', error: 'bg-field-red' }

function HealthBadge({ className = '' }) {
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
      className={`inline-flex items-center gap-1.5 whitespace-nowrap text-[13px] text-paper/70 ${className}`}
    >
      <span aria-hidden="true" className={`size-1.5 rounded-full ${DOT[health.state]}`} />
      {label}
      <span className="sr-only">{health.text}</span>
    </span>
  )
}

// In pipeline order: the overview, building the knowledge (offline), using it (every question), then
// opening it up. Each part has its chapter colour. `module`: the module number in the paper.
const VIEWS = [
  { id: 'system', label: 'Overview', fill: 'bg-field-red text-ink', group: 'overview' },
  { id: 'label', label: 'Label', fill: 'bg-paper text-ink', group: 'build', module: 'M1·M2' },
  { id: 'chat', label: 'Chat', fill: 'bg-field-gold text-ink', group: 'use', module: 'M0' },
  { id: 'case', label: 'Case', fill: 'bg-field-green text-paper', group: 'use' },
  { id: 'verify', label: 'Verify', fill: 'bg-field-blue text-paper', group: 'use', module: 'M5' },
  { id: 'explain', label: 'Explain', fill: 'bg-field-bone text-ink', group: 'trust', module: 'M4' },
]
const GROUP_TITLE = { build: 'Build the knowledge (offline)', use: 'Use it (every question)', trust: 'Open the black box: explanations and bias' }
const fillOf = (id) => VIEWS.find((v) => v.id === id)?.fill || ''

function Nav({ view, className, itemClass, pillClass }) {
  const ref = useRef(null)
  return (
    <nav ref={ref} className={`relative ${className}`} aria-label="Sections">
      <SlidingPill containerRef={ref} selector='[aria-current="page"]' deps={[view]} className={`${fillOf(view).split(' ')[0]} ${pillClass}`} />
      {VIEWS.map((v) => (
        <a
          key={v.id}
          href={`#${v.id}`}
          title={GROUP_TITLE[v.group] ? `${GROUP_TITLE[v.group]}${v.module ? ` · module ${v.module.slice(1)}` : ''}` : 'How the parts fit together'}
          aria-current={view === v.id ? 'page' : undefined}
          className={`press relative flex flex-col items-center justify-center gap-1 text-[13px] font-medium transition-colors duration-300 ${itemClass} ${
            view === v.id ? fillOf(v.id).split(' ')[1] : 'text-paper/65 hover:text-paper'
          }`}
        >
          {TabIcons[v.id]}
          {v.label}
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

const calm = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

export default function App() {
  const [view, setView] = useState(viewFromHash)
  const [wipe, setWipe] = useState(null) // the colour sweeping across while the part changes

  useEffect(() => {
    let timer
    const onHash = () => {
      const next = viewFromHash()
      clearTimeout(timer)
      if (calm()) return setView(next)
      // a new chapter: its colour rises over the page, and the part changes behind it
      setWipe({ fill: fillOf(next), key: Date.now() })
      timer = setTimeout(() => setView(next), 350)
    }
    window.addEventListener('hashchange', onHash)
    return () => {
      window.removeEventListener('hashchange', onHash)
      clearTimeout(timer)
    }
  }, [])

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-slate-50 text-slate-900 md:flex-row">
      <a href="#main" onClick={(e) => { e.preventDefault(); document.getElementById('main')?.focus() }}
        className="sr-only z-50 bg-accent px-3 py-2 text-sm font-medium text-on-accent focus:not-sr-only focus:absolute focus:left-3 focus:top-3">
        Skip to content
      </a>

      {/* wide screens: a black rail down the side */}
      <aside className="relative z-20 hidden w-24 shrink-0 flex-col bg-field-black md:flex">
        <a href="#system" className="press flex h-24 items-center justify-center" aria-label="LegalGPT, go to the overview">
          <LogoMark className="size-11" />
        </a>
        <Nav view={view} className="flex flex-col" itemClass="h-[4.75rem] w-full" pillClass="" />
        <div className="mt-auto flex flex-col items-center gap-3 pb-5">
          <HealthBadge />
          <ThemeMenu />
        </div>
      </aside>

      {/* small screens: a slim bar above, the sections along the bottom */}
      <header className="relative z-20 flex h-12 shrink-0 items-center justify-between bg-field-black pl-3 pr-2 md:hidden">
        <a href="#system" className="flex items-center gap-2 text-paper" aria-label="LegalGPT, go to the overview">
          <LogoMark className="size-7" />
          <span className="text-lg font-semibold tracking-[-0.03em]">LegalGPT</span>
        </a>
        <span className="flex items-center gap-3">
          <HealthBadge />
          <ThemeMenu />
        </span>
      </header>

      {/* The working pages stay mounted on other tabs (only hidden), so a conversation, an analysis, a
          check in progress and its result, and anything still streaming, survive switching tabs. */}
      <main id="main" tabIndex={-1} className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden outline-none">
        <div className={view === 'chat' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
          <ErrorBoundary>
            <ChatWindow />
          </ErrorBoundary>
        </div>
        <div className={view === 'case' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
          <ErrorBoundary>
            <CaseAnalysisView />
          </ErrorBoundary>
        </div>
        <div className={view === 'verify' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
          <ErrorBoundary>
            <VerifyPanel active={view === 'verify'} />
          </ErrorBoundary>
        </div>
        <div className={view === 'label' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
          <ErrorBoundary>
            <LabelView />
          </ErrorBoundary>
        </div>
        <div className={view === 'explain' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
          <ErrorBoundary>
            <ExplainabilityView />
          </ErrorBoundary>
        </div>
        {view === 'system' && (
          <div key="system" className="flex min-h-0 flex-1 flex-col">
            <ErrorBoundary>
              <SystemView />
            </ErrorBoundary>
          </div>
        )}
        {wipe && (
          <div
            key={wipe.key}
            aria-hidden="true"
            onAnimationEnd={() => setWipe(null)}
            className={`pointer-events-none absolute inset-0 z-30 animate-wipe ${wipe.fill}`}
          />
        )}
      </main>

      <Nav view={view} className="relative z-20 flex h-16 shrink-0 bg-field-black md:hidden" itemClass="min-w-0 flex-1" pillClass="" />
    </div>
  )
}
