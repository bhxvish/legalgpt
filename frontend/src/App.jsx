import { useEffect, useState } from 'react'
import { API_BASE_URL } from './api.js'
import AnnotateView from './components/AnnotateView.jsx'
import ChatWindow from './components/ChatWindow.jsx'

const STYLES = {
  loading: 'bg-slate-100 text-slate-600',
  ok: 'bg-emerald-100 text-emerald-800',
  error: 'bg-red-100 text-red-800',
}

function HealthBadge() {
  const [health, setHealth] = useState({ state: 'loading', text: 'checking…' })

  useEffect(() => {
    const controller = new AbortController()
    fetch(`${API_BASE_URL}/health`, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => setHealth({ state: data.status === 'ok' ? 'ok' : 'error', text: `backend ${data.status}` }))
      .catch((err) => {
        if (err.name !== 'AbortError') setHealth({ state: 'error', text: 'backend unreachable' })
      })
    return () => controller.abort()
  }, [])

  return (
    <span data-testid="health-status" title={`${API_BASE_URL}/health`} className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${STYLES[health.state]}`}>
      {health.text}
    </span>
  )
}

const VIEWS = [
  { id: 'chat', label: 'Chat' },
  { id: 'annotate', label: 'Annotate' },
]

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
    <div className="flex h-dvh flex-col bg-slate-50">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 bg-white px-4 py-3">
        <div className="flex items-center gap-4">
          <div>
            <h1 className="text-lg font-semibold text-slate-900">LegalGPT</h1>
            <p className="text-xs text-slate-500">Indian criminal law · legal information, not legal advice</p>
          </div>
          <nav className="flex gap-1" aria-label="Sections">
            {VIEWS.map((v) => (
              <a
                key={v.id}
                href={`#${v.id}`}
                aria-current={view === v.id ? 'page' : undefined}
                className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                  view === v.id ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                {v.label}
              </a>
            ))}
          </nav>
        </div>
        <HealthBadge />
      </header>
      {view === 'annotate' ? <AnnotateView /> : <ChatWindow />}
    </div>
  )
}
