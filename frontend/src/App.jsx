import { useEffect, useState } from 'react'
import { API_BASE_URL } from './api.js'
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

export default function App() {
  return (
    <div className="flex h-dvh flex-col bg-slate-50">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3">
        <div>
          <h1 className="text-lg font-semibold text-slate-900">LegalGPT</h1>
          <p className="text-xs text-slate-500">Indian criminal law · legal information, not legal advice</p>
        </div>
        <HealthBadge />
      </header>
      <ChatWindow />
    </div>
  )
}
