import { useEffect, useState } from 'react'
import { API_BASE_URL } from './api.js'

const STYLES = {
  loading: 'bg-slate-100 text-slate-600',
  ok: 'bg-emerald-100 text-emerald-800',
  error: 'bg-red-100 text-red-800',
}

export default function App() {
  const [health, setHealth] = useState({ state: 'loading', text: 'Checking backend…' })

  useEffect(() => {
    const controller = new AbortController()
    fetch(`${API_BASE_URL}/health`, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => setHealth({ state: data.status === 'ok' ? 'ok' : 'error', text: data.status }))
      .catch((err) => {
        if (err.name !== 'AbortError') setHealth({ state: 'error', text: `unreachable (${err.message})` })
      })
    return () => controller.abort()
  }, [])

  return (
    <main className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md rounded-xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
        <h1 className="text-2xl font-semibold text-slate-900">LegalGPT</h1>
        <p className="mt-1 text-sm text-slate-500">Indian criminal law assistant — Phase 0 scaffold</p>
        <div className="mt-6 flex items-center justify-between">
          <span className="text-sm text-slate-700">Backend health</span>
          <span data-testid="health-status" className={`rounded-full px-3 py-1 text-sm font-medium ${STYLES[health.state]}`}>
            {health.text}
          </span>
        </div>
        <p className="mt-2 text-xs text-slate-400">{API_BASE_URL}/health</p>
      </div>
    </main>
  )
}
