import { Component } from 'react'

/** Shows a message instead of a blank page when a tab fails to render. */
export default class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('LegalGPT view crashed', error, info?.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="flex min-h-0 flex-1 items-start justify-center px-4 pt-16">
        <div className="w-full max-w-md animate-rise rounded-2xl bg-white p-6 text-sm ring-1 ring-slate-200" role="alert">
          <p className="font-semibold text-slate-900">This page hit an error</p>
          <p className="mt-1 text-slate-600">{String(this.state.error?.message || this.state.error)}</p>
          <div className="mt-4 flex gap-2">
            <button
              onClick={() => this.setState({ error: null })}
              className="rounded-lg bg-accent px-3.5 py-2 font-medium text-on-accent hover:bg-accent-hover"
            >
              Try again
            </button>
            <button onClick={() => window.location.reload()} className="rounded-lg px-3.5 py-2 font-medium text-slate-700 ring-1 ring-slate-200 hover:bg-slate-50">
              Reload
            </button>
          </div>
        </div>
      </div>
    )
  }
}
