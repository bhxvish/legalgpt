import { useEffect, useRef } from 'react'

/** Shows one SourceEvidence: its citation path and the exact chunk text sent to the model. */
export default function SourceModal({ source, onClose }) {
  const closeRef = useRef(null)

  useEffect(() => {
    if (!source) return
    closeRef.current?.focus()
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [source, onClose])

  if (!source) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="source-modal-title"
        className="flex max-h-[85vh] w-full max-w-2xl flex-col rounded-xl bg-white shadow-xl ring-1 ring-slate-200"
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-200 px-5 py-4">
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Source [{source.marker}]</p>
            <h2 id="source-modal-title" className="mt-1 text-base font-semibold text-slate-900">
              {source.section ? `Section ${source.section}` : source.citation_path}
              {source.title ? ` — ${source.title}` : ''}
            </h2>
            <p className="mt-1 break-words font-mono text-xs text-slate-600" data-testid="source-citation-path">
              {source.citation_path}
            </p>
          </div>
          <button
            ref={closeRef}
            onClick={onClose}
            className="rounded-md px-2 py-1 text-sm text-slate-500 hover:bg-slate-100 hover:text-slate-800"
            aria-label="Close source"
          >
            ✕
          </button>
        </div>
        <div className="overflow-y-auto px-5 py-4">
          <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-relaxed text-slate-800" data-testid="source-text">
            {source.text}
          </pre>
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-slate-200 px-5 py-3 text-xs text-slate-500">
          <span>Retrieval similarity: {source.similarity.toFixed(2)}</span>
          <span>Role: {source.role}</span>
          <span className="text-slate-400">Similarity measures how well this text matched the question, not whether the answer is correct.</span>
        </div>
      </div>
    </div>
  )
}
