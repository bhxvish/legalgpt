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
      className="fixed inset-0 z-50 flex items-center justify-center bg-scrim/60 p-4 backdrop-blur-[2px] animate-fade"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="source-modal-title"
        className="flex max-h-[85vh] w-full max-w-2xl animate-pop flex-col rounded-2xl bg-white shadow-2xl ring-1 ring-slate-200"
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-200 px-5 py-4">
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Source [{source.marker}]{source.kind === 'judgment' ? ' · judgment' : ''}
            </p>
            <h2 id="source-modal-title" className="mt-1 text-base font-semibold text-slate-900">
              {source.kind === 'judgment'
                ? `${source.title}${source.year ? ` (${source.court?.includes('Supreme') ? 'SC, ' : ''}${source.year})` : ''}`
                : `${source.section ? `Section ${source.section}` : source.citation_path}${source.title ? ` — ${source.title}` : ''}`}
            </h2>
            {source.kind === 'judgment' && (
              <p className="mt-1 text-xs text-slate-500">
                {source.court || 'Court not recorded'} · passage labelled <b>{source.role}</b> by the sentence-role model (InLegalBERT ensemble){' '}
                <a href="#label" onClick={onClose} className="whitespace-nowrap font-medium text-indigo-600 hover:underline">
                  <span className="font-mono text-[10px]">M2</span> how roles are labelled →
                </a>
              </p>
            )}
            <p className="mt-1 break-words font-mono text-xs text-slate-600" data-testid="source-citation-path">
              {source.citation_path}
            </p>
            {source.doc_id === 'BNS' && source.ipc?.length > 0 && (
              <p className="mt-2 rounded-lg bg-indigo-50 px-2.5 py-1.5 text-xs text-indigo-900 ring-1 ring-indigo-200">
                {source.ipc[0] === 'new' ? (
                  <>A provision <b>new in the Bharatiya Nyaya Sanhita, 2023</b>: no IPC section corresponds to it.</>
                ) : (
                  <>Replaced <b>IPC Section {source.ipc.join(', ')}</b> from 1 July 2024.</>
                )}{' '}
                <span className="text-indigo-700/80">From the BPRD correspondence table.</span>
              </p>
            )}
            {source.kind !== 'judgment' && source.bns?.length > 0 && (
              <p className="mt-2 rounded-lg bg-indigo-50 px-2.5 py-1.5 text-xs text-indigo-900 ring-1 ring-indigo-200">
                {source.bns[0] === 'none' ? (
                  <>No direct equivalent in the <b>Bharatiya Nyaya Sanhita, 2023</b> (in force from 1 July 2024): omitted or folded into another provision.</>
                ) : (
                  <>Now <b>Section {source.bns.join(', ')}</b> of the <b>Bharatiya Nyaya Sanhita, 2023</b> (in force from 1 July 2024).</>
                )}{' '}
                <span className="text-indigo-700/80">From the BPRD correspondence table; check the Sanhita's text where it matters.</span>
              </p>
            )}
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
          <span>{source.kind === 'judgment' ? 'Passage' : 'Role'}: {source.role}</span>
          <span className="text-slate-400">Similarity measures how well this text matched the question, not whether the answer is correct.</span>
        </div>
      </div>
    </div>
  )
}
