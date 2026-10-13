import { useEffect, useState } from 'react'
import { zkApi } from '../api.js'

// What this server can prove / verify (GET /api/zk/manifest), fetched once and shared.
let manifestPromise = null
export function useZkManifest() {
  const [m, setM] = useState(null)
  useEffect(() => {
    manifestPromise ||= zkApi.manifest().catch(() => null)
    manifestPromise.then(setM)
  }, [])
  return m
}

function download(name, obj) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 1)], { type: 'application/json' }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function Checks({ result, label = 'Proof' }) {
  return (
    <div className="space-y-1" data-testid="zk-checks">
      <span
        className={`inline-block rounded-md px-2 py-0.5 text-xs font-semibold ring-1 ${
          result.valid ? 'bg-emerald-50 text-emerald-800 ring-emerald-200' : 'bg-red-50 text-red-700 ring-red-200'
        }`}
        data-testid="zk-badge"
      >
        {result.valid ? `${label} verified ✓` : 'Verification failed ✗'}
      </span>
      <ul className="space-y-0.5 text-xs">
        {result.checks.map((c) => (
          <li key={c.name} className={c.ok ? 'text-slate-600' : 'text-red-700'}>
            {c.ok ? '✓' : '✗'} {c.name}
            {c.detail && <span className="text-slate-400"> ({c.detail})</span>}
          </li>
        ))}
      </ul>
    </div>
  )
}

const kb = (obj) => `${Math.round(JSON.stringify(obj).length / 1024)} KB`

/** Verify tab: prove that the verdict follows from the published rules applied to these facts. */
export function VerdictProof({ result }) {
  const manifest = useZkManifest()
  const [bundle, setBundle] = useState(null)
  const [check, setCheck] = useState(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    setBundle(null)
    setCheck(null)
    setError('')
  }, [result])

  if (!manifest?.available?.verdict_prove) return null

  const prove = async () => {
    setBusy('prove')
    setError('')
    setCheck(null)
    try {
      setBundle(await zkApi.proveVerdict(result.facts, result.cited_section))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy('')
    }
  }
  const verify = async () => {
    setBusy('verify')
    setError('')
    try {
      setCheck(await zkApi.verify(bundle))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy('')
    }
  }
  const proven = check?.cited

  return (
    <section className="space-y-2 rounded-xl bg-white p-4 text-sm ring-1 ring-slate-200" data-testid="zk-verdict">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="font-semibold text-slate-800">Zero-knowledge proof</h3>
        <span className="text-xs text-slate-500">Halo2 via EZKL</span>
      </div>
      <p className="text-xs text-slate-600">
        Proves that this verdict is what the published rules give for the extracted facts, without revealing the facts:
        the proof contains only a hash of them. It does <b>not</b> prove the facts were extracted correctly.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={prove}
          disabled={!!busy}
          className="rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-900 disabled:opacity-50"
          data-testid="zk-prove"
        >
          {busy === 'prove' ? 'Generating proof…' : bundle ? 'Generate again' : 'Generate proof'}
        </button>
        {bundle && (
          <>
            <button
              type="button"
              onClick={verify}
              disabled={!!busy}
              className="rounded-lg px-3 py-1.5 text-xs font-medium text-slate-700 ring-1 ring-slate-300 hover:bg-slate-50 disabled:opacity-50"
              data-testid="zk-verify"
            >
              {busy === 'verify' ? 'Verifying…' : 'Verify proof'}
            </button>
            <button
              type="button"
              onClick={() => download(`verdict-proof-s${result.cited_section}.json`, bundle)}
              className="text-xs text-indigo-600 hover:underline"
            >
              Download proof
            </button>
            <span className="text-xs text-slate-500">
              made in {bundle.prove_seconds}s · {kb(bundle.proof)}
            </span>
          </>
        )}
      </div>
      {error && <p className="text-xs text-red-700">{error}</p>}
      {check && <Checks result={check} />}
      {proven && (
        <p className="text-xs text-slate-700">
          The proof says s.{proven.section} is <b>{proven.status}</b>: {proven.elements.filter((e) => e.status === 'satisfied').length} of{' '}
          {proven.elements.length} elements satisfied.
          {proven.status !== result.status && <b className="text-red-700"> This differs from the result shown above.</b>}
        </p>
      )}
    </section>
  )
}

/** Chat answer: signature check of the signed record, and a proof that the sources were not cherry-picked. */
export function AnswerProof({ msg }) {
  const manifest = useZkManifest()
  const [sig, setSig] = useState(null)
  const [retr, setRetr] = useState(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const att = msg.attestation
  if (!att) return null
  const followUp = att.record?.follow_up_context
  const canProve = manifest?.available?.retrieval_prove && !msg.refused && msg.question && !followUp

  const checkSignature = async () => {
    setBusy('sig')
    setError('')
    try {
      setSig(await zkApi.verify(att, { question: msg.question, answer: msg.content }))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy('')
    }
  }
  const proveSources = async () => {
    setBusy('retr')
    setError('')
    try {
      const bundle = await zkApi.proveRetrieval(msg.question)
      const check = await zkApi.verify(bundle)
      const shown = (msg.sources || []).map((s) => s.chunk_id)
      setRetr({ bundle, check, same: JSON.stringify(shown) === JSON.stringify(bundle.sources) })
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy('')
    }
  }

  return (
    <details className="mt-2 text-xs text-slate-600" data-testid="answer-proof">
      <summary className="cursor-pointer select-none">
        <span className="rounded bg-slate-100 px-1.5 py-0.5 font-medium text-slate-700">Signed</span> verifiable answer
      </summary>
      <div className="mt-2 space-y-2 rounded-lg bg-slate-50 p-3 ring-1 ring-slate-200">
        <p>
          This server signed a record of the question, the exact prompt, the cited sections and this answer (Ed25519). That
          proves where the answer came from and that it was not changed — not that it is legally right.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={checkSignature} disabled={!!busy} className="rounded-md px-2 py-1 font-medium ring-1 ring-slate-300 hover:bg-white disabled:opacity-50">
            {busy === 'sig' ? 'Checking…' : 'Check signature'}
          </button>
          {canProve && (
            <button
              type="button"
              onClick={proveSources}
              disabled={!!busy}
              className="rounded-md bg-slate-800 px-2 py-1 font-medium text-white hover:bg-slate-900 disabled:opacity-50"
              data-testid="zk-prove-retrieval"
            >
              {busy === 'retr' ? 'Proving (about 10 s)…' : 'Prove the sources'}
            </button>
          )}
          <button type="button" onClick={() => download('signed-answer.json', att)} className="text-indigo-600 hover:underline">
            Download record
          </button>
        </div>
        {followUp && <p className="text-slate-500">Source proofs cover single questions; this one also searched with the previous question.</p>}
        {error && <p className="text-red-700">{error}</p>}
        {sig && <Checks result={sig} label="Signature" />}
        {retr && (
          <div className="space-y-1">
            <p>
              Zero-knowledge proof of the similarity of your question to all {manifest?.corpus?.chunks ?? ''} IPC chunks, made in{' '}
              {retr.bundle.prove_seconds}s. Re-running the retrieval rules on the proven scores gives{' '}
              {retr.same ? (
                <b className="text-emerald-700">exactly the sources shown above</b>
              ) : (
                <b className="text-amber-700">different sources from those shown ({retr.bundle.sources.length} proven)</b>
              )}
              .
            </p>
            <Checks result={retr.check} />
          </div>
        )}
      </div>
    </details>
  )
}
