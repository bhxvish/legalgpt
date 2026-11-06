// The stages one answer went through, in order, each tagged with its module (Overview): translation,
// retrieval, judgment passages, the model, the source check, BNS, the signature. A stage opens its
// detail in the record under the answer.

function Step({ module, children, onClick, title, tone = 'default' }) {
  const tones = {
    default: 'bg-white text-slate-600 ring-slate-200 hover:ring-slate-300 hover:text-slate-900',
    good: 'bg-emerald-50 text-emerald-800 ring-emerald-200 hover:ring-emerald-300',
    warn: 'bg-amber-50 text-amber-800 ring-amber-200 hover:ring-amber-300',
  }
  const Tag = onClick ? 'button' : 'span'
  return (
    <Tag
      {...(onClick ? { type: 'button', onClick } : {})}
      title={title}
      className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[13px] ring-1 transition ${tones[tone]}`}
    >
      {module && <span className="text-[11px] font-semibold text-indigo-600">{module}</span>}
      {children}
    </Tag>
  )
}

const Arrow = () => (
  <span aria-hidden="true" className="text-slate-300">
    →
  </span>
)

export default function AnswerTrace({ msg, onCite, goTo }) {
  if (msg.mode !== 'legal' || msg.error || !msg.retrieval || (!msg.explanation && !msg.refused)) return null

  const r = msg.retrieval
  const statutes = (msg.sources || []).filter((s) => s.kind !== 'judgment')
  const judgments = (msg.sources || []).filter((s) => s.kind === 'judgment')
  const cited = new Set([...(msg.content || '').matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])))
  const ex = msg.explanation
  const claims = (ex?.attributions || []).filter((a) => a.kind !== 'analysis')
  const supported = claims.filter((a) => a.supported).length
  const why = () => goTo('check')
  const steps = []

  if (r.translation?.failed)
    return (
      <div className="mt-3 flex flex-wrap items-center gap-1.5" data-testid="answer-trace">
        <Step tone="warn" title="No model could translate the question (usage limits), so the IPC was not searched">
          Hindi → English failed · not searched
        </Step>
      </div>
    )
  if (r.background_ignored?.length)
    steps.push(
      <Step key="bg" onClick={why} title="Searched on the facts only">
        Left out of the search, because the law does not turn on it: {r.background_ignored.join(' · ')}
      </Step>,
    )
  if (r.translation)
    steps.push(
      <Step key="hi" onClick={why} title={`Searched in English: "${r.translation.text}"`}>
        Hindi → English
      </Step>,
    )
  if (r.law)
    steps.push(
      <Step key="law" title={`Why this code: ${r.law_reason}`}>
        {r.law === 'bns' ? 'BNS' : 'IPC'}
      </Step>,
    )
  steps.push(
    msg.refused ? (
      <Step key="ret" module="M0" tone="warn" onClick={why} title="No IPC section matched closely enough, so the question was declined">
        no {r.law === 'bns' ? 'BNS' : 'IPC'} match · declined
      </Step>
    ) : (
      <Step key="ret" module="M0" onClick={why} title="How the sources were chosen">
        {statutes.length} {r.law === 'bns' ? 'BNS' : 'IPC'} section{statutes.length === 1 ? '' : 's'}
        {r.rewrite?.used ? ' · read as ' + r.rewrite.offences.map((o) => `s.${o.section}`).join(', ') : ''}
      </Step>
    ),
  )
  if (judgments.length)
    steps.push(
      <Step key="jud" module="M2" onClick={() => onCite(judgments[0])} title="Passages of real judgments; their roles (Ruling, Law Applied, Precedent) were labelled by the InLegalBERT ensemble">
        {judgments.length} judgment{judgments.length === 1 ? '' : 's'} · roles by InLegalBERT
      </Step>,
    )
  if (!msg.refused) {
    steps.push(
      <Step key="llm" module={/lora|adapter/i.test(msg.model || '') ? 'M3' : 'M0'} title="The model that wrote the answer from the numbered sources">
        {msg.model}
      </Step>,
    )
    if (claims.length)
      steps.push(
        <Step key="chk" module="M4" tone={supported === claims.length ? 'good' : 'warn'} onClick={why} title="Each legal statement matched to the source it rests on">
          {supported}/{claims.length} statements backed
        </Step>,
      )
    if (statutes.some((s) => s.bns?.length && s.bns[0] !== 'none'))
      steps.push(
        <Step key="bns" title="BNS (2023) equivalents from the official BPRD correspondence table">
          BNS mapped
        </Step>,
      )
  }
  if (msg.attestation)
    steps.push(
      <Step key="sig" title="A signed record of this answer; the sources can be proved in zero knowledge">
        signed
      </Step>,
    )

  return (
    <div className="mt-3 flex flex-wrap items-center gap-x-1.5 gap-y-1.5" data-testid="answer-trace" aria-label="How this answer was produced">
      {steps.flatMap((s, i) => (i ? [<Arrow key={`a${i}`} />, s] : [s]))}
    </div>
  )
}
