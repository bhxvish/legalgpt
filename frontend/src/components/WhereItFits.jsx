// One line at the top of a tab: which module it is, and how its output flows into the other tabs
// (the System overview shows the whole picture).

const PARTS = {
  annotate: {
    module: 'M1',
    text: 'Labels made here train the sentence-role classifier, which gives every passage of the judgment library its role.',
    links: [['review', 'Review (M2)'], ['system', 'Overview']],
  },
  review: {
    module: 'M2',
    text: 'The InLegalBERT ensemble labels new judgments and you check the unsure sentences. These roles decide what the chat cites (rulings, law applied, precedents) and what Case analysis compares (facts).',
    links: [['chat', 'Chat'], ['case', 'Case analysis'], ['system', 'Overview']],
  },
  verify: {
    module: 'M5',
    text: 'The same rules test the elements of each offence in Case analysis; "Check the facts" under a chat answer opens this tab with the question.',
    links: [['chat', 'Chat'], ['case', 'Case analysis'], ['system', 'Overview']],
  },
  case: {
    module: 'M0 · M2 · M4 · M5',
    text: 'The whole pipeline on one file: offences retrieved from the IPC, elements tested by the rule engine, similar judgments from the library, and a cited, explained analysis.',
    links: [['system', 'Overview']],
  },
}

export default function WhereItFits({ part, className = '' }) {
  const p = PARTS[part]
  if (!p) return null
  return (
    <p className={`flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500 ${className}`} data-testid="where-it-fits">
      <span className="rounded-md bg-indigo-50 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-indigo-700 ring-1 ring-indigo-200">{p.module}</span>
      <span className="min-w-0 flex-1 basis-64 leading-relaxed">{p.text}</span>
      <span className="flex flex-wrap gap-1.5">
        {p.links.map(([id, label]) => (
          <a key={id} href={`#${id}`} className="rounded-full bg-white px-2 py-0.5 font-medium text-slate-600 ring-1 ring-slate-200 transition hover:text-indigo-700 hover:ring-indigo-300">
            {label} →
          </a>
        ))}
      </span>
    </p>
  )
}
