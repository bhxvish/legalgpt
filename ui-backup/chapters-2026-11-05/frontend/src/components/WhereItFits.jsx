// One line at the top of a tab: which module it is, and how its output flows into the other tabs
// (the System overview shows the whole picture).

const PARTS = {
  label: {
    module: 'M1 · M2',
    text: 'Checked labels train InLegalBERT, which gives every passage of the judgment library its role: the chat cites rulings, law applied and precedents, and Case analysis compares facts.',
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
    <p className={`max-w-2xl text-[15px] leading-snug ${className}`} data-testid="where-it-fits">
      {p.text}
      {p.links.map(([id, label]) => (
        <a key={id} href={`#${id}`} className="ml-3 whitespace-nowrap font-medium underline underline-offset-4">
          {label}
        </a>
      ))}
    </p>
  )
}
