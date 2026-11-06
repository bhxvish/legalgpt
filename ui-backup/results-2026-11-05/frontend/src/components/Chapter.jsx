/** A very large title whose lines rise out of a mask. */
export function Lines({ lines, as: Tag = 'h2', className = '', delay = 80 }) {
  return (
    <Tag className={className} aria-label={lines.join(' ')}>
      {lines.map((text, i) => (
        <span key={text} aria-hidden="true" className="mask-line">
          <span className="animate-line-up" style={{ animationDelay: `${delay + i * 110}ms` }}>{text}</span>
        </span>
      ))}
    </Tag>
  )
}
