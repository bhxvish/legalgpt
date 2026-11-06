import { useEffect, useId, useRef, useState } from 'react'

/** The app's own dropdown, used wherever the browser's would show. It opens from its button, marks
 * the chosen option, and works from the keyboard: arrows, Home, End, Enter, Escape.
 * options: [{ value, label, hint?, disabled? }]. `up` opens it above the button. */
export default function Select({ value, onChange, options, label, up = false, disabled = false, align = 'left', className = '', buttonClass = '', testId, menuClass = 'w-64' }) {
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const rootRef = useRef(null)
  const buttonRef = useRef(null)
  const listRef = useRef(null)
  const id = useId()
  const current = options.find((o) => o.value === value)

  useEffect(() => {
    if (!open) return
    setActive(Math.max(0, options.findIndex((o) => o.value === value)))
    listRef.current?.focus()
    const away = (e) => !rootRef.current?.contains(e.target) && setOpen(false)
    document.addEventListener('pointerdown', away)
    return () => document.removeEventListener('pointerdown', away)
  }, [open, options, value])

  const choose = (o) => {
    if (!o || o.disabled) return
    onChange(o.value)
    setOpen(false)
    buttonRef.current?.focus()
  }

  const step = (from, by) => {
    for (let i = 1; i <= options.length; i++) {
      const next = (from + by * i + options.length * i) % options.length
      if (!options[next].disabled) return next
    }
    return from
  }

  const onKey = (e) => {
    if (e.key === 'ArrowDown') setActive((i) => step(i, 1))
    else if (e.key === 'ArrowUp') setActive((i) => step(i, -1))
    else if (e.key === 'Home') setActive(step(-1, 1))
    else if (e.key === 'End') setActive(step(0, -1))
    else if (e.key === 'Enter' || e.key === ' ') choose(options[active])
    else if (e.key === 'Escape' || e.key === 'Tab') {
      setOpen(false)
      if (e.key === 'Escape') buttonRef.current?.focus()
      return
    } else return
    e.preventDefault()
  }

  return (
    <div ref={rootRef} className={`relative inline-flex min-w-0 ${className}`}>
      <button
        ref={buttonRef}
        type="button"
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault()
            setOpen(true)
          }
        }}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={`${id}-list`}
        aria-label={`${label}: ${current?.label || 'none'}`}
        data-testid={testId}
        className={`press inline-flex h-8 min-w-0 max-w-full items-center justify-between gap-1.5 rounded-lg px-2 text-sm font-medium disabled:opacity-50 ${buttonClass} ${
          open ? 'bg-slate-100 text-slate-950' : 'text-slate-700 hover:bg-slate-100 hover:text-slate-950'
        }`}
      >
        <span className="truncate">{current?.label || '—'}</span>
        <svg aria-hidden="true" viewBox="0 0 12 12" className={`size-3 shrink-0 text-slate-500 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}>
          <path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <ul
          ref={listRef}
          id={`${id}-list`}
          role="listbox"
          tabIndex={-1}
          aria-label={label}
          aria-activedescendant={`${id}-${active}`}
          onKeyDown={onKey}
          className={`absolute z-50 max-h-[26rem] animate-menu overflow-y-auto bg-field-panel p-1.5 text-paper shadow-[0_18px_50px_-12px_rgb(0_0_0/0.55)] outline-none ${menuClass} ${
            up ? 'bottom-full mb-2 origin-bottom' : 'top-full mt-2 origin-top'
          } ${align === 'right' ? 'right-0' : 'left-0'}`}
        >
          {options.map((o, i) => (
            <li
              key={o.value}
              id={`${id}-${i}`}
              role="option"
              aria-selected={o.value === value}
              aria-disabled={o.disabled || undefined}
              onPointerEnter={() => !o.disabled && setActive(i)}
              onClick={() => choose(o)}
              className={`flex items-center gap-3 px-3 py-2 ${o.disabled ? 'cursor-not-allowed opacity-40' : 'cursor-pointer'} ${
                i === active && !o.disabled ? 'bg-field-panel-2' : ''
              }`}
            >
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[15px] font-medium leading-tight">{o.label}</span>
                {o.hint && <span className="mt-0.5 block truncate text-[13px] leading-tight text-paper/60">{o.hint}</span>}
              </span>
              <span aria-hidden="true" className={`size-2 shrink-0 bg-field-gold transition-transform duration-200 ${o.value === value ? 'scale-100' : 'scale-0'}`} />
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
