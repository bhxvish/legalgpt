import { useEffect, useId, useRef, useState } from 'react'
import { useTheme } from '../theme.js'

const base = { viewBox: '0 0 20 20', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true }

const THEMES = [
  {
    id: 'system',
    label: 'System',
    hint: 'Follow this device',
    icon: (
      <svg {...base} className="size-4">
        <rect x="2.75" y="3.75" width="14.5" height="9.5" rx="1.75" />
        <path d="M7.5 16.25h5M10 13.25v3" />
      </svg>
    ),
  },
  {
    id: 'light',
    label: 'Light',
    hint: 'Always light',
    icon: (
      <svg {...base} className="size-4">
        <circle cx="10" cy="10" r="3.25" />
        <path d="M10 2.75v1.5M10 15.75v1.5M2.75 10h1.5M15.75 10h1.5M4.87 4.87l1.06 1.06M14.07 14.07l1.06 1.06M4.87 15.13l1.06-1.06M14.07 5.93l1.06-1.06" />
      </svg>
    ),
  },
  {
    id: 'dark',
    label: 'Dark',
    hint: 'Always dark',
    icon: (
      <svg {...base} className="size-4">
        <path d="M16.25 12.1A6.75 6.75 0 0 1 7.9 3.75a6.75 6.75 0 1 0 8.35 8.35Z" />
      </svg>
    ),
  },
]

/** Theme picker: a button that opens a small menu (System / Light / Dark), styled like the app
 * instead of the browser's native dropdown. Arrow keys, Enter, Escape and outside clicks work. */
export default function ThemeMenu() {
  const [theme, setTheme] = useTheme()
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const rootRef = useRef(null)
  const buttonRef = useRef(null)
  const listRef = useRef(null)
  const id = useId()
  const current = THEMES.find((t) => t.id === theme) || THEMES[0]

  useEffect(() => {
    if (!open) return
    setActive(THEMES.findIndex((t) => t.id === theme))
    listRef.current?.focus()
    const onDown = (e) => !rootRef.current?.contains(e.target) && setOpen(false)
    document.addEventListener('pointerdown', onDown)
    return () => document.removeEventListener('pointerdown', onDown)
  }, [open, theme])

  const choose = (t) => {
    setTheme(t.id)
    setOpen(false)
    buttonRef.current?.focus()
  }

  const onKey = (e) => {
    if (e.key === 'ArrowDown') setActive((i) => (i + 1) % THEMES.length)
    else if (e.key === 'ArrowUp') setActive((i) => (i + THEMES.length - 1) % THEMES.length)
    else if (e.key === 'Home') setActive(0)
    else if (e.key === 'End') setActive(THEMES.length - 1)
    else if (e.key === 'Enter' || e.key === ' ') choose(THEMES[active])
    else if (e.key === 'Escape' || e.key === 'Tab') {
      setOpen(false)
      if (e.key === 'Escape') buttonRef.current?.focus()
      return
    } else return
    e.preventDefault()
  }

  return (
    <div ref={rootRef} className="relative">
      <button
        ref={buttonRef}
        type="button"
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
        aria-label={`Colour theme: ${current.label}`}
        title="Colour theme"
        className={`inline-flex h-8 items-center gap-1.5 rounded-lg px-2 text-xs font-medium text-slate-600 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-900 ${open ? 'bg-slate-100 text-slate-900' : ''}`}
        data-testid="theme-select"
      >
        {current.icon}
        <span className="hidden sm:inline">{current.label}</span>
        <svg aria-hidden="true" viewBox="0 0 12 12" className={`size-3 text-slate-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}>
          <path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <ul
          ref={listRef}
          id={`${id}-list`}
          role="listbox"
          tabIndex={-1}
          aria-label="Colour theme"
          aria-activedescendant={`${id}-${THEMES[active]?.id}`}
          onKeyDown={onKey}
          className="absolute right-0 top-full z-50 mt-2 w-48 origin-top-right animate-pop rounded-xl bg-white p-1 shadow-[0_12px_32px_-12px_rgb(15_23_42/0.35)] ring-1 ring-slate-200 outline-none"
        >
          {THEMES.map((t, i) => (
            <li
              key={t.id}
              id={`${id}-${t.id}`}
              role="option"
              aria-selected={t.id === theme}
              onPointerEnter={() => setActive(i)}
              onClick={() => choose(t)}
              className={`flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm ${
                i === active ? 'bg-slate-100 text-slate-900' : 'text-slate-700'
              }`}
            >
              <span className={t.id === theme ? 'text-indigo-600' : 'text-slate-400'}>{t.icon}</span>
              <span className="flex-1">
                <span className="block font-medium leading-tight">{t.label}</span>
                <span className="block text-xs text-slate-500">{t.hint}</span>
              </span>
              {t.id === theme && (
                <svg {...base} strokeWidth={2} className="size-4 text-indigo-600">
                  <path d="m4.5 10.5 3.5 3.5 7.5-8" />
                </svg>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
