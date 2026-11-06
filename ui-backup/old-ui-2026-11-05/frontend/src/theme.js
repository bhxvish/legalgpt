// Light / dark theme. The choice ("system", "light" or "dark") is kept in localStorage; index.html
// applies it before the first paint so the page does not flash light first.
import { useEffect, useState } from 'react'

const KEY = 'legalgpt.theme'
const media = () => window.matchMedia('(prefers-color-scheme: dark)')

function readChoice() {
  try {
    const v = localStorage.getItem(KEY)
    return v === 'light' || v === 'dark' ? v : 'system'
  } catch {
    return 'system'
  }
}

function apply(choice) {
  const dark = choice === 'dark' || (choice === 'system' && media().matches)
  const root = document.documentElement
  const next = dark ? 'dark' : 'light'
  if (root.dataset.theme === next) return
  // switch instantly: without this, every button fades through grey as its colours transition
  root.classList.add('theme-switching')
  root.dataset.theme = next
  requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove('theme-switching')))
}

// the dark palette: "graphite" (neutral, default), "warm" (ink and paper) or "midnight" (navy)
const STYLE_KEY = 'legalgpt.darkStyle'
export const DARK_STYLES = ['graphite', 'warm', 'midnight']

function readStyle() {
  try {
    const v = localStorage.getItem(STYLE_KEY)
    return DARK_STYLES.includes(v) ? v : 'graphite'
  } catch {
    return 'graphite'
  }
}

export function useDarkStyle() {
  const [style, setStyle] = useState(readStyle)
  useEffect(() => {
    const root = document.documentElement
    if (root.dataset.dark !== style) {
      root.classList.add('theme-switching')
      root.dataset.dark = style
      requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove('theme-switching')))
    }
    try {
      localStorage.setItem(STYLE_KEY, style)
    } catch {
      // private window: the choice lasts until reload
    }
  }, [style])
  return [style, setStyle]
}

export function useTheme() {
  const [choice, setChoice] = useState(readChoice)

  useEffect(() => {
    apply(choice)
    try {
      if (choice === 'system') localStorage.removeItem(KEY)
      else localStorage.setItem(KEY, choice)
    } catch {
      // private window: the choice lasts until reload
    }
    if (choice !== 'system') return
    const mq = media()
    const onChange = () => apply('system')
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [choice])

  return [choice, setChoice]
}
