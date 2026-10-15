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
  document.documentElement.dataset.theme = dark ? 'dark' : 'light'
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
