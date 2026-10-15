// Small motion helpers. All animation here uses transform/opacity and stops under
// prefers-reduced-motion (index.css).
import { useEffect, useLayoutEffect, useRef, useState } from 'react'

/**
 * A highlight that slides to the selected option of a segmented control or nav. Put it as the
 * first child of a `relative` container; `selector` finds the selected item inside it.
 */
export function SlidingPill({ containerRef, selector, deps = [], className = '' }) {
  const [box, setBox] = useState(null)
  const first = useRef(true)

  // a passive effect: the parent's ref is attached only after its children's layout effects run
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const measure = () => {
      const el = container.querySelector(selector)
      setBox(el ? { x: el.offsetLeft, y: el.offsetTop, w: el.offsetWidth, h: el.offsetHeight } : null)
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(container)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  // no slide on first paint: the pill starts where the selected item is
  const animate = !first.current
  useLayoutEffect(() => {
    if (box) first.current = false
  }, [box])

  if (!box) return null
  return (
    <span
      aria-hidden="true"
      className={`pointer-events-none absolute left-0 top-0 ${animate ? 'transition-[transform,width,height] duration-300 ease-[cubic-bezier(0.3,0,0,1)]' : ''} ${className}`}
      style={{ width: box.w, height: box.h, transform: `translate(${box.x}px, ${box.y}px)` }}
    />
  )
}

/** Delay for the n-th item of a staggered entry (used with animate-rise). */
export const stagger = (i, step = 60, start = 0) => ({ animationDelay: `${start + i * step}ms` })
