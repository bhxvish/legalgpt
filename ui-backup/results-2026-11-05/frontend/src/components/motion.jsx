// Small motion helpers. All animation here uses transform, opacity or clip-path and stops under
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
      className={`pointer-events-none absolute left-0 top-0 ${animate ? 'transition-[transform,width,height,background-color] duration-[420ms] ease-[cubic-bezier(0.76,0,0.24,1)]' : ''} ${className}`}
      style={{ width: box.w, height: box.h, transform: `translate(${box.x}px, ${box.y}px)` }}
    />
  )
}

/** Delay for the n-th item of a staggered entry (used with animate-rise). */
export const stagger = (i, step = 60, start = 0) => ({ animationDelay: `${start + i * step}ms` })

/** True once the element has scrolled into view; it stays true (reveals never replay). */
export function useSeen(ref, margin = '0px 0px -12% 0px') {
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        setSeen(true)
        io.disconnect()
      }
    }, { rootMargin: margin, threshold: 0.05 })
    io.observe(el)
    return () => io.disconnect()
  }, [ref, margin])
  return seen
}

/** A part of a section that arrives as it scrolls in. `clip` opens it from the top like a shutter. */
export function Reveal({ as: Tag = 'div', delay = 0, clip = false, className = '', style, children, ...rest }) {
  const ref = useRef(null)
  const seen = useSeen(ref)
  return (
    <Tag ref={ref} className={`${clip ? 'reveal-clip' : 'reveal'} ${seen ? 'is-in' : ''} ${className}`} style={{ transitionDelay: `${delay}ms`, ...style }} {...rest}>
      {children}
    </Tag>
  )
}

/** A plain number counts up once when `run` turns true; anything else is shown as it is. */
export function useCount(text, run) {
  const target = /^[\d,]+$/.test(String(text)) ? Number(String(text).replace(/,/g, '')) : null
  const [n, setN] = useState(0)
  useEffect(() => {
    if (target == null || !run) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return setN(target)
    const start = performance.now()
    let raf
    const tick = (t) => {
      const p = Math.min(1, (t - start) / 1200)
      setN(Math.round(target * (1 - (1 - p) ** 3)))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, run])
  return target == null ? text : n.toLocaleString('en-IN')
}
