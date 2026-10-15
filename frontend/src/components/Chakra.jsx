import { useEffect, useRef, useState } from 'react'

// Line drawing of the Ashoka Chakra (24 spokes), used as quiet decoration on the empty chat.
// Not the State Emblem of India, whose use is restricted by the State Emblem of India
// (Prohibition of Improper Use) Act, 2005.
const SPOKES = 24
const C = 100 // centre of the 200 x 200 drawing

function polar(r, deg) {
  const a = ((deg - 90) * Math.PI) / 180
  return [C + r * Math.cos(a), C + r * Math.sin(a)]
}

// each spoke: a slim leaf from the hub to the rim
const spoke = (deg) => {
  const [x1, y1] = polar(13, deg)
  const [x2, y2] = polar(84, deg)
  const [lx, ly] = polar(46, deg - 2.4)
  const [rx, ry] = polar(46, deg + 2.4)
  return `M${x1.toFixed(2)} ${y1.toFixed(2)} Q${lx.toFixed(2)} ${ly.toFixed(2)} ${x2.toFixed(2)} ${y2.toFixed(2)} Q${rx.toFixed(2)} ${ry.toFixed(2)} ${x1.toFixed(2)} ${y1.toFixed(2)}Z`
}

export default function Chakra({ className = '' }) {
  const step = 360 / SPOKES
  return (
    <svg viewBox="0 0 200 200" fill="none" stroke="currentColor" strokeWidth="0.9" aria-hidden="true" className={className}>
      <circle cx={C} cy={C} r="96" />
      <circle cx={C} cy={C} r="88" />
      <circle cx={C} cy={C} r="13" />
      <circle cx={C} cy={C} r="6" />
      {Array.from({ length: SPOKES }, (_, i) => {
        const deg = i * step
        const [bx, by] = polar(88, deg + step / 2)
        return (
          <g key={i}>
            <path d={spoke(deg)} />
            {/* the small half-circles on the rim, between the spokes */}
            <circle cx={bx} cy={by} r="2.4" />
          </g>
        )
      })}
    </svg>
  )
}

/**
 * The Chakra behind the chat (wide screens). Empty chat: large, mid-right, turning very slowly.
 * While chatting: it glides into the lower-right corner, smaller and fainter; while an answer is
 * being written it speeds up, with a soft glow, and eases back to its slow turn when it is done.
 * The speed is changed through the Web Animations playbackRate (blended over ~1.2 s), so the
 * wheel never jumps.
 */
export function ChakraBackdrop({ chatting = false, busy = false }) {
  const wheelRef = useRef(null)
  const animRef = useRef(null)
  // fast answers (Groq) finish in about a second: keep the whirl going for at least 2.5 s so it
  // reads as a gesture rather than a twitch
  const [spin, setSpin] = useState(false)
  const busySince = useRef(0)
  useEffect(() => {
    if (busy) {
      busySince.current = performance.now()
      setSpin(true)
      return
    }
    const t = setTimeout(() => setSpin(false), Math.max(0, 2500 - (performance.now() - busySince.current)))
    return () => clearTimeout(t)
  }, [busy])

  useEffect(() => {
    const el = wheelRef.current
    if (!el?.animate || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const anim = el.animate([{ transform: 'rotate(0deg)' }, { transform: 'rotate(360deg)' }], { duration: 180000, iterations: Infinity })
    animRef.current = anim
    return () => anim.cancel()
  }, [])

  useEffect(() => {
    const anim = animRef.current
    if (!anim) return
    const from = anim.playbackRate
    const to = spin ? 30 : 1 // one turn in 6 s while answering, 3 min otherwise
    const start = performance.now()
    let raf
    const step = (now) => {
      const t = Math.min(1, (now - start) / 1200)
      const eased = 1 - Math.pow(1 - t, 3)
      anim.playbackRate = from + (to - from) * eased
      if (t < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    // frames do not run in a background tab: make sure the wheel still ends at the right speed
    const done = setTimeout(() => (anim.playbackRate = to), 1300)
    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(done)
    }
  }, [spin])

  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10 hidden overflow-hidden xl:block">
      <div
        className="absolute -right-28 top-1/2 transition-[transform,opacity] duration-[1200ms] ease-[cubic-bezier(0.3,0,0,1)]"
        style={{
          transform: chatting ? 'translate(3.5rem, -2%) scale(0.58)' : 'translate(0, -50%) scale(1)',
          opacity: chatting ? 0.7 : 1,
        }}
      >
        <div
          className={`absolute inset-[18%] rounded-full bg-[radial-gradient(closest-side,var(--color-indigo-300),transparent)] blur-2xl transition-opacity duration-1000 ${
            spin ? 'animate-pulse opacity-30' : 'opacity-0'
          }`}
        />
        <div ref={wheelRef} className="relative">
          <Chakra className="size-[30rem] text-indigo-900/[0.08] [mask-image:radial-gradient(circle,#000_55%,transparent_75%)]" />
        </div>
      </div>
    </div>
  )
}
