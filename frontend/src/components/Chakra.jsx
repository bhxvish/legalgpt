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
