// Passing work from one tab to another ("check these facts in the rule engine"): the sender leaves a
// payload for a view and switches to it; the view reads it when it opens. The payload is dropped
// shortly after the first read (not at once: React's StrictMode runs initialisers twice).
const pending = {}

export function sendTo(view, payload) {
  pending[view] = payload
  window.location.hash = view
}

export function takeHandoff(view) {
  const p = pending[view]
  if (p) setTimeout(() => pending[view] === p && delete pending[view], 1000)
  return p
}
