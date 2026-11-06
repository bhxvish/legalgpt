// Saved conversations, kept in this browser's localStorage (nothing is sent to the server).
// A chat is { id, title, createdAt, updatedAt, messages }; newest first.

const KEY = 'legalgpt.chats.v1'
export const MAX_CHATS = 50

export function newId() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

export function chatTitle(messages) {
  const first = messages.find((m) => m.role === 'user')?.content?.trim() || 'New chat'
  return first.length > 60 ? `${first.slice(0, 57)}…` : first
}

/** Answers that were still streaming when the page closed can never finish: mark them. */
function settle(messages) {
  return messages.map((m) =>
    m.role === 'assistant' && !m.content && !m.error ? { ...m, error: 'interrupted: the page was closed before the answer finished' } : m,
  )
}

export function loadChats() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || '[]')
    if (!Array.isArray(raw)) return []
    return raw
      .filter((c) => c && c.id && Array.isArray(c.messages))
      .map((c) => ({ ...c, messages: settle(c.messages) }))
  } catch {
    return [] // private window, blocked storage or a corrupt entry: start empty
  }
}

export function saveChats(chats) {
  let list = chats.slice(0, MAX_CHATS)
  // Sources and explanations make chats large; if storage is full, drop the oldest until it fits.
  while (list.length) {
    try {
      localStorage.setItem(KEY, JSON.stringify(list))
      return true
    } catch {
      list = list.slice(0, -1)
    }
  }
  try {
    localStorage.removeItem(KEY)
  } catch {
    /* storage unavailable */
  }
  return false
}

export function relativeTime(ts, now = Date.now()) {
  const s = Math.round((now - ts) / 1000)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  if (s < 7 * 86400) return `${Math.floor(s / 86400)} d ago`
  return new Date(ts).toLocaleDateString()
}
