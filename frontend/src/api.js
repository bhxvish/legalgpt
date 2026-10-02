export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

/**
 * POST /api/chat and dispatch each Server-Sent Event as it arrives.
 * (EventSource only supports GET, so the stream is read from fetch directly.)
 *
 * @param {{question: string, mode: 'legal'|'general', history: {role: string, content: string}[]}} body
 * @param {(event: string, data: any) => void} onEvent
 * @param {AbortSignal} [signal]
 */
export async function streamChat(body, onEvent, signal) {
  const res = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status}${detail ? `: ${detail.slice(0, 200)}` : ''}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let event = 'message'
      const data = []
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''))
      }
      if (data.length) onEvent(event, JSON.parse(data.join('\n')))
    }
  }
}

async function json(res) {
  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      detail = res.statusText
    }
    throw new Error(`HTTP ${res.status}: ${detail}`)
  }
  return res.json()
}

const enc = encodeURIComponent

export const annotationApi = {
  scheme: () => fetch(`${API_BASE_URL}/api/annotation/scheme`).then(json),
  cases: (annotator) => fetch(`${API_BASE_URL}/api/annotation/cases?annotator=${enc(annotator)}`).then(json),
  getCase: (caseId, annotator) =>
    fetch(`${API_BASE_URL}/api/annotation/cases/${enc(caseId)}?annotator=${enc(annotator)}`).then(json),
  saveLabels: (caseId, annotator, labels) =>
    fetch(`${API_BASE_URL}/api/annotation/cases/${enc(caseId)}/labels`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ annotator, labels }),
    }).then(json),
}

export const reviewApi = {
  cases: () => fetch(`${API_BASE_URL}/api/review/cases`).then(json),
  getCase: (caseId) => fetch(`${API_BASE_URL}/api/review/cases/${enc(caseId)}`).then(json),
  setLabel: (caseId, sentenceId, reviewer, label) =>
    fetch(`${API_BASE_URL}/api/review/cases/${enc(caseId)}/items/${enc(sentenceId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer, label }),
    }).then(json),
  signOff: (caseId, reviewer) =>
    fetch(`${API_BASE_URL}/api/review/cases/${enc(caseId)}/sign-off`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer }),
    }).then(json),
}

export const verifyApi = {
  sections: () => fetch(`${API_BASE_URL}/api/verify/sections`).then(json),
  verify: (caseText, citedSection) =>
    fetch(`${API_BASE_URL}/api/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ case_text: caseText, cited_section: citedSection }),
    }).then(json),
}
