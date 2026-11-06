const PROVIDER_NAMES = { gemini: 'Gemini', openai: 'OpenAI', openrouter: 'OpenRouter' }

/** Short name for a model: "gemini:gemini-2.5-flash" -> "Gemini 2.5 flash". */
export function modelLabel(id) {
  if (id === 'adapter') return 'Local (LoRA)'
  if (id === 'local-heavy') return 'Local heavy (7B)'
  if (id === 'groq') return 'Base (Groq)'
  const [provider, model = ''] = id.split(/:(.*)/s)
  if (provider === 'openai') return model.replace(/^gpt/, 'GPT').replace(/-(mini|nano|pro)$/, ' $1') // "GPT-5.4 mini"
  const name = model.split('/').pop().replace(/^gemini-/, '').replace(/-/g, ' ')
  return `${PROVIDER_NAMES[provider] || provider} ${name}`.trim()
}

/** Where a model runs, for the line under its name. */
export const modelNote = (m) => (m.id === 'adapter' || m.id === 'local-heavy' ? 'Runs on this computer' : m.id.startsWith('openai:') ? 'Paid per use' : 'Free hosted model')
