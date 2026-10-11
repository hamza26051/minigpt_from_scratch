/** Characters the model added after the prompt, or null if the response does not start with the prompt. */
export function countNewCharacters(prompt: string, text: string): number | null {
  return prompt.length > 0 && text.startsWith(prompt) ? text.length - prompt.length : null
}

export function formatSeconds(ms: number, digits = 2): string {
  return `${(ms / 1000).toFixed(digits)} s`
}

export function formatClockTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

/** One-line preview of a prompt for list items. */
export function summarizePrompt(prompt: string, maxLength = 40): string {
  const flat = prompt.replace(/\s+/g, ' ').trim()
  if (flat.length === 0) return '(whitespace only)'
  return flat.length > maxLength ? `${flat.slice(0, maxLength)}…` : flat
}