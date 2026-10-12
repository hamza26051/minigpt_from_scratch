const rawBaseUrl = import.meta.env.VITE_API_BASE_URL

/** Backend base URL without a trailing slash. Defaults to the Vite dev proxy. */
export const API_BASE_URL: string = (
  rawBaseUrl && rawBaseUrl.trim() !== '' ? rawBaseUrl.trim() : '/api'
).replace(/\/+$/, '')