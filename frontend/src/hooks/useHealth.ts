import { useCallback, useEffect, useState } from 'react'
import { ApiError, getHealth } from '../services/api'
import type { HealthResponse } from '../types/api'

export type ConnectionStatus = 'checking' | 'connected' | 'unreachable'

const POLL_INTERVAL_MS = 60_000
const RETRY_DELAY_MS = 4_000

/**
 * Verifies the backend with real GET /health requests.
 * Status is "connected" only after a successful response with status "ok".
 * A single failed check is retried once before the status becomes "unreachable".
 */
export function useHealth() {
  const [status, setStatus] = useState<ConnectionStatus>('checking')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [checkId, setCheckId] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let failures = 0
    let retryTimer: number | undefined

    async function check() {
      window.clearTimeout(retryTimer)
      try {
        const result = await getHealth(controller.signal)
        if (result.status !== 'ok') {
          throw new ApiError('malformed', `The health endpoint reported status "${result.status}" instead of "ok".`)
        }
        if (controller.signal.aborted) return
        failures = 0
        setHealth(result)
        setError(null)
        setStatus('connected')
      } catch (err) {
        if (controller.signal.aborted) return
        failures += 1
        setError(err instanceof Error ? err.message : 'Unknown error while checking the API.')
        if (failures < 2) {
          // First failure: keep the current status and retry shortly (covers cold starts and blips).
          retryTimer = window.setTimeout(() => void check(), RETRY_DELAY_MS)
          return
        }
        setStatus('unreachable')
      }
    }

    function handleVisibility() {
      if (!document.hidden) void check()
    }

    void check()
    const interval = window.setInterval(() => {
      if (!document.hidden) void check()
    }, POLL_INTERVAL_MS)
    document.addEventListener('visibilitychange', handleVisibility)

    return () => {
      controller.abort()
      window.clearInterval(interval)
      window.clearTimeout(retryTimer)
      document.removeEventListener('visibilitychange', handleVisibility)
    }
  }, [checkId])

  const recheck = useCallback(() => {
    setStatus('checking')
    setCheckId((id) => id + 1)
  }, [])

  return {
    status,
    health: status === 'connected' ? health : null,
    error,
    recheck,
  }
}