import { Playground } from './components/playground/Playground'
import { useHealth } from './hooks/useHealth'

export default function App() {
  const { status, health, error, recheck } = useHealth()

  return (
    <Playground
      connectionStatus={status}
      health={health}
      connectionError={error}
      onRecheck={recheck}
    />
  )
}