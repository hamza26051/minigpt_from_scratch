import { useId, useLayoutEffect, useRef, useState } from 'react'
import { AlertTriangle, ChevronDown, ChevronUp, Lightbulb, Loader2, Send } from 'lucide-react'
import { CONTEXT_LENGTH, EXAMPLE_PROMPTS } from '../../lib/limits'
import { CONTENT_COLUMN } from '../../lib/ui'
import { Button } from '../ui/Button'

type ComposerProps = {
  prompt: string
  onPromptChange: (value: string) => void
  onSubmit: () => void
  loading: boolean
  unsupportedCharacters: string[]
}

const MAX_TEXTAREA_HEIGHT = 168

function visible(ch: string): string {
  // Makes control characters such as tabs readable in the warning.
  return JSON.stringify(ch).slice(1, -1)
}

/**
 * One surface for the whole interaction: prompt, example prompts and the Generate button share a
 * single rounded container, with supporting information underneath.
 */
export function Composer({
  prompt,
  onPromptChange,
  onSubmit,
  loading,
  unsupportedCharacters,
}: ComposerProps) {
  const textareaId = useId()
  const helpId = useId()
  const examplesId = useId()
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [examplesOpen, setExamplesOpen] = useState(true)

  const isEmpty = prompt.length === 0
  const overContext = prompt.length > CONTEXT_LENGTH

  // Grow the textarea with its content, up to a maximum height.
  useLayoutEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`
  }, [prompt])

  return (
    <div className="shrink-0 pb-3 pt-1 sm:pb-4">
      <div className={CONTENT_COLUMN}>
        <div className="rounded-2xl border border-line/70 bg-panel shadow-lg shadow-black/10 transition-colors focus-within:border-accent/60 focus-within:ring-2 focus-within:ring-accent/15">
          <label htmlFor={textareaId} className="sr-only">
            Prompt text
          </label>
          <textarea
            ref={textareaRef}
            id={textareaId}
            value={prompt}
            rows={1}
            spellCheck={false}
            aria-describedby={helpId}
            placeholder="Enter a prompt, for example ROMEO:"
            onChange={(e) => onPromptChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault()
                onSubmit()
              }
            }}
            className="no-scrollbar block w-full resize-none bg-transparent px-4 pb-2 pt-3.5 font-mono text-sm leading-6 text-ink outline-none placeholder:text-faint"
          />

          <div aria-live="polite" className="empty:hidden">
            {unsupportedCharacters.length > 0 && (
              <p className="flex gap-2 px-4 pb-2 text-xs text-amber-300">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>
                  Not in the model&apos;s 65-character vocabulary:{' '}
                  <span className="font-mono">{unsupportedCharacters.map(visible).join(' ')}</span>. The
                  backend may reject this prompt.
                </span>
              </p>
            )}
          </div>

          <div className="flex items-center gap-2 border-t border-line/40 px-2.5 py-2">
            <button
              type="button"
              aria-expanded={examplesOpen}
              aria-controls={examplesId}
              onClick={() => setExamplesOpen((open) => !open)}
              title="Show or hide example prompts"
              className="inline-flex h-8 shrink-0 items-center gap-1 rounded-lg px-2 text-xs text-muted transition-colors hover:bg-raised hover:text-ink"
            >
              <Lightbulb className="h-3.5 w-3.5" aria-hidden="true" />
              <span className="sr-only">Example prompts</span>
              {examplesOpen ? (
                <ChevronUp className="h-3.5 w-3.5" aria-hidden="true" />
              ) : (
                <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
              )}
            </button>

            <div
              id={examplesId}
              role="group"
              aria-label="Example prompts"
              hidden={!examplesOpen}
              className="no-scrollbar flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto [mask-image:linear-gradient(to_right,black_calc(100%-1.5rem),transparent)]"
            >
              {EXAMPLE_PROMPTS.map((example) => (
                <button
                  key={example}
                  type="button"
                  disabled={loading}
                  onClick={() => {
                    onPromptChange(example)
                    textareaRef.current?.focus()
                  }}
                  className="inline-flex h-8 shrink-0 items-center rounded-lg bg-raised/60 px-2.5 font-mono text-xs text-muted transition-colors hover:bg-raised hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {example}
                </button>
              ))}
            </div>
            {!examplesOpen && <div className="flex-1" />}

            <Button
              variant="primary"
              onClick={onSubmit}
              disabled={isEmpty}
              loading={loading}
              aria-label={loading ? 'Generating text' : 'Generate text'}
              className="px-3 sm:px-4"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              ) : (
                <Send className="h-4 w-4" aria-hidden="true" />
              )}
              <span className="hidden sm:inline">{loading ? 'Generating…' : 'Generate Text'}</span>
            </Button>
          </div>
        </div>

        <div id={helpId} className="mt-2 flex flex-wrap justify-between gap-x-4 px-1 text-xs text-faint">
          <span>Press Enter to send, Shift+Enter for new line.</span>
          <span>
            {prompt.length} {prompt.length === 1 ? 'character' : 'characters'}.{' '}
            {overContext
              ? `Only the last ${CONTEXT_LENGTH} are used as context.`
              : `Context window: ${CONTEXT_LENGTH}.`}
          </span>
        </div>
      </div>
    </div>
  )
}