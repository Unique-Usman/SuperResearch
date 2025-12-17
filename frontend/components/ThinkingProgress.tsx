'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { Brain, CheckCircle2, Loader2, AlertCircle } from 'lucide-react'

interface ProgressMessage {
  stage: string
  message: string
  timestamp: number
}

interface ThinkingProgressProps {
  messages: ProgressMessage[]
  isComplete: boolean
  hasError: boolean
}

export default function ThinkingProgress({ messages, isComplete, hasError }: ThinkingProgressProps) {
  if (messages.length === 0 && !isComplete && !hasError) return null

  const getStageIcon = (stage: string) => {
    const icons: Record<string, JSX.Element> = {
      init: <Brain className="w-4 h-4 text-accent animate-pulse" />,
      refining: <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />,
      searching: <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />,
      embedding: <Loader2 className="w-4 h-4 text-yellow-400 animate-spin" />,
      retrieving: <Loader2 className="w-4 h-4 text-green-400 animate-spin" />,
      generating: <Loader2 className="w-4 h-4 text-accent animate-spin" />,
      finalizing: <CheckCircle2 className="w-4 h-4 text-green-400" />,
      complete: <CheckCircle2 className="w-4 h-4 text-green-400" />
    }
    return icons[stage] || <Loader2 className="w-4 h-4 text-zinc-400 animate-spin" />
  }

  const getStageColor = (stage: string) => {
    const colors: Record<string, string> = {
      init: 'text-accent',
      refining: 'text-blue-400',
      searching: 'text-purple-400',
      embedding: 'text-yellow-400',
      retrieving: 'text-green-400',
      generating: 'text-accent',
      finalizing: 'text-green-400',
      complete: 'text-green-400'
    }
    return colors[stage] || 'text-zinc-400'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="card mb-6"
    >
      <div className="flex items-center gap-3 mb-4">
        <Brain className="w-6 h-6 text-accent" />
        <h3 className="text-lg font-semibold">
          {isComplete ? 'Research Complete' : hasError ? 'Error Occurred' : 'Thinking...'}
        </h3>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        <AnimatePresence mode="popLayout">
          {messages.map((msg, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.05 * index }}
              className="flex items-start gap-3 p-3 bg-zinc-800 rounded-lg"
            >
              <div className="mt-0.5">
                {getStageIcon(msg.stage)}
              </div>
              <div className="flex-1">
                <p className={`text-sm ${getStageColor(msg.stage)}`}>
                  {msg.message}
                </p>
                <p className="text-xs text-zinc-500 mt-1">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {hasError && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-3 p-4 bg-red-900/20 border border-red-900/50 rounded-lg"
          >
            <AlertCircle className="w-5 h-5 text-red-400" />
            <p className="text-sm text-red-400">
              Research failed. Please try again.
            </p>
          </motion.div>
        )}

        {isComplete && !hasError && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-3 p-4 bg-green-900/20 border border-green-900/50 rounded-lg"
          >
            <CheckCircle2 className="w-5 h-5 text-green-400" />
            <p className="text-sm text-green-400">
              Research completed successfully!
            </p>
          </motion.div>
        )}
      </div>
    </motion.div>
  )
}
