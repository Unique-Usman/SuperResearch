'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ResearchForm from '@/components/ResearchForm'
import ResultsDisplay from '@/components/ResultsDisplay'
import TokenCounter from '@/components/TokenCounter'
import ThinkingProgress from '@/components/ThinkingProgress'

interface ProgressMessage {
  stage: string
  message: string
  timestamp: number
}

export default function ResearchPage() {
  const [results, setResults] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [tokens, setTokens] = useState<any>(null)
  const [progressMessages, setProgressMessages] = useState<ProgressMessage[]>([])
  const [isComplete, setIsComplete] = useState(false)
  const [hasError, setHasError] = useState(false)

  const handleResearch = async (data: any) => {
    setLoading(true)
    setProgressMessages([])
    setIsComplete(false)
    setHasError(false)
    setResults(null)
    
    try {
      // Use fetch with streaming for SSE
      const response = await fetch('http://localhost:8000/research/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      })
      
      if (!response.ok) {
        throw new Error('Research failed')
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body')
      }

      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        // Decode the chunk
        buffer += decoder.decode(value, { stream: true })

        // Process complete messages (SSE format: "data: {...}\n\n")
        const messages = buffer.split('\n\n')
        buffer = messages.pop() || '' // Keep incomplete message in buffer

        for (const message of messages) {
          if (!message.trim() || !message.startsWith('data: ')) continue

          try {
            const eventData = JSON.parse(message.slice(6)) // Remove "data: " prefix

            if (eventData.type === 'progress') {
              // Add progress message
              setProgressMessages(prev => [
                ...prev,
                {
                  stage: eventData.stage,
                  message: eventData.message,
                  timestamp: Date.now()
                }
              ])
            } else if (eventData.type === 'complete') {
              // Research complete
              setIsComplete(true)
              setLoading(false)
              setResults(eventData)
              setTokens(eventData.tokens_used)
            } else if (eventData.type === 'error') {
              throw new Error(eventData.message)
            }
          } catch (parseError) {
            console.error('Failed to parse SSE message:', parseError)
          }
        }
      }
    } catch (error) {
      console.error('Research failed:', error)
      setHasError(true)
      setLoading(false)
      alert('Research failed. Please check that the backend is running and try again.')
    }
  }

  return (
    <div className="container mx-auto px-6 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto"
      >
        <h1 className="text-4xl font-bold mb-2">Research Agent</h1>
        <p className="text-zinc-400 mb-8">
          AI-powered research with parallel search and RAG
        </p>

        {tokens && !loading && (
          <div className="mb-6">
            <TokenCounter tokens={tokens} />
          </div>
        )}

        {!results ? (
          <div className="space-y-6">
            <ResearchForm onSubmit={handleResearch} loading={loading} />
            
            {/* Show thinking progress while loading */}
            <AnimatePresence>
              {(loading || progressMessages.length > 0) && (
                <ThinkingProgress
                  messages={progressMessages}
                  isComplete={isComplete}
                  hasError={hasError}
                />
              )}
            </AnimatePresence>
          </div>
        ) : (
          <ResultsDisplay
            results={results}
            onNewResearch={() => {
              setResults(null)
              setTokens(null)
              setProgressMessages([])
              setIsComplete(false)
              setHasError(false)
            }}
            onTokenUpdate={setTokens}
          />
        )}
      </motion.div>
    </div>
  )
}
