'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Loader2 } from 'lucide-react'

const CONTEXTS = [
  { value: 'general', label: 'General', desc: 'Flexible research' },
  { value: 'academic', label: 'Academic', desc: 'Scholarly analysis' },
  { value: 'business', label: 'Business', desc: 'Market & strategy' },
  { value: 'product', label: 'Product', desc: 'Features & reviews' },
  { value: 'investment', label: 'Investment', desc: 'Financial analysis' },
  { value: 'technical', label: 'Technical', desc: 'Implementation docs' }
]

interface ResearchFormProps {
  onSubmit: (data: { question: string; context: string }) => void
  loading: boolean
}

export default function ResearchForm({ onSubmit, loading }: ResearchFormProps) {
  const [question, setQuestion] = useState('')
  const [context, setContext] = useState('general')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (question.trim()) {
      onSubmit({ question, context })
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Question Input */}
      <div>
        <label className="block text-sm font-medium mb-2">
          Research Question
        </label>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What would you like to research?"
          className="w-full px-4 py-3 bg-zinc-900 border border-zinc-700 rounded-lg focus:border-accent focus:ring-1 focus:ring-accent outline-none resize-none h-32"
          required
          disabled={loading}
        />
      </div>

      {/* Context Selector */}
      <div>
        <label className="block text-sm font-medium mb-3">
          Research Context
        </label>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {CONTEXTS.map((ctx) => (
            <motion.button
              key={ctx.value}
              type="button"
              onClick={() => setContext(ctx.value)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              disabled={loading}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                context === ctx.value
                  ? 'border-accent bg-accent/10'
                  : 'border-zinc-800 hover:border-zinc-700'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <div className="font-semibold mb-1">{ctx.label}</div>
              <div className="text-sm text-zinc-400">{ctx.desc}</div>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={loading || !question.trim()}
        className="w-full btn-accent py-4 text-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Researching...
          </>
        ) : (
          'Start Research'
        )}
      </button>
    </form>
  )
}
