'use client'

import { motion } from 'framer-motion'
import { Activity, Zap, DollarSign } from 'lucide-react'

interface TokenCounterProps {
  tokens: {
    input: number
    output: number
    total: number
  }
}

export default function TokenCounter({ tokens }: TokenCounterProps) {
  // Calculate cost (gpt-4o-mini pricing)
  const cost = (tokens.input * 0.15 + tokens.output * 0.6) / 1000000
  
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="card"
    >
      <div className="flex items-center gap-3 mb-4">
        <Activity className="w-6 h-6 text-accent" />
        <h3 className="text-lg font-semibold">Request Metrics</h3>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Input Tokens */}
        <div className="bg-zinc-800 rounded-lg p-4">
          <div className="text-xs text-zinc-400 mb-1">Input Tokens</div>
          <div className="text-2xl font-bold font-mono text-white">
            {tokens.input.toLocaleString()}
          </div>
        </div>
        
        {/* Output Tokens */}
        <div className="bg-zinc-800 rounded-lg p-4">
          <div className="text-xs text-zinc-400 mb-1">Output Tokens</div>
          <div className="text-2xl font-bold font-mono text-white">
            {tokens.output.toLocaleString()}
          </div>
        </div>
        
        {/* Total Tokens */}
        <div className="bg-zinc-800 rounded-lg p-4 border-2 border-accent">
          <div className="text-xs text-zinc-400 mb-1 flex items-center gap-1">
            <Zap className="w-3 h-3" />
            Total Tokens
          </div>
          <div className="text-2xl font-bold font-mono text-accent">
            {tokens.total.toLocaleString()}
          </div>
        </div>
        
        {/* Cost */}
        <div className="bg-zinc-800 rounded-lg p-4">
          <div className="text-xs text-zinc-400 mb-1 flex items-center gap-1">
            <DollarSign className="w-3 h-3" />
            Estimated Cost
          </div>
          <div className="text-2xl font-bold font-mono text-green-400">
            ${cost.toFixed(4)}
          </div>
        </div>
      </div>
      
      {/* Additional Info */}
      <div className="mt-4 pt-4 border-t border-zinc-800 text-sm text-zinc-400">
        <div className="flex justify-between">
          <span>Model: gpt-4o-mini</span>
          <span>Pricing: $0.15/M input, $0.60/M output</span>
        </div>
      </div>
    </motion.div>
  )
}
