'use client'

import { motion } from 'framer-motion'
import { Sparkles, Search, FileText, Zap, Database, TrendingUp } from 'lucide-react'
import Link from 'next/link'

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="container mx-auto px-6 py-20">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center max-w-4xl mx-auto"
        >
          <motion.div 
            className="inline-block mb-6"
            animate={{ rotate: [0, 10, 0] }}
            transition={{ repeat: Infinity, duration: 3 }}
          >
            <Sparkles className="w-16 h-16 text-accent" />
          </motion.div>
          
          <h1 className="text-6xl font-bold mb-6">
            AI-Powered Research
            <span className="block text-accent mt-2">Made Simple</span>
          </h1>
          
          <p className="text-xl text-zinc-400 mb-10 leading-relaxed">
            Advanced research agent with parallel multi-source search, RAG,
            and context-aware report generation. Get comprehensive insights
            in minutes, not hours.
          </p>
          
          <Link href="/research">
            <button className="btn-accent text-lg px-10 py-4 shadow-lg shadow-accent/20">
              Start Researching →
            </button>
          </Link>
        </motion.div>
      </section>

      {/* Features */}
      <section className="container mx-auto px-6 py-20">
        <h2 className="text-4xl font-bold text-center mb-16">
          Powerful Features
        </h2>
        
        <div className="grid md:grid-cols-3 gap-8">
          {[
            {
              icon: Search,
              title: '6 Search Sources',
              description: 'Parallel search across Tavily, ArXiv, PubMed, Wikipedia, Semantic Scholar, and News APIs'
            },
            {
              icon: Database,
              title: 'RAG System',
              description: 'Semantic embeddings with FAISS for accurate, source-backed responses'
            },
            {
              icon: FileText,
              title: 'Context-Aware',
              description: 'Academic, Business, Product, Investment, Technical, and General templates'
            },
            {
              icon: Zap,
              title: 'Smart Feedback',
              description: 'AI detects if feedback needs new info or just rephrasing'
            },
            {
              icon: TrendingUp,
              title: 'Cost Efficient',
              description: '~$0.05 per report with real-time token tracking'
            },
            {
              icon: FileText,
              title: 'PDF Export',
              description: 'Professional PDF reports with citations'
            }
          ].map((feature, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className="card hover:border-accent/50 transition-all cursor-pointer group"
            >
              <feature.icon className="w-12 h-12 text-accent mb-4 group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
              <p className="text-zinc-400">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section className="container mx-auto px-6 py-20">
        <h2 className="text-4xl font-bold text-center mb-16">
          How It Works
        </h2>
        
        <div className="max-w-3xl mx-auto space-y-8">
          {[
            'Enter your research question and select context type',
            'AI refines your question into 4 targeted queries',
            'Parallel search across 4 sources simultaneously',
            'RAG system processes and ranks relevant information',
            'Template-based generation creates structured report',
            'Download PDF or provide feedback for refinement'
          ].map((step, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex items-start gap-4"
            >
              <div className="w-10 h-10 rounded-full bg-accent flex items-center justify-center font-bold flex-shrink-0">
                {i + 1}
              </div>
              <p className="text-lg text-zinc-300 pt-2">{step}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="container mx-auto px-6 py-20 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          className="card max-w-2xl mx-auto"
        >
          <h2 className="text-3xl font-bold mb-4">Ready to get started?</h2>
          <p className="text-zinc-400 mb-8">
            Join researchers, analysts, and decision-makers using AI-powered research.
          </p>
          <Link href="/research">
            <button className="btn-accent text-lg">
              Start Your First Research
            </button>
          </Link>
        </motion.div>
      </section>
    </div>
  )
}
