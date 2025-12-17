'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Download, MessageSquare, RefreshCw, Loader2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

interface ResultsDisplayProps {
  results: any
  onNewResearch: () => void
  onTokenUpdate: (tokens: any) => void
}

export default function ResultsDisplay({ results, onNewResearch, onTokenUpdate }: ResultsDisplayProps) {
  const [feedback, setFeedback] = useState('')
  const [submittingFeedback, setSubmittingFeedback] = useState(false)
  const [downloadingPDF, setDownloadingPDF] = useState(false)
  const [currentReport, setCurrentReport] = useState(results.report)

  const handleFeedback = async () => {
    if (!feedback.trim()) return
    
    setSubmittingFeedback(true)
    try {
      const response = await fetch('http://localhost:8000/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: results.session_id,
          feedback: feedback
        })
      })
      
      if (!response.ok) {
        throw new Error('Feedback failed')
      }
      
      const result = await response.json()
      setCurrentReport(result.report)
      onTokenUpdate(result.tokens_used)
      setFeedback('')
      alert(`Report updated successfully! ${result.needs_new_info ? '(New information added)' : '(Used existing information)'}`)
    } catch (error) {
      console.error('Feedback failed:', error)
      alert('Failed to process feedback. Please try again.')
    } finally {
      setSubmittingFeedback(false)
    }
  }

  const handleDownloadPDF = async () => {
    setDownloadingPDF(true)
    try {
      const response = await fetch('http://localhost:8000/pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: results.session_id })
      })
      
      if (!response.ok) {
        throw new Error('PDF generation failed')
      }
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `research_report_${results.context}.pdf`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('PDF download failed:', error)
      alert('Failed to download PDF. Please try again.')
    } finally {
      setDownloadingPDF(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Actions */}
      <div className="flex flex-wrap gap-4">
        <button 
          onClick={handleDownloadPDF} 
          disabled={downloadingPDF} 
          className="btn-accent flex items-center gap-2"
        >
          {downloadingPDF ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Generating PDF...
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              Download PDF
            </>
          )}
        </button>
        <button 
          onClick={onNewResearch} 
          className="px-6 py-3 bg-zinc-800 hover:bg-zinc-700 rounded-lg font-semibold transition-all flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          New Research
        </button>
      </div>

      {/* Report */}
      <div className="card">
        <h2 className="text-2xl font-bold mb-4">Research Report</h2>
        <div className="prose prose-invert max-w-none">
          <ReactMarkdown>{currentReport}</ReactMarkdown>
        </div>
      </div>

      {/* Feedback */}
      <div className="card">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-accent" />
          Provide Feedback
        </h3>
        <p className="text-sm text-zinc-400 mb-4">
          Our AI will intelligently determine if your feedback requires new information or just rephrasing.
        </p>
        <textarea
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          placeholder="What would you like to improve or add to this report? (e.g., 'Add more about 2024 developments' or 'Make the conclusion shorter')"
          className="w-full px-4 py-3 bg-zinc-800 border border-zinc-700 rounded-lg focus:border-accent focus:ring-1 focus:ring-accent outline-none resize-none h-24 mb-4"
          disabled={submittingFeedback}
        />
        <button 
          onClick={handleFeedback}
          disabled={submittingFeedback || !feedback.trim()}
          className="btn-accent disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {submittingFeedback ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Processing Feedback...
            </>
          ) : (
            'Submit Feedback'
          )}
        </button>
      </div>

      {/* Citations */}
      {results.citations && results.citations.length > 0 && (
        <div className="card">
          <h3 className="text-xl font-bold mb-4">Sources ({results.citations.length})</h3>
          <div className="space-y-3">
            {results.citations.slice(0, 10).map((citation: any, i: number) => (
              <div key={i} className="text-sm">
                <span className="text-accent font-semibold">[{citation.id}]</span>{' '}
                <span className="text-zinc-300">{citation.title}</span>
                <br />
                <a 
                  href={citation.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-zinc-500 hover:text-accent transition-colors text-xs"
                >
                  {citation.url}
                </a>
                <span className="text-zinc-600 text-xs ml-2">({citation.source})</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
