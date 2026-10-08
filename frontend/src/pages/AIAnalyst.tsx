import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { BrainCircuit, Send, User, Loader2, AlertCircle } from 'lucide-react'

const API_URL = 'http://127.0.0.1:8000/api'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  data?: any
}

export function AIAnalyst() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'system', content: 'CreditRisk AI Analyst is ready. Ask questions about your portfolio, model performance, or individual applicants. All answers are grounded in ML model evidence and database analytics — the AI does not independently determine credit risk.' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const suggestedQuestions = [
    "What are the main risk drivers in our model?",
    "What is the default rate for borrowers with credit scores below 600?",
    "Analyze applicant #42",
    "Compare default rates across loan purposes",
    "What does our model say about high-income borrowers?",
  ]

  const handleSend = async (question?: string) => {
    const q = question || input.trim()
    if (!q) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: q }])
    setLoading(true)

    try {
      let response = '';
      let data: any = null;

      // Send the entire conversation history to the agent (excluding the system prompt to keep it simple, or include it if desired)
      // The backend agent handles tool calling dynamically
      const historyToSent = [...messages, { role: 'user', content: q }]
        .filter(m => m.role !== 'system') // Agent backend has its own system prompt
        .map(m => ({ role: m.role, content: m.content }));

      const res = await axios.post(`${API_URL}/chat`, { messages: historyToSent });
      response = res.data.response;

      setMessages(prev => [...prev, { role: 'assistant', content: response, data }]);
    } catch (e) {
      console.error(e)
      setMessages(prev => [...prev, { role: 'assistant', content: 'I encountered an error processing your query. Please try rephrasing your question.' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)]">
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-white mb-1">AI Risk Analyst</h2>
        <p className="text-slate-400">Ask questions grounded in model evidence and portfolio data.</p>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
            {msg.role !== 'user' && (
              <div className="w-8 h-8 rounded-lg bg-primary-600/20 flex items-center justify-center shrink-0">
                <BrainCircuit size={16} className="text-primary-500" />
              </div>
            )}
            <div className={`max-w-3xl rounded-xl px-4 py-3 ${
              msg.role === 'user' ? 'bg-primary-600 text-white' :
              msg.role === 'system' ? 'bg-primary-600/5 border border-primary-600/20 text-slate-300' :
              'bg-surface border border-border text-slate-200'
            }`}>
              {msg.role === 'system' && (
                <div className="flex items-center gap-2 mb-2 text-primary-500 text-xs font-medium">
                  <AlertCircle size={14} /> System
                </div>
              )}
              <div className="text-sm whitespace-pre-wrap" dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content) }} />
            </div>
            {msg.role === 'user' && (
              <div className="w-8 h-8 rounded-lg bg-slate-700 flex items-center justify-center shrink-0">
                <User size={16} className="text-slate-300" />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary-600/20 flex items-center justify-center shrink-0">
              <BrainCircuit size={16} className="text-primary-500" />
            </div>
            <div className="bg-surface border border-border rounded-xl px-4 py-3">
              <Loader2 size={16} className="animate-spin text-primary-500" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggested Questions */}
      {messages.length <= 2 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {suggestedQuestions.map((q, i) => (
            <button key={i} onClick={() => handleSend(q)} className="text-xs bg-slate-800 border border-slate-700 text-slate-300 px-3 py-1.5 rounded-lg hover:bg-slate-700 transition-colors">
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Ask about risk drivers, applicants, portfolio segments..."
          className="input-field flex-1"
          disabled={loading}
        />
        <button onClick={() => handleSend()} disabled={loading || !input.trim()} className="btn-primary px-4">
          <Send size={18} />
        </button>
      </div>
    </div>
  )
}

function formatApplicantAnalysis(data: any): string {
  return `## Applicant Risk Assessment

| Metric | Value |
|--------|-------|
| **Probability of Default** | ${(data.probability_of_default * 100).toFixed(1)}% |
| **Risk Score** | ${data.risk_score}/100 |
| **Risk Grade** | ${data.risk_grade} (${data.risk_level}) |
| **Decision** | ${data.decision} |
| **Expected Loss** | $${data.expected_loss?.toLocaleString()} |
| **EAD** | $${data.ead?.toLocaleString()} |
| **LGD** | ${(data.lgd_used * 100).toFixed(0)}% (${data.lgd_source}) |

${data.decision_escalated ? '⚠️ **Decision was escalated by policy rules.**' : ''}

### Top SHAP Contributors
${Object.entries(data.shap_contributions || {}).slice(0, 5).map(([f, v]: [string, any]) =>
  `- **${f}**: ${v > 0 ? '↑ risk' : '↓ risk'} (${v.toFixed(4)})`
).join('\n')}

*Model version: ${data.model_version}*`
}

function formatMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/## (.*?)$/gm, '<h3 class="text-base font-semibold text-white mt-3 mb-1">$1</h3>')
    .replace(/### (.*?)$/gm, '<h4 class="text-sm font-semibold text-white mt-2 mb-1">$1</h4>')
    .replace(/\| (.*?) \|/g, (match) => `<span class="font-mono text-xs">${match}</span>`)
    .replace(/\n/g, '<br/>')
}
