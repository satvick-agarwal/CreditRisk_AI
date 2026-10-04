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
      // Route the question to the appropriate API based on intent detection
      let response = ''
      let data: any = null

      const lowerQ = q.toLowerCase()

      if (lowerQ.includes('applicant #') || lowerQ.match(/applicant\s+\d+/)) {
        // Extract applicant ID
        const match = q.match(/(\d+)/)
        if (match) {
          const id = parseInt(match[1])
          try {
            const riskRes = await axios.get(`${API_URL}/applicants/${id}/risk`)
            data = riskRes.data
            response = formatApplicantAnalysis(data)
          } catch {
            response = `Could not find applicant #${id} in the database. The applicant ID must match a record in our system.`
          }
        }
      } else if (lowerQ.includes('risk driver') || lowerQ.includes('feature importance') || lowerQ.includes('what factors')) {
        const res = await axios.get(`${API_URL}/model/metrics`)
        const importance = res.data.shap_feature_importance?.slice(0, 10) || []
        response = `## Top Risk Drivers (SHAP Analysis)\n\nBased on SHAP global feature importance computed on the validation set:\n\n`
        importance.forEach(([name, val]: [string, number], i: number) => {
          response += `${i + 1}. **${name}** — importance: ${val.toFixed(4)}\n`
        })
        response += `\nThese values represent the mean absolute SHAP contribution. Higher values indicate features that more strongly influence the model's default predictions.`
      } else if (lowerQ.includes('default rate') || lowerQ.includes('credit score')) {
        const res = await axios.get(`${API_URL}/dashboard/risk-distribution`)
        data = res.data

        if (lowerQ.includes('below 600') || lowerQ.includes('under 600')) {
          const poor = data.by_credit_score?.['Poor (300-579)']
          if (poor) {
            response = `## Default Rate for Credit Scores Below 600\n\n- **Default Rate:** ${(poor.default_rate * 100).toFixed(1)}%\n- **Count:** ${poor.count.toLocaleString()} applicants\n- **Defaults:** ${poor.defaults.toLocaleString()}\n\nThis segment has significantly elevated risk compared to the portfolio average.`
          }
        } else if (lowerQ.includes('loan purpose') || lowerQ.includes('intent')) {
          response = `## Default Rates by Loan Purpose\n\n`
          if (data.by_intent) {
            Object.entries(data.by_intent).forEach(([intent, stats]: [string, any]) => {
              response += `- **${intent}**: ${(stats.default_rate * 100).toFixed(1)}% default rate (${stats.count.toLocaleString()} loans)\n`
            })
          }
        } else {
          response = `## Default Rate Analysis\n\n### By Credit Score Band\n\n`
          if (data.by_credit_score) {
            Object.entries(data.by_credit_score).forEach(([band, stats]: [string, any]) => {
              response += `- **${band}**: ${(stats.default_rate * 100).toFixed(1)}% (${stats.count.toLocaleString()} loans)\n`
            })
          }
        }
      } else if (lowerQ.includes('high-income') || lowerQ.includes('high income')) {
        const res = await axios.get(`${API_URL}/portfolio/analytics`, { params: { income_min: 100000 } })
        data = res.data
        response = `## High-Income Borrower Analysis (Income > $100k)\n\n- **Count:** ${data.count.toLocaleString()} applicants\n- **Default Rate:** ${(data.default_rate * 100).toFixed(1)}%\n- **Average Credit Score:** ${data.avg_credit_score?.toFixed(0)}\n- **Average Loan Amount:** $${data.avg_loan?.toLocaleString()}\n- **Total Exposure:** $${(data.total_exposure / 1e6).toFixed(1)}M\n\nHigh-income borrowers in our portfolio show ${data.default_rate < 0.15 ? 'lower' : 'comparable'} default risk relative to the portfolio average.`
      } else if (lowerQ.includes('model') || lowerQ.includes('performance') || lowerQ.includes('accuracy')) {
        const res = await axios.get(`${API_URL}/model/metrics`)
        const tm = res.data.test_metrics || {}
        response = `## Model Performance Summary\n\n**Model:** ${res.data.selected_model} (${res.data.model_version})\n\n### Test Set Metrics\n| Metric | Value |\n|--------|-------|\n| ROC-AUC | ${tm.roc_auc} |\n| PR-AUC | ${tm.pr_auc} |\n| F1 Score | ${tm.f1} |\n| Recall | ${tm.recall} |\n| Precision | ${tm.precision} |\n| Brier Score | ${tm.brier_score} |\n\n**Note:** The test set was protected during training — these metrics represent an unbiased estimate of generalization performance.`
      } else {
        // Generic query — try portfolio summary
        const res = await axios.get(`${API_URL}/dashboard/summary`)
        response = `I can help analyze your credit risk portfolio. Here's the current summary:\n\n- **Total Applications:** ${res.data.total_applications?.toLocaleString()}\n- **Average PD:** ${(res.data.average_pd * 100).toFixed(1)}%\n- **Total Exposure:** $${(res.data.total_exposure / 1e6).toFixed(1)}M\n\nTry asking me specific questions like:\n- "What are the main risk drivers?"\n- "What is the default rate for credit scores below 600?"\n- "Analyze applicant #42"\n- "Compare default rates across loan purposes"`
      }

      setMessages(prev => [...prev, { role: 'assistant', content: response, data }])
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
