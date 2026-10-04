import React, { useState } from 'react'
import axios from 'axios'
import { ArrowRight, Activity, TrendingDown, TrendingUp, RefreshCw } from 'lucide-react'

const API_URL = 'http://127.0.0.1:8000/api'

export function Simulator() {
  const [baseForm, setBaseForm] = useState({
    person_age: 28, person_education: "Bachelor", person_income: 65000,
    person_emp_exp: 5, person_home_ownership: "RENT", loan_amnt: 15000,
    loan_intent: "PERSONAL", loan_int_rate: 11.5, cb_person_cred_hist_length: 4,
    credit_score: 680, previous_loan_defaults_on_file: "No"
  })

  const [simForm, setSimForm] = useState({ ...baseForm })
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const handleBaseChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    const isNum = ['person_age','person_income','person_emp_exp','loan_amnt','loan_int_rate','cb_person_cred_hist_length','credit_score'].includes(name)
    const newVal = isNum ? Number(value) : value
    setBaseForm(prev => ({ ...prev, [name]: newVal }))
    // Also update sim form to stay in sync unless already modified
    setSimForm(prev => ({ ...prev, [name]: newVal }))
  }

  const handleSimChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    const isNum = ['person_age','person_income','person_emp_exp','loan_amnt','loan_int_rate','cb_person_cred_hist_length','credit_score'].includes(name)
    setSimForm(prev => ({ ...prev, [name]: isNum ? Number(value) : value }))
  }

  const runSimulation = async () => {
    setLoading(true)
    try {
      const res = await axios.post(`${API_URL}/simulate`, {
        original_application: baseForm,
        modified_application: simForm
      })
      setResult(res.data)
    } catch (e) {
      console.error(e)
      alert("Simulation failed.")
    } finally {
      setLoading(false)
    }
  }

  const resetSim = () => { setSimForm({ ...baseForm }); setResult(null) }

  const modifiableFields = [
    { key: 'person_income', label: 'Income ($)', type: 'number' },
    { key: 'loan_amnt', label: 'Loan Amount ($)', type: 'number' },
    { key: 'credit_score', label: 'Credit Score', type: 'number' },
    { key: 'loan_int_rate', label: 'Interest Rate (%)', type: 'number' },
    { key: 'person_emp_exp', label: 'Employment Exp (Yrs)', type: 'number' },
    { key: 'previous_loan_defaults_on_file', label: 'Previous Default', type: 'select', options: ['No', 'Yes'] },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Risk Simulator</h2>
        <p className="text-slate-400">Explore how changing applicant characteristics affects the risk decision.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Base Scenario */}
        <div className="card border-primary-600/30">
          <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-primary-500"></div>
            Base Scenario
          </h3>
          <div className="grid grid-cols-2 gap-3">
            {modifiableFields.map(f => (
              <div key={`base-${f.key}`}>
                <label className="label">{f.label}</label>
                {f.type === 'select' ? (
                  <select name={f.key} value={(baseForm as any)[f.key]} onChange={handleBaseChange} className="input-field">
                    {f.options?.map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : (
                  <input type="number" step="any" name={f.key} value={(baseForm as any)[f.key]} onChange={handleBaseChange} className="input-field" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Modified Scenario */}
        <div className="card border-risk-moderate/30">
          <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-risk-moderate"></div>
            What-If Scenario
            <button onClick={resetSim} className="ml-auto text-xs text-slate-400 hover:text-white flex items-center gap-1"><RefreshCw size={12} /> Reset</button>
          </h3>
          <div className="grid grid-cols-2 gap-3">
            {modifiableFields.map(f => {
              const changed = (simForm as any)[f.key] !== (baseForm as any)[f.key]
              return (
                <div key={`sim-${f.key}`}>
                  <label className={`label ${changed ? 'text-risk-moderate' : ''}`}>{f.label} {changed && '●'}</label>
                  {f.type === 'select' ? (
                    <select name={f.key} value={(simForm as any)[f.key]} onChange={handleSimChange} className={`input-field ${changed ? 'border-risk-moderate/50' : ''}`}>
                      {f.options?.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                  ) : (
                    <input type="number" step="any" name={f.key} value={(simForm as any)[f.key]} onChange={handleSimChange} className={`input-field ${changed ? 'border-risk-moderate/50' : ''}`} />
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div className="flex justify-center">
        <button onClick={runSimulation} disabled={loading} className="btn-primary flex items-center gap-2 px-8 py-3 text-lg">
          {loading ? <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin"></div> : <Activity size={20} />}
          Run Simulation
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-4">
          <h3 className="text-lg font-medium text-white">Simulation Results</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <ComparisonCard label="Probability of Default"
              before={`${(result.original_prediction.probability_of_default * 100).toFixed(1)}%`}
              after={`${(result.simulated_prediction.probability_of_default * 100).toFixed(1)}%`}
              delta={result.deltas.pd_change}
              invert
            />
            <ComparisonCard label="Risk Score"
              before={`${result.original_prediction.risk_score}/100`}
              after={`${result.simulated_prediction.risk_score}/100`}
              delta={result.deltas.risk_score_change}
              invert
            />
            <ComparisonCard label="Expected Loss"
              before={`$${result.original_prediction.expected_loss.toLocaleString()}`}
              after={`$${result.simulated_prediction.expected_loss.toLocaleString()}`}
              delta={result.deltas.expected_loss_change}
              invert
            />
            <div className="card">
              <div className="text-sm text-slate-400 mb-2">Decision</div>
              <div className="flex items-center gap-2 text-lg font-bold">
                <DecisionBadge decision={result.original_prediction.decision} />
                <ArrowRight size={18} className="text-slate-500" />
                <DecisionBadge decision={result.simulated_prediction.decision} />
              </div>
              {result.deltas.decision_changed && (
                <div className="mt-2 text-xs text-risk-moderate bg-risk-moderate/10 rounded px-2 py-1 inline-block">Decision changed</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ComparisonCard({ label, before, after, delta, invert = false }: { label: string, before: string, after: string, delta: number, invert?: boolean }) {
  const improved = invert ? delta < 0 : delta > 0
  return (
    <div className="card">
      <div className="text-sm text-slate-400 mb-3">{label}</div>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-slate-400">{before}</span>
        <ArrowRight size={14} className="text-slate-600" />
        <span className="text-white font-medium">{after}</span>
      </div>
      <div className={`text-sm flex items-center gap-1 ${improved ? 'text-risk-low' : 'text-risk-severe'}`}>
        {improved ? <TrendingDown size={14} /> : <TrendingUp size={14} />}
        {delta > 0 ? '+' : ''}{typeof delta === 'number' ? (Math.abs(delta) < 1 ? (delta * 100).toFixed(1) + ' pp' : delta.toFixed(0)) : delta}
      </div>
    </div>
  )
}

function DecisionBadge({ decision }: { decision: string }) {
  const color = decision === 'APPROVE' ? 'text-risk-low bg-risk-low/10' : decision === 'REJECT' ? 'text-risk-severe bg-risk-severe/10' : 'text-risk-moderate bg-risk-moderate/10'
  return <span className={`px-2 py-0.5 rounded text-sm font-bold ${color}`}>{decision}</span>
}
