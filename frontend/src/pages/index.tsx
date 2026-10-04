import React, { useState } from 'react'
import axios from 'axios'
import { Activity, AlertTriangle, CheckCircle, ShieldAlert, FileText, ChevronRight } from 'lucide-react'

export { Dashboard } from './Dashboard'
export { Simulator } from './Simulator'
export { Portfolio } from './Portfolio'
export { ModelInsights } from './ModelInsights'
export { AIAnalyst } from './AIAnalyst'

const API_URL = 'http://127.0.0.1:8000/api'

export function Assessment() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  
  const [formData, setFormData] = useState({
    person_age: 28,
    person_education: "Bachelor",
    person_income: 65000,
    person_emp_exp: 5,
    person_home_ownership: "RENT",
    loan_amnt: 15000,
    loan_intent: "PERSONAL",
    loan_int_rate: 11.5,
    cb_person_cred_hist_length: 4,
    credit_score: 680,
    previous_loan_defaults_on_file: "No"
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    // Convert numeric fields
    const isNumeric = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length', 'credit_score'].includes(name)
    setFormData(prev => ({
      ...prev,
      [name]: isNumeric ? (value === '' ? '' : Number(value)) : value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await axios.post(`${API_URL}/predictions`, formData)
      setResult(res.data)
    } catch (err) {
      console.error(err)
      alert("Error evaluating applicant.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Applicant Assessment</h2>
          <p className="text-slate-400">Evaluate a new applicant using the ML decision engine.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form Column */}
        <div className="lg:col-span-2">
          <div className="card">
            <form onSubmit={handleSubmit} className="space-y-6">
              
              <div>
                <h3 className="text-lg font-medium text-white mb-4 border-b border-border pb-2">Applicant Profile</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="label">Age</label>
                    <input type="number" name="person_age" value={formData.person_age} onChange={handleChange} className="input-field" required min="18" max="100"/>
                  </div>
                  <div>
                    <label className="label">Annual Income ($)</label>
                    <input type="number" name="person_income" value={formData.person_income} onChange={handleChange} className="input-field" required min="0"/>
                  </div>
                  <div>
                    <label className="label">Employment Experience (Yrs)</label>
                    <input type="number" name="person_emp_exp" value={formData.person_emp_exp} onChange={handleChange} className="input-field" required min="0" max="80"/>
                  </div>
                  <div>
                    <label className="label">Home Ownership</label>
                    <select name="person_home_ownership" value={formData.person_home_ownership} onChange={handleChange} className="input-field">
                      <option value="RENT">Rent</option>
                      <option value="MORTGAGE">Mortgage</option>
                      <option value="OWN">Own</option>
                      <option value="OTHER">Other</option>
                    </select>
                  </div>
                  <div>
                    <label className="label">Education</label>
                    <select name="person_education" value={formData.person_education} onChange={handleChange} className="input-field">
                      <option value="High School">High School</option>
                      <option value="Associate">Associate</option>
                      <option value="Bachelor">Bachelor</option>
                      <option value="Master">Master</option>
                      <option value="Doctorate">Doctorate</option>
                    </select>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium text-white mb-4 border-b border-border pb-2 mt-2">Loan Request</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="label">Loan Amount ($)</label>
                    <input type="number" name="loan_amnt" value={formData.loan_amnt} onChange={handleChange} className="input-field" required min="100"/>
                  </div>
                  <div>
                    <label className="label">Loan Intent</label>
                    <select name="loan_intent" value={formData.loan_intent} onChange={handleChange} className="input-field">
                      <option value="PERSONAL">Personal</option>
                      <option value="EDUCATION">Education</option>
                      <option value="MEDICAL">Medical</option>
                      <option value="VENTURE">Venture</option>
                      <option value="DEBTCONSOLIDATION">Debt Consolidation</option>
                      <option value="HOMEIMPROVEMENT">Home Improvement</option>
                    </select>
                  </div>
                  <div>
                    <label className="label">Interest Rate (%)</label>
                    <input type="number" step="0.01" name="loan_int_rate" value={formData.loan_int_rate} onChange={handleChange} className="input-field" required min="0"/>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium text-white mb-4 border-b border-border pb-2 mt-2">Credit Profile</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="label">Credit Score</label>
                    <input type="number" name="credit_score" value={formData.credit_score} onChange={handleChange} className="input-field" required min="300" max="850"/>
                  </div>
                  <div>
                    <label className="label">Credit History Length (Yrs)</label>
                    <input type="number" name="cb_person_cred_hist_length" value={formData.cb_person_cred_hist_length} onChange={handleChange} className="input-field" required min="0" max="60"/>
                  </div>
                  <div>
                    <label className="label">Previous Defaults?</label>
                    <select name="previous_loan_defaults_on_file" value={formData.previous_loan_defaults_on_file} onChange={handleChange} className="input-field">
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="pt-4 flex justify-end">
                <button type="submit" disabled={loading} className="btn-primary w-full md:w-auto flex items-center justify-center gap-2">
                  {loading ? (
                    <span className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                      Evaluating Risk...
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <Activity size={18} />
                      Run Risk Assessment
                    </span>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Results Column */}
        <div className="lg:col-span-1">
          {result ? (
            <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
              
              {/* Decision Card */}
              <div className={`rounded-xl p-6 shadow-lg border ${
                result.decision === 'APPROVE' ? 'bg-risk-low/10 border-risk-low/30' :
                result.decision === 'REVIEW' ? 'bg-risk-moderate/10 border-risk-moderate/30' :
                'bg-risk-severe/10 border-risk-severe/30'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-sm font-medium uppercase tracking-wider text-slate-400">Final Decision</span>
                  {result.decision === 'APPROVE' && <CheckCircle className="text-risk-low" />}
                  {result.decision === 'REVIEW' && <AlertTriangle className="text-risk-moderate" />}
                  {result.decision === 'REJECT' && <ShieldAlert className="text-risk-severe" />}
                </div>
                <div className={`text-4xl font-bold mb-2 ${
                  result.decision === 'APPROVE' ? 'text-risk-low' :
                  result.decision === 'REVIEW' ? 'text-risk-moderate' :
                  'text-risk-severe'
                }`}>
                  {result.decision}
                </div>
                {result.decision_escalated && (
                  <p className="text-sm text-risk-moderate bg-risk-moderate/10 rounded px-2 py-1 inline-block mt-2">
                    Escalated by policy rules
                  </p>
                )}
              </div>

              {/* Core Metrics */}
              <div className="card space-y-4">
                <div className="flex justify-between items-end pb-4 border-b border-border">
                  <div>
                    <div className="text-sm text-slate-400 mb-1">Probability of Default</div>
                    <div className="text-2xl font-semibold text-white">{(result.probability_of_default * 100).toFixed(1)}%</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-slate-400 mb-1">Risk Score</div>
                    <div className="text-2xl font-semibold text-white">{result.risk_score} <span className="text-sm text-slate-500 font-normal">/ 100</span></div>
                  </div>
                </div>
                
                <div className="flex justify-between items-center">
                  <div className="text-sm text-slate-400">Risk Grade</div>
                  <div className="px-3 py-1 rounded bg-slate-800 text-white font-bold border border-slate-700">
                    {result.risk_grade} <span className="text-slate-400 font-normal ml-1">({result.risk_level})</span>
                  </div>
                </div>
              </div>

              {/* Financial Impact */}
              <div className="card">
                <h4 className="text-sm font-medium text-slate-400 mb-4 flex items-center gap-2">
                  <FileText size={16} /> Financial Exposure
                </h4>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Exposure (EAD)</span>
                    <span className="text-white">${result.ead.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Expected Loss</span>
                    <span className="text-white font-medium">${result.expected_loss.toLocaleString()}</span>
                  </div>
                  <div className="text-xs text-slate-500 mt-2 bg-slate-900 p-2 rounded">
                    LGD assumed at {(result.lgd_used * 100).toFixed(0)}% ({result.lgd_source})
                  </div>
                </div>
              </div>

              {/* SHAP Explanations */}
              {Object.keys(result.shap_contributions || {}).length > 0 && (
                <div className="card">
                  <h4 className="text-sm font-medium text-slate-400 mb-4">Risk Drivers (SHAP)</h4>
                  <div className="space-y-3">
                    {Object.entries(result.shap_contributions).slice(0, 5).map(([feature, impact]: [string, any]) => (
                      <div key={feature}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-slate-300 truncate pr-2">{feature}</span>
                          <span className={impact > 0 ? "text-risk-severe" : "text-risk-low"}>
                            {impact > 0 ? "+" : ""}{impact.toFixed(3)}
                          </span>
                        </div>
                        <div className="w-full bg-slate-900 rounded-full h-1.5 flex">
                          {impact < 0 && (
                            <div className="bg-risk-low h-1.5 rounded-full ml-auto" style={{ width: `${Math.min(Math.abs(impact) * 10, 100)}%` }}></div>
                          )}
                          {impact > 0 && (
                            <div className="bg-risk-severe h-1.5 rounded-full mr-auto" style={{ width: `${Math.min(Math.abs(impact) * 10, 100)}%` }}></div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            </div>
          ) : (
            <div className="card h-full flex flex-col items-center justify-center text-center text-slate-500 p-12 min-h-[400px]">
              <Activity size={48} className="mb-4 text-slate-700" />
              <h3 className="text-lg font-medium text-slate-300 mb-2">No Assessment Yet</h3>
              <p className="text-sm">Fill out the applicant form and run the assessment to see risk metrics and AI explanations.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
