import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts'
import { Activity, CheckCircle, AlertTriangle, Info, Shield } from 'lucide-react'

const API_URL = 'http://127.0.0.1:8000/api'

export function ModelInsights() {
  const [metrics, setMetrics] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchMetrics() {
      try {
        const res = await axios.get(`${API_URL}/model/metrics`)
        setMetrics(res.data)
      } catch (e) {
        console.error(e)
      } finally {
        setLoading(false)
      }
    }
    fetchMetrics()
  }, [])

  if (loading) return <div className="flex h-64 items-center justify-center"><div className="animate-spin text-primary-500"><Activity size={40} /></div></div>
  if (!metrics) return <div className="text-slate-400">Failed to load model metrics.</div>

  const testMetrics = metrics.test_metrics || {}
  const validationMetrics = metrics.validation_metrics || {}
  const calibration = metrics.calibration || {}
  const shapImportance = (metrics.shap_feature_importance || []).slice(0, 15)

  // Model comparison data
  const comparisonData = Object.entries(validationMetrics).map(([name, m]: [string, any]) => ({
    name: name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    'ROC-AUC': m.roc_auc,
    'PR-AUC': m.pr_auc,
    'F1': m.f1,
    'Recall': m.recall,
    'Precision': m.precision,
  }))

  // Feature importance chart data
  const featureData = shapImportance.map(([name, value]: [string, number]) => ({
    name: name.length > 25 ? name.slice(0, 22) + '...' : name,
    importance: Math.round(value * 10000) / 10000,
  })).reverse()

  // Confusion matrix
  const cm = testMetrics.confusion_matrix || [[0,0],[0,0]]

  // Radar chart for test metrics
  const radarData = [
    { metric: 'ROC-AUC', value: testMetrics.roc_auc || 0 },
    { metric: 'PR-AUC', value: testMetrics.pr_auc || 0 },
    { metric: 'F1', value: testMetrics.f1 || 0 },
    { metric: 'Recall', value: testMetrics.recall || 0 },
    { metric: 'Precision', value: testMetrics.precision || 0 },
    { metric: '1 - Brier', value: 1 - (testMetrics.brier_score || 0) },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Model Insights</h2>
        <p className="text-slate-400">Evaluation metrics, feature importance, and calibration analysis.</p>
      </div>

      {/* Model Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="text-sm text-slate-400 mb-1">Selected Model</div>
          <div className="text-xl font-bold text-white capitalize">{metrics.selected_model?.replace(/_/g, ' ')}</div>
          <div className="text-xs text-slate-500 mt-1">{metrics.model_version}</div>
        </div>
        <div className="card">
          <div className="text-sm text-slate-400 mb-1">Test ROC-AUC</div>
          <div className="text-2xl font-bold text-risk-low">{testMetrics.roc_auc?.toFixed(4)}</div>
        </div>
        <div className="card">
          <div className="text-sm text-slate-400 mb-1">Test Recall</div>
          <div className="text-2xl font-bold text-white">{testMetrics.recall?.toFixed(4)}</div>
          <div className="text-xs text-slate-500 mt-1">FNR: {testMetrics.false_negative_rate?.toFixed(4)}</div>
        </div>
        <div className="card">
          <div className="text-sm text-slate-400 mb-1">Calibration</div>
          <div className="text-lg font-bold text-white flex items-center gap-2">
            {calibration.calibration_applied ? (
              <><CheckCircle size={18} className="text-risk-low" /> Applied ({calibration.calibration_method})</>
            ) : (
              <><Info size={18} className="text-slate-400" /> Not needed</>
            )}
          </div>
          {calibration.calibration_applied && (
            <div className="text-xs text-slate-500 mt-1">Brier improved by {calibration.improvement?.toFixed(6)}</div>
          )}
        </div>
      </div>

      {/* Selection Note */}
      <div className="bg-primary-600/5 border border-primary-600/20 rounded-xl p-4 flex items-start gap-3">
        <Shield size={20} className="text-primary-500 mt-0.5 shrink-0" />
        <div className="text-sm text-slate-300">
          <strong className="text-white">Model Selection:</strong> {metrics.model_selection_note}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Comparison */}
        <div className="card">
          <h3 className="text-lg font-medium text-white mb-4">Model Comparison (Validation Set)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" domain={[0, 1]} tick={{ fill: '#94a3b8' }} />
              <YAxis dataKey="name" type="category" tick={{ fill: '#94a3b8', fontSize: 12 }} width={120} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#fff' }} />
              <Bar dataKey="ROC-AUC" fill="#0ea5e9" radius={[0,4,4,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Test Performance Radar */}
        <div className="card">
          <h3 className="text-lg font-medium text-white mb-4">Test Set Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <PolarRadiusAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
              <Radar name="Score" dataKey="value" stroke="#0ea5e9" fill="#0ea5e9" fillOpacity={0.2} strokeWidth={2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* SHAP Feature Importance */}
        <div className="card lg:col-span-2">
          <h3 className="text-lg font-medium text-white mb-1">SHAP Global Feature Importance</h3>
          <p className="text-xs text-slate-500 mb-4">Mean absolute SHAP values across the validation set. Higher = more influential on predictions.</p>
          <ResponsiveContainer width="100%" height={Math.max(300, featureData.length * 28)}>
            <BarChart data={featureData} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" tick={{ fill: '#94a3b8' }} />
              <YAxis dataKey="name" type="category" tick={{ fill: '#94a3b8', fontSize: 11 }} width={180} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#fff' }} />
              <Bar dataKey="importance" name="Mean |SHAP|" fill="#0ea5e9" radius={[0,4,4,0]}>
                {featureData.map((_, idx) => (
                  <Cell key={idx} fill={idx < 3 ? '#f59e0b' : '#0ea5e9'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrix */}
      <div className="card">
        <h3 className="text-lg font-medium text-white mb-4">Confusion Matrix (Test Set)</h3>
        <div className="grid grid-cols-2 max-w-md mx-auto gap-1">
          <div className="text-center p-6 bg-risk-low/10 border border-risk-low/30 rounded-tl-xl">
            <div className="text-2xl font-bold text-risk-low">{cm[0]?.[0]?.toLocaleString()}</div>
            <div className="text-xs text-slate-400 mt-1">True Negative</div>
          </div>
          <div className="text-center p-6 bg-risk-moderate/10 border border-risk-moderate/30 rounded-tr-xl">
            <div className="text-2xl font-bold text-risk-moderate">{cm[0]?.[1]?.toLocaleString()}</div>
            <div className="text-xs text-slate-400 mt-1">False Positive</div>
          </div>
          <div className="text-center p-6 bg-risk-severe/10 border border-risk-severe/30 rounded-bl-xl">
            <div className="text-2xl font-bold text-risk-severe">{cm[1]?.[0]?.toLocaleString()}</div>
            <div className="text-xs text-slate-400 mt-1">False Negative</div>
          </div>
          <div className="text-center p-6 bg-risk-low/10 border border-risk-low/30 rounded-br-xl">
            <div className="text-2xl font-bold text-risk-low">{cm[1]?.[1]?.toLocaleString()}</div>
            <div className="text-xs text-slate-400 mt-1">True Positive</div>
          </div>
        </div>
        <p className="text-xs text-slate-500 text-center mt-3">
          False negatives (missed defaults) are financially costly. The model was tuned to balance recall with precision.
        </p>
      </div>

      {/* Excluded Features & Audit */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2"><AlertTriangle size={18} className="text-risk-moderate" /> Excluded Features</h3>
          {metrics.excluded_features && Object.entries(metrics.excluded_features).map(([feature, reason]: [string, any]) => (
            <div key={feature} className="mb-3 p-3 bg-slate-900 rounded-lg">
              <div className="text-sm font-medium text-white">{feature}</div>
              <div className="text-xs text-slate-400 mt-1">{reason}</div>
            </div>
          ))}
        </div>
        <div className="card">
          <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2"><Info size={18} className="text-primary-500" /> Feature Leakage Audit</h3>
          {metrics.leakage_audit && Object.entries(metrics.leakage_audit).map(([feature, note]: [string, any]) => (
            <div key={feature} className="mb-3 p-3 bg-slate-900 rounded-lg">
              <div className="text-sm font-medium text-white">{feature}</div>
              <div className="text-xs text-slate-400 mt-1">{note}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
