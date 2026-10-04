import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend, LineChart, Line } from 'recharts'
import { Activity, TrendingUp, AlertTriangle, DollarSign, Filter } from 'lucide-react'

const API_URL = 'http://127.0.0.1:8000/api'

const RISK_COLORS = { 
  low: '#10b981', moderate: '#f59e0b', high: '#f97316', severe: '#ef4444',
  primary: '#0ea5e9', slate: '#64748b'
}

const CREDIT_COLORS = ['#ef4444', '#f97316', '#f59e0b', '#22c55e', '#10b981']

export function Portfolio() {
  const [data, setData] = useState<any>(null)
  const [distribution, setDistribution] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState({
    credit_min: '', credit_max: '', income_min: '', income_max: '', intent: '', home_ownership: ''
  })

  const fetchData = async () => {
    setLoading(true)
    try {
      const params: any = {}
      Object.entries(filters).forEach(([k, v]) => { if (v !== '') params[k] = v })

      const [analyticsRes, distRes] = await Promise.all([
        axios.get(`${API_URL}/portfolio/analytics`, { params }),
        axios.get(`${API_URL}/dashboard/risk-distribution`),
      ])
      setData(analyticsRes.data)
      setDistribution(distRes.data)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

  const handleFilter = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFilters(prev => ({ ...prev, [e.target.name]: e.target.value }))
  }

  if (loading) {
    return <div className="flex h-64 items-center justify-center"><div className="animate-spin text-primary-500"><Activity size={40} /></div></div>
  }

  // Chart data from risk distribution
  const creditChartData = distribution?.by_credit_score
    ? Object.entries(distribution.by_credit_score).map(([band, stats]: [string, any]) => ({
        name: band.replace(' (', '\n('), count: stats.count, default_rate: (stats.default_rate * 100).toFixed(1)
      }))
    : []

  const intentChartData = distribution?.by_intent
    ? Object.entries(distribution.by_intent).map(([intent, stats]: [string, any]) => ({
        name: intent, count: stats.count, default_rate: (stats.default_rate * 100).toFixed(1), defaults: stats.defaults
      }))
    : []

  const incomeChartData = distribution?.by_income
    ? Object.entries(distribution.by_income).map(([band, stats]: [string, any]) => ({
        name: band, count: stats.count, default_rate: (stats.default_rate * 100).toFixed(1)
      }))
    : []

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Portfolio Analytics</h2>
        <p className="text-slate-400">Interactive analysis of the loan portfolio with cross-filtering.</p>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex items-center gap-2 text-sm text-slate-400 mb-4">
          <Filter size={16} />
          <span className="font-medium">Filters</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <div><label className="label">Credit Min</label><input type="number" name="credit_min" value={filters.credit_min} onChange={handleFilter} className="input-field" placeholder="300" /></div>
          <div><label className="label">Credit Max</label><input type="number" name="credit_max" value={filters.credit_max} onChange={handleFilter} className="input-field" placeholder="850" /></div>
          <div><label className="label">Income Min</label><input type="number" name="income_min" value={filters.income_min} onChange={handleFilter} className="input-field" placeholder="0" /></div>
          <div><label className="label">Income Max</label><input type="number" name="income_max" value={filters.income_max} onChange={handleFilter} className="input-field" placeholder="1M" /></div>
          <div><label className="label">Intent</label>
            <select name="intent" value={filters.intent} onChange={handleFilter} className="input-field">
              <option value="">All</option>
              <option value="PERSONAL">Personal</option>
              <option value="EDUCATION">Education</option>
              <option value="MEDICAL">Medical</option>
              <option value="VENTURE">Venture</option>
              <option value="DEBTCONSOLIDATION">Debt Consolidation</option>
              <option value="HOMEIMPROVEMENT">Home Improvement</option>
            </select>
          </div>
          <div className="flex items-end">
            <button onClick={fetchData} className="btn-primary w-full">Apply</button>
          </div>
        </div>
      </div>

      {/* KPIs */}
      {data && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="card"><div className="text-sm text-slate-400 mb-1">Matched Records</div><div className="text-2xl font-bold text-white">{data.count.toLocaleString()}</div></div>
          <div className="card"><div className="text-sm text-slate-400 mb-1">Default Rate</div><div className={`text-2xl font-bold ${data.default_rate > 0.2 ? 'text-risk-severe' : 'text-risk-low'}`}>{(data.default_rate * 100).toFixed(1)}%</div></div>
          <div className="card"><div className="text-sm text-slate-400 mb-1">Avg Credit Score</div><div className="text-2xl font-bold text-white">{data.avg_credit_score?.toFixed(0) || '—'}</div></div>
          <div className="card"><div className="text-sm text-slate-400 mb-1">Total Exposure</div><div className="text-2xl font-bold text-white">${(data.total_exposure / 1e6).toFixed(1)}M</div></div>
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Default by Credit Score */}
        <div className="card">
          <h3 className="text-lg font-medium text-white mb-4">Default Rate by Credit Score</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={creditChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} interval={0} />
              <YAxis tick={{ fill: '#94a3b8' }} unit="%" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#fff' }} />
              <Bar dataKey="default_rate" name="Default Rate %" radius={[4,4,0,0]}>
                {creditChartData.map((_, idx) => <Cell key={idx} fill={CREDIT_COLORS[idx % CREDIT_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Default by Intent */}
        <div className="card">
          <h3 className="text-lg font-medium text-white mb-4">Default Rate by Loan Purpose</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={intentChartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" tick={{ fill: '#94a3b8' }} unit="%" />
              <YAxis dataKey="name" type="category" tick={{ fill: '#94a3b8', fontSize: 12 }} width={120} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#fff' }} />
              <Bar dataKey="default_rate" name="Default Rate %" fill={RISK_COLORS.primary} radius={[0,4,4,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Default by Income */}
        <div className="card lg:col-span-2">
          <h3 className="text-lg font-medium text-white mb-4">Default Rate by Income Band</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={incomeChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8' }} />
              <YAxis tick={{ fill: '#94a3b8' }} unit="%" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#fff' }} />
              <Bar dataKey="default_rate" name="Default Rate %" fill={RISK_COLORS.moderate} radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
