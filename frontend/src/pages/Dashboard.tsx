import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { Users, AlertTriangle, ShieldCheck, PieChart, Activity, DollarSign } from 'lucide-react'

const API_URL = 'http://127.0.0.1:8000/api'

export function Dashboard() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await axios.get(`${API_URL}/dashboard/summary`)
        setData(res.data)
      } catch (e) {
        console.error(e)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="animate-spin text-primary-500"><Activity size={48} /></div>
      </div>
    )
  }

  if (!data) return <div>Failed to load dashboard</div>

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Portfolio Overview</h2>
          <p className="text-slate-400">High-level metrics and risk distribution for the entire portfolio.</p>
        </div>
        {data.source === 'historical_baseline' && (
          <div className="bg-slate-800 border border-slate-700 rounded px-3 py-1 text-xs text-slate-400 flex items-center gap-2">
            <AlertTriangle size={14} className="text-risk-moderate" />
            Showing historical baseline. Run predictions to populate live dashboard.
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KpiCard 
          title="Total Applications" 
          value={data.total_applications.toLocaleString()} 
          icon={<Users size={20} />} 
        />
        <KpiCard 
          title="Total Exposure" 
          value={`$${(data.total_exposure / 1000000).toFixed(1)}M`} 
          icon={<DollarSign size={20} />} 
        />
        <KpiCard 
          title="Average PD" 
          value={`${(data.average_pd * 100).toFixed(1)}%`} 
          icon={<Activity size={20} />} 
          trend={data.average_pd > 0.15 ? "high" : "normal"}
        />
        <KpiCard 
          title="Expected Loss" 
          value={`$${(data.total_expected_loss / 1000000).toFixed(2)}M`} 
          icon={<PieChart size={20} />} 
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card lg:col-span-2 min-h-[300px] flex items-center justify-center relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-primary-600/5 to-transparent"></div>
          <div className="text-center z-10 w-full">
            <h3 className="text-lg font-medium text-white mb-2">Decision Funnel</h3>
            <div className="flex justify-center items-end gap-6 h-48 mt-8 w-full">
              <div className="flex flex-col items-center h-full w-24">
                <div className="flex-1 w-full flex items-end pb-2">
                  <div className="w-full bg-risk-low/20 border border-risk-low/50 rounded-t-lg transition-all duration-1000 ease-out" style={{ height: `${Math.max(data.approval_rate * 100, 2)}%` }}></div>
                </div>
                <div className="mt-auto text-risk-low font-bold">{(data.approval_rate * 100).toFixed(1)}%</div>
                <div className="text-xs text-slate-400 uppercase tracking-wider">Approve</div>
              </div>
              
              <div className="flex flex-col items-center h-full w-24">
                <div className="flex-1 w-full flex items-end pb-2">
                  <div className="w-full bg-risk-moderate/20 border border-risk-moderate/50 rounded-t-lg transition-all duration-1000 ease-out" style={{ height: `${Math.max(data.review_rate * 100, 2)}%` }}></div>
                </div>
                <div className="mt-auto text-risk-moderate font-bold">{(data.review_rate * 100).toFixed(1)}%</div>
                <div className="text-xs text-slate-400 uppercase tracking-wider">Review</div>
              </div>
              
              <div className="flex flex-col items-center h-full w-24">
                <div className="flex-1 w-full flex items-end pb-2">
                  <div className="w-full bg-risk-severe/20 border border-risk-severe/50 rounded-t-lg transition-all duration-1000 ease-out" style={{ height: `${Math.max(data.rejection_rate * 100, 2)}%` }}></div>
                </div>
                <div className="mt-auto text-risk-severe font-bold">{(data.rejection_rate * 100).toFixed(1)}%</div>
                <div className="text-xs text-slate-400 uppercase tracking-wider">Reject</div>
              </div>
            </div>
          </div>
        </div>

        <div className="card flex flex-col justify-between">
          <div>
            <h3 className="text-lg font-medium text-white mb-4">Risk Profile</h3>
            <p className="text-sm text-slate-400 mb-6">
              The portfolio currently exhibits an average risk score of <strong className="text-white">{data.average_risk_score.toFixed(0)}/100</strong>.
            </p>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700">
              <div className="flex items-center gap-3">
                <ShieldCheck className="text-risk-low" />
                <span className="text-sm font-medium text-white">Automated Approvals</span>
              </div>
              <span className="text-risk-low font-bold">{(data.approval_rate * 100).toFixed(1)}%</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700">
              <div className="flex items-center gap-3">
                <AlertTriangle className="text-risk-severe" />
                <span className="text-sm font-medium text-white">Automated Rejections</span>
              </div>
              <span className="text-risk-severe font-bold">{(data.rejection_rate * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function KpiCard({ title, value, icon, trend }: { title: string, value: string, icon: React.ReactNode, trend?: 'high' | 'normal' }) {
  return (
    <div className="card hover:border-slate-600 transition-colors">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-slate-400">{title}</h3>
        <div className={`p-2 rounded-lg ${trend === 'high' ? 'bg-risk-severe/10 text-risk-severe' : 'bg-primary-600/10 text-primary-500'}`}>
          {icon}
        </div>
      </div>
      <div className={`text-3xl font-bold ${trend === 'high' ? 'text-risk-severe' : 'text-white'}`}>
        {value}
      </div>
    </div>
  )
}
