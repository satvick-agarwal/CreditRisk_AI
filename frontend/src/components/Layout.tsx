import React from 'react'
import { Outlet, NavLink } from 'react-router-dom'
import { LayoutDashboard, Users, Calculator, PieChart, Activity, BrainCircuit } from 'lucide-react'

export function Layout() {
  const navItems = [
    { to: "/", icon: <LayoutDashboard size={20} />, label: "Dashboard" },
    { to: "/assessment", icon: <Users size={20} />, label: "Assessment" },
    { to: "/simulator", icon: <Calculator size={20} />, label: "Risk Simulator" },
    { to: "/portfolio", icon: <PieChart size={20} />, label: "Portfolio" },
    { to: "/model-insights", icon: <Activity size={20} />, label: "Model Insights" },
    { to: "/ai-analyst", icon: <BrainCircuit size={20} />, label: "AI Analyst" },
  ]

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 border-r border-border bg-surface flex flex-col">
        <div className="h-16 flex items-center px-6 border-b border-border">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-primary-600 flex items-center justify-center text-white font-bold text-lg">
              C
            </div>
            <span className="font-bold text-lg tracking-tight text-white">CreditRisk <span className="text-primary-500">AI</span></span>
          </div>
        </div>
        
        <nav className="flex-1 py-6 px-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                  isActive 
                    ? "bg-primary-600/10 text-primary-500 font-medium" 
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                }`
              }
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))}
        </nav>
        
        <div className="p-4 border-t border-border text-xs text-slate-500">
          <p>CreditRisk AI Platform</p>
          <p>Model: v1.0 (LightGBM)</p>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <header className="h-16 border-b border-border bg-surface/50 backdrop-blur flex items-center px-8 shrink-0">
          <h1 className="text-xl font-semibold text-white">Decision Intelligence Platform</h1>
        </header>
        
        <div className="flex-1 overflow-y-auto p-8">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  )
}
