import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Dashboard, Assessment, Simulator, Portfolio, ModelInsights, AIAnalyst } from './pages'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="assessment" element={<Assessment />} />
          <Route path="simulator" element={<Simulator />} />
          <Route path="portfolio" element={<Portfolio />} />
          <Route path="model-insights" element={<ModelInsights />} />
          <Route path="ai-analyst" element={<AIAnalyst />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
