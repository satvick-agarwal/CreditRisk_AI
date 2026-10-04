/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0f172a', // slate-900
        surface: '#1e293b', // slate-800
        border: '#334155', // slate-700
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9', // sky-500
          600: '#0284c7',
        },
        risk: {
          low: '#10b981', // emerald-500
          moderate: '#f59e0b', // amber-500
          high: '#f97316', // orange-500
          severe: '#ef4444', // red-500
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
