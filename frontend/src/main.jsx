import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './hooks/useTheme.js' // apply saved theme before first paint
import App from './App.jsx'
import { bootstrapProject } from './lib/projectSync'

const root = createRoot(document.getElementById('root'))

root.render(
  <div className="boot-screen">
    <span className="boot-dot" />
    Loading your project…
  </div>,
)

// The state hooks read localStorage on first render, so the saved project is
// loaded into it before the app mounts.
bootstrapProject().finally(() => {
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
})
