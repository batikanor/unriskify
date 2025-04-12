import { useState, useEffect } from 'react'
import './App.css'
import MarkdownViewer from './components/MarkdownViewer'

function App() {
  const [loading, setLoading] = useState(true)
  const [message, setMessage] = useState('')

  useEffect(() => {
    // Use environment variable or fallback to localhost
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
    fetch(`${apiUrl}/api/hello`)
      .then(response => response.json())
      .then(data => {
        setMessage(data.message)
        setLoading(false)
      })
      .catch(error => {
        console.error('Error fetching data:', error)
        setLoading(false)
      })
  }, [])

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo-container">
          <img src="/logo-unriskify.png" alt="Unriskify Logo" className="app-logo" />
        </div>
        <div className="header-text">
          <h2>Eats your risk away</h2>
        </div>
      </header>
      
      <main className="app-content">
        <MarkdownViewer markdownPath="/writeup.md" />
      </main>

      <footer className="app-footer">
        <div className="api-status">
          Backend status: {loading ? 'Connecting...' : message}
        </div>
      </footer>
    </div>
  )
}

export default App
