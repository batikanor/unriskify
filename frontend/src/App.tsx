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
        <h1>Writeup Viewer</h1>
        <p>Select text to view character count</p>
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
