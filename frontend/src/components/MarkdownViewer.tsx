import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { marked } from 'marked';
import { motion, AnimatePresence } from 'framer-motion';
import '../styles/MarkdownViewer.css';

interface MarkdownViewerProps {
  markdownPath: string;
}

// Configure marked options for cleaner rendering
marked.setOptions({
  gfm: true,
  breaks: true
});

const MarkdownViewer = ({ markdownPath }: MarkdownViewerProps) => {
  const [markdown, setMarkdown] = useState<string>('');
  const [selectedText, setSelectedText] = useState<string>('');
  const [characterCount, setCharacterCount] = useState<number | null>(null);
  const [popupPosition, setPopupPosition] = useState<{ x: number; y: number } | null>(null);
  const [isPopupVisible, setIsPopupVisible] = useState<boolean>(false);
  const markdownRef = useRef<HTMLDivElement>(null);

  // Load markdown content
  useEffect(() => {
    fetch(markdownPath)
      .then(response => response.text())
      .then(data => {
        setMarkdown(data);
      })
      .catch(error => {
        console.error('Error loading markdown file:', error);
        setMarkdown('# Error loading content\nUnable to load the markdown file.');
      });
  }, [markdownPath]);

  // Count characters when text is selected
  const countCharacters = async (text: string) => {
    if (!text) {
      console.log('No text selected');
      return;
    }
    
    try {
      console.log('Sending text to backend:', text);
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${apiUrl}/api/count-characters`, { text });
      console.log('Backend response:', response.data);
      setCharacterCount(response.data.count);
      setIsPopupVisible(true);
    } catch (error) {
      console.error('Error counting characters:', error);
    }
  };
  
  // Process the selection and show character count
  const processSelection = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      const text = selection.toString().trim();
      console.log('Selected text:', text);
      setSelectedText(text);
      
      // Get selection position for popup
      if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        setPopupPosition({
          x: rect.left + (rect.width / 2),
          y: rect.bottom + window.scrollY + 10
        });
        
        // Count characters
        countCharacters(text);
      }
    }
  };
  
  // Handle mouseup event to capture selection
  const handleMouseUp = (e: React.MouseEvent) => {
    // Small delay to ensure selection is complete
    setTimeout(() => {
      processSelection();
    }, 10);
  };
  
  // Keep selection when clicking inside content
  const handleClick = (e: React.MouseEvent) => {
    if (selectedText) {
      e.preventDefault();
      e.stopPropagation();
    }
  };
  
  // Close the popup
  const closePopup = () => {
    setIsPopupVisible(false);
  };
  
  // Handle document clicks to maintain selection
  useEffect(() => {
    const handleDocumentClick = (e: MouseEvent) => {
      // Keep the selection alive unless we click outside the content
      const markdownContent = markdownRef.current;
      if (markdownContent && !markdownContent.contains(e.target as Node)) {
        // Only close the popup if we click outside
        closePopup();
      }
    };
    
    document.addEventListener('click', handleDocumentClick);
    return () => {
      document.removeEventListener('click', handleDocumentClick);
    };
  }, []);
  
  return (
    <div className="markdown-viewer">
      <div 
        ref={markdownRef}
        className="markdown-content"
        onMouseUp={handleMouseUp}
        onClick={handleClick}
        dangerouslySetInnerHTML={{ __html: marked(markdown) }}
      />
      
      <AnimatePresence>
        {isPopupVisible && popupPosition && characterCount !== null && (
          <motion.div 
            className="character-popup"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0, transition: { duration: 0.3 } }}
            exit={{ opacity: 0, y: -10 }}
            style={{ 
              position: 'absolute',
              left: `${popupPosition.x}px`,
              top: `${popupPosition.y}px`,
              transform: 'translateX(-50%)'
            }}
          >
            <p>Selected: <strong>{characterCount}</strong> characters</p>
            <button 
              className="close-popup" 
              onClick={closePopup}
              aria-label="Close"
            >
              ✕
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default MarkdownViewer; 