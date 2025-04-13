import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { marked } from 'marked';
import { motion, AnimatePresence } from 'framer-motion';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import Draggable from 'react-draggable';
import '../styles/MarkdownViewer.css';

// Icons for mode selectors
const ModeIcons = {
  'char-counter': '📊',
  'sample-graph': '🔍',
  'chat-rfq': '💬',
  'nothing': '⚪'
};

// Theme colors for consistency
const ThemeColors = {
  primary: '#4f2913',    // Dark brown
  secondary: '#7b8c43',  // Olive green
  accent: '#c07140',     // Terracotta
  light: '#f1e8d8',      // Beige/cream
};

// Define the available modes
type ViewerMode = 'char-counter' | 'nothing' | 'sample-graph' | 'chat-rfq';

interface MarkdownViewerProps {
  markdownPath: string;
}

// Sample graph data interfaces
interface GraphNode {
  id: string;
  name: string;
  val: number;
  color: string;
  group: string;
  content?: string;
  fullText?: string;
  position?: [number, number];
  relevance?: number;
  reason?: string;
  // Force graph adds these properties during simulation
  x?: number;
  y?: number;
  z?: number;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  value: number;
  color: string;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

interface SampleGraphData {
  sources: string[];
  opacities: number[];
  content: string[];
  selected_text: string;
  graph_data: GraphData;
  model_used?: string;
  error?: string;
}

interface ModelListData {
  openai_models: string[];
  ollama_models: string[];
}

// Message interface for chat
interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  webSearchUsed?: boolean;
  webSearchResults?: {
    answer?: string;
    results?: Array<{
      title: string;
      content: string;
      url: string;
      score: number;
    }>;
    provider?: string;
  };
}

// Configure marked with GFM (GitHub Flavored Markdown) support
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
  const [mode, setMode] = useState<ViewerMode>('char-counter');
  const [activeHighlight, setActiveHighlight] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<SampleGraphData | null>(null);
  const [isGraphVisible, setIsGraphVisible] = useState<boolean>(false);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [isInteractingWithGraph, setIsInteractingWithGraph] = useState<boolean>(false);
  const [aiModels, setAiModels] = useState<ModelListData>({ openai_models: [], ollama_models: [] });
  const [selectedModel, setSelectedModel] = useState<string>('gpt-4o');
  const [chunkSize, setChunkSize] = useState<number>(5);
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [lineConstraint, setLineConstraint] = useState<number | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [detailedNode, setDetailedNode] = useState<GraphNode | null>(null);
  const [paragraphHighlights, setParagraphHighlights] = useState<Array<{
    position: [number, number];
    relevance: number;
    color: string;
  }>>([]);
  
  // New chat-related state
  const [isChatVisible, setIsChatVisible] = useState<boolean>(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState<string>('');
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  
  const markdownContentRef = useRef<HTMLDivElement>(null);
  const highlightLayerRef = useRef<HTMLDivElement>(null);
  const contentContainerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
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

  // Fetch AI models when component mounts or when mode changes to sample-graph or chat-rfq
  useEffect(() => {
    if (mode === 'sample-graph' || mode === 'chat-rfq') {
      fetchModels();
    }
  }, [mode]);

  // Fetch available AI models
  const fetchModels = async () => {
    setIsLoadingModels(true);
    setModelError(null);
    
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.get(`${apiUrl}/api/models`);
      console.log('AI models:', response.data);
      
      setAiModels(response.data);
      
      // Set default model if none selected
      if (!selectedModel && response.data.openai_models.length > 0) {
        setSelectedModel(response.data.openai_models[0]);
      }
    } catch (error) {
      console.error('Error fetching AI models:', error);
      setModelError('Failed to load AI models');
      
      // Set a fallback default model
      if (!selectedModel) {
        setSelectedModel('gpt-4o');
      }
    } finally {
      setIsLoadingModels(false);
    }
  };

  // Handle mode change
  const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newMode = e.target.value as ViewerMode;
    setMode(newMode);
    // Close the popup when changing modes
    setIsPopupVisible(false);
    setIsGraphVisible(false);
    setHoveredNode(null);
    
    // Handle chat visibility
    if (newMode === 'chat-rfq') {
      setIsChatVisible(true);
      // Add initial welcome message if no messages exist
      if (chatMessages.length === 0) {
        setChatMessages([{
          id: Date.now().toString(),
          content: "👋 Welcome to your RfQ assistant! I can help analyze this document, provide insights, or answer questions about its content. I can also search the web for relevant external information when needed. How can I assist you today?",
          sender: 'ai',
          timestamp: new Date()
        }]);
      }
    } else {
      setIsChatVisible(false);
    }
    
    // Clear any active highlights
    clearCustomHighlights();
  };

  // Count characters when text is selected
  const countCharacters = async (text: string) => {
    if (!text || mode !== 'char-counter') {
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
  
  // Get sample graph data for the selected text
  const getSampleGraphData = async (text: string) => {
    if (!text || mode !== 'sample-graph') {
      return;
    }
    
    // Set loading state
    console.log('Setting loading state to true');
    setIsLoading(true);
    
    try {
      // Get the full document content from markdown
      const documentContent = markdown;
      
      console.log(`Getting sample graph data for: "${text}" using model: ${selectedModel}, chunk size: ${chunkSize}, line constraint: ${lineConstraint}`);
      console.log(`Including full document content: ${documentContent.length} characters`);
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${apiUrl}/api/sample-graph`, { 
        text,
        document_content: documentContent,  // Send full document content for context
        model: selectedModel,
        chunk_size: chunkSize,
        line_constraint: lineConstraint
      });
      console.log('Sample graph data received:', response.data);
      setGraphData(response.data);
      setIsGraphVisible(true);
    } catch (error) {
      console.error('Error getting sample graph data:', error);
      let errorMessage = "Failed to generate graph";
      
      if (axios.isAxiosError(error) && error.response) {
        errorMessage = `Error: ${error.response.data.detail || error.message}`;
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }
      
      // Create minimal graph data with error
      setGraphData({
        sources: ["Error"],
        opacities: [0.5],
        content: [errorMessage],
        selected_text: text,
        error: errorMessage,
        graph_data: {
          nodes: [
            {
              id: "error-node",
              name: "Error",
              val: 20,
              color: "#FF0000",
              group: "error",
              content: errorMessage
            }
          ],
          links: []
        }
      });
      setIsGraphVisible(true);
    } finally {
      // Reset loading state with a slight delay to ensure it's visible
      setTimeout(() => {
        console.log('Setting loading state to false');
        setIsLoading(false);
      }, 300);
    }
  };
  
  // Create a persistent custom highlight for the selected text
  const createPersistentHighlight = (selection: Selection) => {
    // Clear any existing highlights first
    clearCustomHighlights();
    
    if (!selection || !selection.rangeCount || selection.toString().trim() === '') {
      return;
    }
    
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const containerRect = contentContainerRef.current?.getBoundingClientRect();
    
    if (!containerRect) return;
    
    // Calculate position relative to content container
    const relativeLeft = rect.left - containerRect.left;
    const relativeTop = rect.top - containerRect.top;
    
    // Create a highlight element
    const highlightId = `highlight-${Date.now()}`;
    const highlightElement = document.createElement('div');
    highlightElement.id = highlightId;
    highlightElement.className = 'custom-highlight';
    
    // Position the highlight to match the selection
    highlightElement.style.position = 'absolute';
    highlightElement.style.left = `${relativeLeft}px`;
    highlightElement.style.top = `${relativeTop}px`;
    highlightElement.style.width = `${rect.width}px`;
    highlightElement.style.height = `${rect.height}px`;
    
    // Append to highlight layer
    if (highlightLayerRef.current) {
      highlightLayerRef.current.appendChild(highlightElement);
      setActiveHighlight(highlightId);
    }
    
    return { highlightId, rect };
  };
  
  // Clear all custom highlights
  const clearCustomHighlights = () => {
    if (highlightLayerRef.current) {
      highlightLayerRef.current.innerHTML = '';
      setActiveHighlight(null);
    }
  };
  
  // Process the selection and show character count or graph data
  const processSelection = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      const text = selection.toString().trim();
      setSelectedText(text);
      
      // Create a persistent highlight
      const highlightData = createPersistentHighlight(selection);
      
      if (highlightData) {
        // Get selection position for popup
        setPopupPosition({
          x: highlightData.rect.left + (highlightData.rect.width / 2),
          y: highlightData.rect.bottom + window.scrollY + 10
        });
        
        // Process according to mode
        if (mode === 'char-counter') {
          countCharacters(text);
        } else if (mode === 'sample-graph') {
          getSampleGraphData(text);
        }
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
    if (activeHighlight) {
      e.preventDefault();
      e.stopPropagation();
    }
  };
  
  // Close the popup and clear highlights
  const closePopup = () => {
    setIsPopupVisible(false);
    setIsGraphVisible(false);
    setHoveredNode(null);
    clearCustomHighlights();
  };
  
  // Handle window resize for graph
  useEffect(() => {
    const handleGraphResize = () => {
      if (isGraphVisible && fgRef.current) {
        // Force graph to update dimensions
        fgRef.current.width(window.innerWidth * 0.8);
        fgRef.current.height(window.innerHeight * 0.7);
      }
    };
    
    window.addEventListener('resize', handleGraphResize);
    return () => {
      window.removeEventListener('resize', handleGraphResize);
    };
  }, [isGraphVisible]);
  
  // Handle window resize to update highlight positions
  useEffect(() => {
    const handleHighlightResize = () => {
      // Force recreation of highlights on resize
      if (activeHighlight) {
        clearCustomHighlights();
        // Use a small delay to let the layout stabilize
        setTimeout(() => {
          const selection = window.getSelection();
          if (selection && selection.toString().trim()) {
            createPersistentHighlight(selection);
          }
        }, 100);
      }
    };
    
    window.addEventListener('resize', handleHighlightResize);
    return () => {
      window.removeEventListener('resize', handleHighlightResize);
    };
  }, [activeHighlight]);
  
  // Handle document clicks to maintain selection
  useEffect(() => {
    const handleDocumentClick = (e: MouseEvent) => {
      // If we're interacting with the graph, don't close the popup
      if (isInteractingWithGraph) {
        return;
      }
      
      // Keep the selection alive unless we click outside the content
      const markdownContent = markdownContentRef.current;
      const graphPopup = document.querySelector('.graph-popup');
      
      // Don't close if click is within markdown content or graph popup
      if ((markdownContent && markdownContent.contains(e.target as Node)) || 
          (graphPopup && graphPopup.contains(e.target as Node))) {
        return;
      }
      
      // Close popup if click is outside
      closePopup();
    };
    
    document.addEventListener('click', handleDocumentClick);
    return () => {
      document.removeEventListener('click', handleDocumentClick);
    };
  }, [isInteractingWithGraph]);
  
  // Focus on a node in the 3D graph and zoom in without showing detail panel
  const handleNodeClick = useCallback((node: any, event: any) => {
    if (fgRef.current && node) {
      // Mark that we're interacting with the graph to prevent popup closure
      setIsInteractingWithGraph(true);
      
      // Prevent the event from bubbling up to the document click handler
      if (event && event.srcEvent) {
        event.srcEvent.stopPropagation();
        event.srcEvent.preventDefault();
      }
      
      // Closer camera zoom for better reading
      const distance = 40; // Reduced distance for closer zoom
      const distRatio = 1 + distance/Math.hypot(node.x || 0, node.y || 0, node.z || 0);

      fgRef.current.cameraPosition(
        { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio }, // new position
        { x: node.x || 0, y: node.y || 0, z: node.z || 0 }, // lookAt position
        800  // ms transition duration - slightly faster
      );
      
      // Reset the interaction flag after animation completes
      setTimeout(() => {
        setIsInteractingWithGraph(false);
      }, 900);
    }
  }, []);
  
  // Custom Three.js object for nodes
  const nodeThreeObject = useCallback((node: any) => {
    // Extract color
    const color = node.color || '#888888';
    
    // Handle different node types
    switch (node.group) {
      case 'detail': // Alternative remarks
        return createRemarkNode(node);
      case 'source': // Source documents
        return createMarkdownNode(node);
      case 'selected': // Selected text
      default:
        return createStandardTextNode(node, color);
    }
  }, []);
  
  // Create a node for alternative remarks that includes full text and reasoning - bigger version
  const createRemarkNode = (node: any) => {
    // Create a custom HTML-like canvas for the remark
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    const pixelRatio = window.devicePixelRatio || 1;
    
    // Use the complete remark text instead of the number
    const remarkText = node.content || 'No content available';
    const reasonText = node.reason || '';
    
    // Set up the canvas with appropriate dimensions - MUCH LARGER
    const maxWidth = 3000; // 3x larger (was 1000)
    const lineHeight = 126; // 3x larger (was 42)
    
    // Measure and wrap text to fit within maxWidth
    ctx.font = `72px Arial`; // 3x larger (was 24px)
    const wrappedRemarkText = wrapText(ctx, remarkText, maxWidth - 50);
    const wrappedReasonText = reasonText ? wrapText(ctx, `Reasoning: ${reasonText}`, maxWidth - 50) : [];
    
    // Calculate canvas height based on text content
    const remarkLines = wrappedRemarkText.length;
    const reasonLines = wrappedReasonText.length;
    const headerHeight = 150; // 3x larger (was 50)
    const remarkHeight = remarkLines * lineHeight + 120; // 3x larger (was 40)
    const reasonHeight = reasonLines > 0 ? (reasonLines * lineHeight + 120) : 0; // 3x larger (was 40)
    const footerHeight = 60; // 3x larger (was 20)
    
    const totalHeight = headerHeight + remarkHeight + reasonHeight + footerHeight;
    
    // Set canvas dimensions
    canvas.width = maxWidth * pixelRatio;
    canvas.height = totalHeight * pixelRatio;
    
    // Scale context
    ctx.scale(pixelRatio, pixelRatio);
    
    // Clear canvas
    ctx.clearRect(0, 0, maxWidth, totalHeight);
    
    // Draw background - solid black
    ctx.fillStyle = '#000000'; // Pure black, no opacity
    roundRect(ctx, 0, 0, maxWidth, totalHeight, 42); // 3x larger (was 14)
    
    // Draw header
    ctx.fillStyle = node.color || '#4a6fa5';
    roundRect(ctx, 0, 0, maxWidth, headerHeight, { tl: 42, tr: 42, bl: 0, br: 0 }); // 3x larger (was 14)
    
    // Draw header text
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 66px Arial'; // 3x larger (was 22px)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Alternative Remark', maxWidth / 2, headerHeight / 2);
    
    // Draw remark content
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '54px Arial'; // 3x larger (was 18px)
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    
    wrappedRemarkText.forEach((line, i) => {
      ctx.fillText(line, 75, headerHeight + 60 + (i * lineHeight)); // 3x larger (was 25, 20)
    });
    
    // Draw separator if there's a reason
    if (reasonLines > 0) {
      const separatorY = headerHeight + remarkHeight;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'; // Keep separator opacity
      ctx.lineWidth = 4.5; // 3x thicker (was 1.5)
      ctx.beginPath();
      ctx.moveTo(75, separatorY); // 3x larger (was 25)
      ctx.lineTo(maxWidth - 75, separatorY); // 3x larger (was 25)
      ctx.stroke();
      
      // Draw reason text
      ctx.fillStyle = '#CCCCCC';
      ctx.font = 'italic 51px Arial'; // 3x larger (was 17px)
      
      wrappedReasonText.forEach((line, i) => {
        ctx.fillText(line, 75, separatorY + 60 + (i * lineHeight)); // 3x larger (was 25, 20)
      });
    }
    
    // Create sprite with the canvas texture
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ 
      map: texture,
      transparent: true
    });
    
    const sprite = new THREE.Sprite(material);
    
    // Set scale - MUCH larger for better readability
    const aspectRatio = totalHeight / maxWidth;
    const scale = node.val * 2.4;  // 3x bigger (was 0.8)
    sprite.scale.set(scale, scale * aspectRatio, 1);
    
    return sprite;
  };
  
  // Create a node for source documents that renders markdown content - improved rendering
  const createMarkdownNode = (node: any) => {
    // Create a custom HTML-like canvas for the markdown content
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    const pixelRatio = window.devicePixelRatio || 1;
    
    // Extract source name and full content
    const sourceName = node.name || 'Source Document';
    const contentPreview = node.content || 'No content available';
    
    // Detect if this is tabular data based on content patterns
    const isTabularData = contentPreview.includes('|') && 
                          (contentPreview.includes('\n|') || contentPreview.match(/\|\s*\d+\s*\|/g));
    
    // For tabular data, use a specialized parser
    if (isTabularData) {
      return createTableNode(node, sourceName, contentPreview);
    }
    
    // Parse the markdown to get properly formatted text
    const parsedContent = parseSimpleMarkdown(contentPreview);
    
    // Set up the canvas with appropriate dimensions - distinct from remark nodes
    const maxWidth = 3200;
    const lineHeight = 72;
    
    // Measure and wrap text to fit within maxWidth
    ctx.font = `18px Arial`;
    
    // Flatten and wrap paragraphs for rendering
    const allWrappedText: {text: string, style: string}[] = [];
    
    // Add document title as first line
    allWrappedText.push({
      text: sourceName,
      style: 'heading1'
    });
    
    // Add a separator line
    allWrappedText.push({text: '', style: 'normal'});
    
    // Get first few paragraphs for preview
    let paragraphCount = 0;
    const maxParagraphs = 5;
    
    parsedContent.forEach(item => {
      // Limit the number of paragraphs for preview
      if (item.style === 'normal' && item.text.trim().length > 0) {
        paragraphCount++;
      }
      
      if (paragraphCount <= maxParagraphs) {
        const font = getMarkdownFont(item.style);
        ctx.font = font;
        
        const wrappedLines = wrapText(ctx, item.text, maxWidth - 160); // Wider margin for cleaner look
        wrappedLines.forEach(line => {
          allWrappedText.push({
            text: line,
            style: item.style
          });
        });
        
        // Add a blank line between paragraphs
        if (item.style !== 'code') {
          allWrappedText.push({text: '', style: 'normal'});
        }
      }
    });
    
    // Add "View more..." if content was truncated
    if (paragraphCount > maxParagraphs) {
      allWrappedText.push({text: 'View more...', style: 'italic'});
    }
    
    // Calculate canvas height based on content
    const topMargin = 40;
    const contentLines = allWrappedText.length;
    const contentHeight = contentLines * lineHeight + 80; // Add extra space
    
    const totalHeight = topMargin + contentHeight;
    
    // Set canvas dimensions
    canvas.width = maxWidth * pixelRatio;
    canvas.height = totalHeight * pixelRatio;
    
    // Scale context
    ctx.scale(pixelRatio, pixelRatio);
    
    // Clear canvas
    ctx.clearRect(0, 0, maxWidth, totalHeight);
    
    // Create gradient background for document-like appearance
    const gradient = ctx.createLinearGradient(0, 0, maxWidth, 0);
    gradient.addColorStop(0, '#4f2913'); // Dark brown from theme
    gradient.addColorStop(0.4, '#7b5131'); // Lighter shade
    gradient.addColorStop(1, '#4f2913'); // Back to dark brown
    
    // Draw background with document styling
    ctx.fillStyle = gradient;
    roundRect(ctx, 0, 0, maxWidth, totalHeight, 20);
    
    // Add a "paper" inset for document appearance
    ctx.fillStyle = '#f1e8d8'; // Light cream color
    roundRect(ctx, 80, topMargin, maxWidth - 160, contentHeight - 40, 12);
    
    // Add thick border around the paper
    ctx.strokeStyle = '#7b8c43'; // Olive green from theme (same as binding color)
    ctx.lineWidth = 8;
    ctx.beginPath();
    roundRect(ctx, 80, topMargin, maxWidth - 160, contentHeight - 40, 12);
    ctx.stroke();
    
    // Add subtle shadow to the paper
    ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
    ctx.shadowBlur = 15;
    ctx.shadowOffsetX = 5;
    ctx.shadowOffsetY = 5;
    
    // Add decorative elements for document appearance
    // Top document binding
    ctx.fillStyle = '#7b8c43'; // Olive green from theme
    ctx.fillRect(120, 8, maxWidth - 240, 24);
    
    // Document binding holes
    const holePositions = [maxWidth * 0.2, maxWidth * 0.4, maxWidth * 0.6, maxWidth * 0.8];
    holePositions.forEach(x => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.beginPath();
      ctx.arc(x, 20, 10, 0, Math.PI * 2);
      ctx.fill();
    });
    
    // Reset shadow for text
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    
    // Draw content with document styling
    let yPos = topMargin + 40; // Start below top margin
    
    // Draw content with different styles for headings, lists, etc.
    allWrappedText.forEach((item, idx) => {
      const {text, style} = item;
      
      ctx.font = getMarkdownFont(style);
      
      if (style === 'heading1') {
        ctx.fillStyle = '#4f2913'; // Dark brown
        ctx.font = 'bold 32px "Arial Black", Arial, sans-serif';
        // Center the title
        ctx.textAlign = 'center';
        ctx.fillText(text, maxWidth / 2, yPos);
        ctx.textAlign = 'left'; // Reset for other text
        yPos += 10; // Extra space after title
      } else if (style === 'heading2') {
        ctx.fillStyle = '#7b5131'; // Medium brown
        yPos += 3; // Extra space before subheadings
      } else if (style === 'bold') {
        ctx.fillStyle = '#4f2913'; // Dark brown
      } else if (style === 'italic') {
        ctx.fillStyle = '#7b5131'; // Medium brown
        ctx.font = 'italic 18px Arial';
        // Right-align "View more..."
        ctx.textAlign = 'right';
        ctx.fillText(text, maxWidth - 120, yPos);
        ctx.textAlign = 'left'; // Reset alignment
      } else if (style === 'code') {
        ctx.fillStyle = '#4f2913'; // Dark brown
        // Draw code background
        ctx.fillStyle = 'rgba(211, 191, 169, 0.5)'; // Light beige for code blocks
        roundRect(ctx, 100, yPos - 5, maxWidth - 200, lineHeight + 10, 4);
        ctx.fillStyle = '#4f2913'; // Dark brown text
      } else if (style === 'list') {
        ctx.fillStyle = '#4f2913'; // Dark brown
        // Draw bullet point
        ctx.fillText('•', 100, yPos);
        ctx.fillText(text, 120, yPos); // Indented text
        yPos += lineHeight;
        return; // Skip normal rendering below
      } else {
        ctx.fillStyle = '#333333'; // Dark gray for normal text
      }
      
      if (text && style !== 'heading1' && style !== 'italic') {
        ctx.fillText(text, 100, yPos);
      }
      
      yPos += lineHeight;
    });
    
    // Add a small document footer with page number styling
    ctx.fillStyle = '#7b5131'; // Medium brown
    ctx.font = 'italic 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`Document: ${sourceName}`, maxWidth / 2, totalHeight - 20);
    
    // Create sprite with the canvas texture
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ 
      map: texture,
      transparent: true 
    });
    
    const sprite = new THREE.Sprite(material);
    
    // Set scale - different proportion than remark nodes
    const aspectRatio = totalHeight / maxWidth;
    const scale = node.val * 1.25;
    sprite.scale.set(scale, scale * aspectRatio, 1);
    
    return sprite;
  };
  
  // Create a specialized node for tabular data
  const createTableNode = (node: any, sourceName: string, content: string) => {
    // Create a custom HTML-like canvas for the table content
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    const pixelRatio = window.devicePixelRatio || 1;
    
    // Set up constants for table rendering
    const maxWidth = 4800; // Doubled
    const lineHeight = 96; // Doubled
    const cellPadding = 40; // Doubled
    const headerHeight = 200; // Doubled
    const footerHeight = 120;
    
    // Parse the table data
    const tableData = parseTableData(content);
    
    // Calculate dimensions based on table structure
    const columnWidths = calculateColumnWidths(ctx, tableData);
    const tableWidth = Math.min(maxWidth, columnWidths.reduce((sum, width) => sum + width, 0) + (columnWidths.length + 1) * cellPadding);
    const tableHeight = tableData.length * lineHeight;
    
    const totalHeight = headerHeight + tableHeight + footerHeight;
    
    // Set canvas dimensions
    canvas.width = tableWidth * pixelRatio;
    canvas.height = totalHeight * pixelRatio;
    
    // Scale context
    ctx.scale(pixelRatio, pixelRatio);
    
    // Clear canvas
    ctx.clearRect(0, 0, tableWidth, totalHeight);
    
    // Draw background - solid black
    ctx.fillStyle = '#000000'; // Pure black, no opacity
    roundRect(ctx, 0, 0, tableWidth, totalHeight, 16); // Same size
    
    // Draw header
    ctx.fillStyle = node.color || '#305f85';
    roundRect(ctx, 0, 0, tableWidth, headerHeight, { tl: 16, tr: 16, bl: 0, br: 0 });
    
    // Draw header text
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 28px Arial'; // Larger header text for tables
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`Source: ${sourceName}`, tableWidth / 2, headerHeight / 2);
    
    // Draw table
    let yPos = headerHeight + 10;
    
    // Draw table header row with different styling
    if (tableData.length > 0) {
      ctx.fillStyle = 'rgba(60, 80, 120, 0.7)'; // Header row background
      ctx.fillRect(cellPadding, yPos, tableWidth - 2 * cellPadding, lineHeight);
      
      // Draw header cells
      let xPos = cellPadding * 2;
      ctx.font = 'bold 20px Arial';
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      
      tableData[0].forEach((cell, colIndex) => {
        // Draw the cell text
        ctx.fillText(cell, xPos, yPos + lineHeight / 2);
        xPos += columnWidths[colIndex] + cellPadding;
      });
      
      yPos += lineHeight + 5; // Extra space after header row
    }
    
    // Draw data rows
    for (let rowIndex = 1; rowIndex < tableData.length; rowIndex++) {
      // Alternate row background for readability
      if (rowIndex % 2 === 0) {
        ctx.fillStyle = 'rgba(50, 50, 60, 0.4)';
        ctx.fillRect(cellPadding, yPos, tableWidth - 2 * cellPadding, lineHeight);
      }
      
      let xPos = cellPadding * 2;
      ctx.font = '18px Arial';
      ctx.fillStyle = '#EEEEEE';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      
      tableData[rowIndex].forEach((cell, colIndex) => {
        // Use bold for numeric values
        if (/^-?[\d,.]+%?$/.test(cell.trim())) {
          ctx.font = 'bold 18px Arial';
        } else {
          ctx.font = '18px Arial';
        }
        
        // Draw the cell text
        ctx.fillText(cell, xPos, yPos + lineHeight / 2);
        xPos += columnWidths[colIndex] + cellPadding;
      });
      
      yPos += lineHeight;
    }
    
    // Create sprite with the canvas texture
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ 
      map: texture,
      transparent: true 
    });
    
    const sprite = new THREE.Sprite(material);
    
    // Set scale - MUCH larger for better readability of tables
    const aspectRatio = totalHeight / tableWidth;
    const scale = node.val * 1.92;  // Doubled
    sprite.scale.set(scale, scale * aspectRatio, 1);
    
    return sprite;
  };
  
  // Parse table data from markdown content
  const parseTableData = (content: string): string[][] => {
    const rows: string[][] = [];
    
    // Split by lines
    const lines = content.split('\n');
    
    // Process each line
    for (const line of lines) {
      // Skip empty lines
      if (!line.trim()) continue;
      
      // If it's a table row (contains pipe characters)
      if (line.includes('|')) {
        // Split by pipe, remove empty cells from ends, and trim each cell
        const cells = line.split('|')
                         .filter((cell, i, arr) => i !== 0 && i !== arr.length - 1 || cell.trim().length > 0)
                         .map(cell => cell.trim());
        
        // Skip separator rows (contains only dashes and pipes)
        if (cells.every(cell => /^[-:]+$/.test(cell))) continue;
        
        rows.push(cells);
      } else if (rows.length > 0) {
        // If we already started parsing a table, add this as a row with one cell
        rows.push([line.trim()]);
      }
    }
    
    return rows;
  };
  
  // Calculate optimal column widths based on content
  const calculateColumnWidths = (ctx: CanvasRenderingContext2D, tableData: string[][]): number[] => {
    if (tableData.length === 0) return [];
    
    // Initialize array to track max width of each column
    const maxColumnCount = tableData.reduce((max, row) => Math.max(max, row.length), 0);
    const columnWidths = Array(maxColumnCount).fill(0);
    
    // Measure each cell and find the maximum width for each column
    tableData.forEach(row => {
      row.forEach((cell, colIndex) => {
        if (colIndex < maxColumnCount) {
          ctx.font = /^-?[\d,.]+%?$/.test(cell.trim()) ? 'bold 18px Arial' : '18px Arial';
          const cellWidth = ctx.measureText(cell).width;
          columnWidths[colIndex] = Math.max(columnWidths[colIndex], cellWidth);
        }
      });
    });
    
    // Ensure minimum column width and cap maximum width
    return columnWidths.map(width => Math.min(Math.max(width, 40), 300));
  };
  
  // Helper function to parse simple markdown into styled segments
  const parseSimpleMarkdown = (text: string): Array<{text: string, style: string}> => {
    const result: Array<{text: string, style: string}> = [];
    
    // Split by paragraphs
    const paragraphs = text.split('\n\n');
    
    paragraphs.forEach(para => {
      para = para.trim();
      if (!para) return;
      
      // Check for headings
      if (para.startsWith('# ')) {
        result.push({
          text: para.substring(2),
          style: 'heading1'
        });
      } else if (para.startsWith('## ')) {
        result.push({
          text: para.substring(3),
          style: 'heading2'
        });
      } else if (para.startsWith('```')) {
        // Handle code blocks
        const code = para.replace(/```/g, '').trim();
        result.push({
          text: code,
          style: 'code'
        });
      } else if (para.startsWith('- ') || para.startsWith('* ')) {
        // Handle list items
        const items = para.split('\n');
        items.forEach(item => {
          if (item.startsWith('- ') || item.startsWith('* ')) {
            result.push({
              text: item.substring(2),
              style: 'list'
            });
          }
        });
      } else {
        // Handle bold text within paragraphs
        if (para.includes('**') || para.includes('__')) {
          // Simple handling - just mark the whole paragraph as bold if it contains bold markers
          result.push({
            text: para.replace(/\*\*/g, '').replace(/__/g, ''),
            style: para.includes('**') || para.includes('__') ? 'bold' : 'normal'
          });
        } else {
          // Normal paragraph
          result.push({
            text: para,
            style: 'normal'
          });
        }
      }
    });
    
    return result;
  };
  
  // Helper function to get appropriate font based on markdown style
  const getMarkdownFont = (style: string): string => {
    switch(style) {
      case 'heading1':
        return 'bold 24px Arial';
      case 'heading2':
        return 'bold 22px Arial';
      case 'bold':
        return 'bold 18px Arial';
      case 'code':
        return '18px Courier New';
      case 'list':
        return '18px Arial';
      default:
        return '18px Arial';
    }
  };
  
  // Create a standard text node for selected text
  const createStandardTextNode = (node: any, color: string) => {
    const fontSize = 56; // Increased font size for better readability
    
    // Create a sprite with text texture
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: new THREE.CanvasTexture(createTextCanvas(node.name, fontSize, color)),
        transparent: true
      })
    );
    
    // Scale sprite based on node size - improved proportions for more professional look
    const scale = node.val / 1.0;
    sprite.scale.set(scale * 7.2, scale * 1.8, 1); // More balanced width-to-height ratio
    
    return sprite;
  };
  
  // Helper function to create a standard text canvas
  const createTextCanvas = (text: string, fontSize: number, color: string) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    const pixelRatio = window.devicePixelRatio || 1;
    
    // Set canvas dimensions based on text length - wider canvas with better proportions
    const textLength = text.length;
    const width = Math.max(600, textLength * fontSize * 0.8);
    
    canvas.width = width * pixelRatio;
    canvas.height = 120 * pixelRatio; // Increased height for better vertical spacing
    
    ctx.scale(pixelRatio, pixelRatio);
    
    // Make the canvas transparent
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Get the actual text width for sizing the background
    ctx.font = `${fontSize}px "Arial Black", Arial, sans-serif`;
    
    // Try to render the full text first
    let displayText = text;
    let actualTextWidth = ctx.measureText(text).width + 80; // Added more horizontal padding
    
    // If text is very long, limit the width but keep as much text as possible
    if (actualTextWidth > width * 0.9) {
      // Find how many characters we can fit
      let fitChars = 0;
      let currWidth = 0;
      
      for (let i = 0; i < text.length; i++) {
        const charWidth = ctx.measureText(text[i]).width;
        if (currWidth + charWidth + ctx.measureText('...').width > width * 0.9) {
          break;
        }
        currWidth += charWidth;
        fitChars++;
      }
      
      // Truncate and add ellipsis
      displayText = text.substring(0, fitChars) + '...';
      actualTextWidth = ctx.measureText(displayText).width + 80;
    }
    
    // Draw background with rounded corners - better gradient background
    const radius = 20; // Larger radius for more modern look
    
    // Create gradient background
    const gradient = ctx.createLinearGradient(0, 0, actualTextWidth, canvas.height / pixelRatio);
    gradient.addColorStop(0, '#4f2913'); // Dark brown from theme
    gradient.addColorStop(1, '#7b5131'); // Lighter shade for dimension
    
    ctx.fillStyle = gradient;
    
    // Draw rounded rectangle with better proportions
    roundRect(ctx, 0, 0, actualTextWidth, canvas.height / pixelRatio, radius);
    
    // Add subtle border for depth
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0 + radius, 0);
    ctx.lineTo(actualTextWidth - radius, 0);
    ctx.quadraticCurveTo(actualTextWidth, 0, actualTextWidth, 0 + radius);
    ctx.lineTo(actualTextWidth, canvas.height / pixelRatio - radius);
    ctx.quadraticCurveTo(actualTextWidth, canvas.height / pixelRatio, actualTextWidth - radius, canvas.height / pixelRatio);
    ctx.lineTo(0 + radius, canvas.height / pixelRatio);
    ctx.quadraticCurveTo(0, canvas.height / pixelRatio, 0, canvas.height / pixelRatio - radius);
    ctx.lineTo(0, 0 + radius);
    ctx.quadraticCurveTo(0, 0, 0 + radius, 0);
    ctx.closePath();
    ctx.stroke();
    
    // Prepare text style - using Arial Black for more professional appearance
    ctx.font = `${fontSize}px "Arial Black", Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Add subtle text shadow for better readability
    ctx.shadowColor = 'rgba(0,0,0,0.6)';
    ctx.shadowBlur = 4;
    ctx.shadowOffsetX = 2;
    ctx.shadowOffsetY = 2;
    
    // Draw text with color
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(displayText, actualTextWidth / 2, canvas.height / (2 * pixelRatio));
    
    return canvas;
  };
  
  // Helper function to wrap text
  const wrapText = (ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] => {
    const words = text.split(' ');
    const lines: string[] = [];
    let currentLine = '';

    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      const width = ctx.measureText(currentLine + word + ' ').width;
      
      if (width < maxWidth) {
        currentLine += word + ' ';
      } else {
        lines.push(currentLine.trim());
        currentLine = word + ' ';
      }
    }
    
    // Push the last line
    lines.push(currentLine.trim());
    
    return lines;
  };
  
  // Helper function for drawing rounded rectangles
  const roundRect = (
    ctx: CanvasRenderingContext2D, 
    x: number, 
    y: number, 
    width: number, 
    height: number, 
    radius: number | { tl: number, tr: number, bl: number, br: number }
  ) => {
    if (typeof radius === 'number') {
      radius = { tl: radius, tr: radius, br: radius, bl: radius };
    }
    
    ctx.beginPath();
    ctx.moveTo(x + radius.tl, y);
    ctx.lineTo(x + width - radius.tr, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius.tr);
    ctx.lineTo(x + width, y + height - radius.br);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius.br, y + height);
    ctx.lineTo(x + radius.bl, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius.bl);
    ctx.lineTo(x, y + radius.tl);
    ctx.quadraticCurveTo(x, y, x + radius.tl, y);
    ctx.closePath();
    ctx.fill();
  };
  
  // Get node size for 3D visualization - BIGGER
  const getNodeSize = (node: any) => {
    // Different sizing for different node types
    switch(node.group) {
      case 'detail':
        return node.val * 40.5; // 3x bigger alternative remarks (was 13.5)
      case 'source':
        return node.val * 20.5; // 20% smaller than before (was 25.6)
      case 'selected':
        return node.val * 4.8; // Keep selected text the same
      default:
        return node.val * 4.0; // Keep default the same
    }
  };
  
  // Get link color for 3D visualization
  const getLinkColor = (link: any) => {
    // Make links more visible with higher opacity
    if (link.color) {
      // If it's an rgba color, increase opacity
      if (link.color.startsWith('rgba')) {
        return link.color.replace(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*[\d.]+\)/, 'rgba($1, $2, $3, 0.5)');
      }
      return link.color;
    }
    return 'rgba(120, 120, 120, 0.5)'; // More visible edges with higher opacity and darker color
  };
  
  // Get link width for 3D visualization - made thicker for visibility
  const getLinkWidth = (link: any) => {
    return link.value * 2.0 + 0.5; // Increased width for better visibility
  };
  
  // Render the 3D force graph
  const render3DGraph = () => {
    if (!graphData?.graph_data) return null;
    
    // Set background color to match our theme
    const backgroundColor = '#f1e8d8'; // Light cream background from logo
    
    return (
      <div className="graph-3d-container">
        <ForceGraph3D
          ref={fgRef}
          graphData={graphData.graph_data}
          nodeLabel={null} // Disable default labels since we show content directly
          nodeColor={(node: any) => node.color}
          nodeVal={getNodeSize}
          nodeThreeObject={nodeThreeObject}
          nodeThreeObjectExtend={false}
          linkColor={getLinkColor}
          linkWidth={getLinkWidth}
          linkDirectionalParticles={2} // Increased to 2 particles for better visibility
          linkDirectionalParticleSpeed={0.01} // Slowed down for better visibility
          linkDirectionalParticleWidth={(link: any) => link.value * 1.5} // Thicker particles
          linkDirectionalParticleColor={() => '#4f2913'} // Dark brown from logo for particles
          onNodeClick={handleNodeClick}
          onNodeHover={null} // Disable hover effects
          backgroundColor={backgroundColor}
          width={window.innerWidth * 0.9} // Use more screen space
          height={window.innerHeight * 0.82} // Taller for better visibility
          controlType="orbit"
          enableNodeDrag={true}
          enableNavigationControls={true}
          showNavInfo={false}
          nodeResolution={32} // Higher resolution for smoother text
          warmupTicks={200} // More simulation steps for better layout
          cooldownTicks={200}
          d3AlphaDecay={0.008} // Even slower decay for better layout
          d3VelocityDecay={0.15} // Less friction for more natural movement
          d3AlphaMin={0.001} // Lower minimum alpha for better stabilization
          onBackgroundClick={(e: any) => {
            if (e && e.srcEvent) {
              e.srcEvent.stopPropagation();
              e.srcEvent.preventDefault();
            }
            setIsInteractingWithGraph(true);
            // Reset after a short delay to allow event to complete
            setTimeout(() => setIsInteractingWithGraph(false), 100);
          }}
          onEngineStop={() => {
            // Reset interaction flag when graph simulation stops
            setTimeout(() => setIsInteractingWithGraph(false), 100);
          }}
        />
      </div>
    );
  };
  
  // Handle model change
  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedModel(e.target.value);
  };

  // Handle chunk size change
  const handleChunkSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value > 0) {
      setChunkSize(value);
    }
  };

  // Handle line constraint change
  const handleLineConstraintChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    setLineConstraint(value === "null" ? null : parseInt(value));
  };

  // Function to apply paragraph highlighting based on graph data - now disabled since we don't want to highlight paragraphs
  const applyParagraphHighlights = useCallback(() => {
    // Clear existing highlights if highlight layer exists
    if (highlightLayerRef.current) {
      highlightLayerRef.current.innerHTML = '';
    }
    
    // We no longer highlight paragraphs in the document,
    // instead we provide alternative remarks in the graph
    return;
  }, [graphData, isLoading]);
  
  // Trigger highlighting when graph data changes
  useEffect(() => {
    if (graphData) {
      applyParagraphHighlights();
    }
  }, [graphData, applyParagraphHighlights]);

  // Add new JSX component for relevance legend
  const RelevanceLegend = () => (
    <div className="relevance-legend">
      <h4>Alternative Remarks</h4>
      <div className="legend-items">
        <div className="legend-item">
          <div className="legend-color relevance-high"></div>
          <div className="legend-label">High Relevance (80-100%)</div>
        </div>
        <div className="legend-item">
          <div className="legend-color relevance-medium"></div>
          <div className="legend-label">Medium Relevance (50-80%)</div>
        </div>
        <div className="legend-item">
          <div className="legend-color relevance-low"></div>
          <div className="legend-label">Low Relevance (0-50%)</div>
        </div>
      </div>
    </div>
  );

  interface ContextMenuProps {
    hoveredNode: GraphNode | null;
    onNodeSelect: (node: GraphNode) => void;
    position: { x: number; y: number };
    onClose: () => void;
    centerPosition?: { x: number; y: number };
  }

  const ContextMenu = ({ hoveredNode, onNodeSelect, position, onClose, centerPosition }: ContextMenuProps) => {
    // Don't show the context menu if the hovered node doesn't exist or has no content
    if (!hoveredNode || !hoveredNode.content) return null;

    // Determine if this is a source, detail, or selected node
    const isSource = hoveredNode.group === 'source';
    const isDetail = hoveredNode.group === 'detail';
    const isSelected = hoveredNode.group === 'selected';

    const maxContentLength = 350;
    const content = hoveredNode.content.length > maxContentLength
      ? hoveredNode.content.substring(0, maxContentLength) + '...'
      : hoveredNode.content;

    // Calculate position for the context menu
    const menuStyle = {
      left: `${position.x}px`,
      top: `${position.y}px`,
    };

    return (
      <div className="graph-context-menu" style={menuStyle}>
        <div className="context-menu-header">
          <h3>{
            hoveredNode.name?.startsWith('Alternative Remark') ? 
            'Alternative Remark' + hoveredNode.name.substring('Alternative Remark'.length) : 
            hoveredNode.name
          }</h3>
          <button className="close-button" onClick={onClose}>×</button>
        </div>
        <div className="context-menu-content">
          {isDetail && (
            <div className="detail-content-container">
              <p><strong>Alternative Remark</strong> {hoveredNode.relevance ? `(Relevance: ${Math.round(hoveredNode.relevance * 100)}%)` : ''}</p>
              <div className="content-preview" dangerouslySetInnerHTML={{ __html: marked.parse(content) }} />
              {hoveredNode.reason && (
                <div className="remark-reason">
                  <p><strong>Reasoning:</strong> {hoveredNode.reason}</p>
                </div>
              )}
            </div>
          )}
          {isSource && (
            <div className="source-content-container">
              <p><strong>Source Document:</strong></p>
              <div className="content-preview" dangerouslySetInnerHTML={{ __html: marked.parse(content) }} />
            </div>
          )}
          {isSelected && (
            <div className="selected-content-container">
              <p><strong>Your Selection:</strong></p>
              <div className="content-preview">{content}</div>
            </div>
          )}
          <div className="context-menu-actions">
            <button className="view-detail-btn" onClick={() => onNodeSelect(hoveredNode)}>View Full Content</button>
          </div>
        </div>
      </div>
    );
  };

  // Handle sending a chat message
  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return;
    
    // Add user message to chat
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: currentMessage,
      sender: 'user',
      timestamp: new Date()
    };
    
    setChatMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsChatLoading(true);
    
    try {
      // Get the full document content from markdown
      const documentContent = markdown;
      
      console.log(`Sending chat message: "${currentMessage}" using model: ${selectedModel}, chunk size: ${chunkSize}, line constraint: ${lineConstraint}`);
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      
      // Format chat history for the API
      const chatHistoryForAPI = chatMessages.map(msg => ({
        role: msg.sender === 'user' ? 'user' : 'assistant',
        content: msg.content
      }));
      
      const response = await axios.post(`${apiUrl}/api/chat-rfq`, { 
        text: currentMessage,
        document_content: documentContent,
        chat_history: chatHistoryForAPI,
        model: selectedModel,
        chunk_size: chunkSize,
        line_constraint: lineConstraint
      });
      
      console.log('Chat response received:', response.data);
      
      // Add special logging for web search 
      if (response.data.web_search_used) {
        console.log('%c🔎 WEB SEARCH WAS USED IN THIS RESPONSE 🔎', 'background: #4f2913; color: #f1e8d8; padding: 5px; font-size: 14px; font-weight: bold;');
      }
      
      // Add AI response to chat
      const aiMessage: ChatMessage = {
        id: Date.now().toString(),
        content: response.data.response,
        sender: 'ai',
        timestamp: new Date(),
        webSearchUsed: response.data.web_search_used || false,
        webSearchResults: response.data.web_search_results || undefined
      };
      
      setChatMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error getting chat response:', error);
      let errorMessage = "Sorry, I encountered an error processing your request.";
      
      if (axios.isAxiosError(error) && error.response) {
        errorMessage = `Error: ${error.response.data.detail || error.message}`;
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }
      
      // Add error message to chat
      const errorAiMessage: ChatMessage = {
        id: Date.now().toString(),
        content: errorMessage,
        sender: 'ai',
        timestamp: new Date()
      };
      
      setChatMessages(prev => [...prev, errorAiMessage]);
    } finally {
      setIsChatLoading(false);
    }
  };
  
  // Handle key press in chat input
  const handleChatKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (currentMessage.trim()) {
        handleSendMessage();
      }
    }
  };
  
  // Scroll to bottom of chat when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages]);
  
  // Render web search results
  const WebSearchResultsDisplay = ({ results }: { 
    results: {
      answer?: string;
      results?: Array<{
        title: string;
        content: string;
        url: string;
        score: number;
      }>;
      provider?: string;
    } 
  }) => {
    if (!results || (!results.results?.length && !results.answer)) {
      return null;
    }

    return (
      <div className="web-search-results">
        <div className="web-search-header">
          <span className="web-search-icon">🔎</span>
          <span className="web-search-title">Web Search Results</span>
          {results.provider && (
            <span className="web-search-provider">via {results.provider}</span>
          )}
        </div>
        
        {results.answer && (
          <div className="web-search-answer">
            <p><strong>Summary:</strong> {results.answer}</p>
          </div>
        )}
        
        {results.results && results.results.length > 0 && (
          <div className="web-search-sources">
            <p><strong>Sources:</strong></p>
            <ul>
              {results.results.map((result, index) => (
                <li key={index}>
                  <a href={result.url} target="_blank" rel="noopener noreferrer">
                    {result.title || 'Untitled Source'}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };
  
  // Render chat message
  const renderChatMessage = (message: ChatMessage) => {
    // Format message timestamps
    const formatTime = (date: Date) => {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };
    
    return (
      <motion.div 
        key={message.id}
        className={`chat-message ${message.sender === 'user' ? 'user-message' : 'ai-message'}`}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="message-content">
          <div 
            className="message-text"
            dangerouslySetInnerHTML={{ 
              __html: marked.parse(message.content)
            }}
          />
          {message.webSearchUsed && message.webSearchResults && (
            <WebSearchResultsDisplay results={message.webSearchResults} />
          )}
        </div>
        <div className="message-time">
          {formatTime(message.timestamp)}
          {message.webSearchUsed && (
            <span className="web-search-indicator" title="Web search was used for this response">
              {" "}🔎
            </span>
          )}
        </div>
      </motion.div>
    );
  };
  
  // Render the chat loading indicator with animation
  const renderChatLoading = () => (
    <motion.div 
      className="chat-loading"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="chat-loading-dots">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </motion.div>
  );

  const renderChatInterface = () => {
    return (
      <div className="chat-interface">
        <div className="chat-header">
          <h3>Chat with RfQ</h3>
          <div className="chat-features">
            {selectedModel && (
              <div className="chat-model-info">
                Using <span className="model-name">{selectedModel}</span>
              </div>
            )}
            <div className="chat-feature-info">
              <span className="feature-icon" title="Web search capability enabled">🔎</span>
              <span className="feature-text">Web Search</span>
            </div>
          </div>
        </div>
        
        <div className="chat-messages">
          {chatMessages.map(message => renderChatMessage(message))}
          
          {isChatLoading && renderChatLoading()}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="chat-input-container">
          <textarea
            ref={chatInputRef}
            className="chat-input"
            placeholder="Ask about this document or related external information..."
            value={currentMessage}
            onChange={(e) => setCurrentMessage(e.target.value)}
            onKeyDown={handleChatKeyPress}
            disabled={isChatLoading}
            rows={1}
            style={{ height: Math.min(100, Math.max(50, currentMessage.split('\n').length * 24)) }}
          />
          
          <button 
            className="chat-send-button" 
            onClick={handleSendMessage}
            disabled={isChatLoading || !currentMessage.trim()}
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="markdown-viewer">
      <div className="viewer-controls">
        <div className="control-group">
          <label htmlFor="mode-selector">Mode:</label>
          <select 
            id="mode-selector" 
            className="mode-selector"
            value={mode}
            onChange={handleModeChange}
          >
            <option value="char-counter">{ModeIcons['char-counter']} Character Counter</option>
            <option value="sample-graph">{ModeIcons['sample-graph']} Source Analysis</option>
            <option value="chat-rfq">{ModeIcons['chat-rfq']} Chat with your RfQ</option>
            <option value="nothing">{ModeIcons['nothing']} No Action</option>
          </select>
        </div>
        
        {(mode === 'sample-graph' || mode === 'chat-rfq') && (
          <>
            <div className="control-group">
              <label htmlFor="model-selector">AI Model:</label>
              <select 
                id="model-selector" 
                className="model-selector"
                value={selectedModel}
                onChange={handleModelChange}
                disabled={isLoadingModels || isLoading || isChatLoading}
              >
                {isLoadingModels && (
                  <option value="">⏳ Loading models...</option>
                )}
                
                {aiModels.openai_models.length > 0 && (
                  <optgroup label="OpenAI Models">
                    {aiModels.openai_models.map(model => (
                      <option key={model} value={model}>🤖 {model}</option>
                    ))}
                  </optgroup>
                )}
                
                {aiModels.ollama_models.length > 0 && (
                  <optgroup label="Local Models">
                    {aiModels.ollama_models.map(model => (
                      <option key={model} value={model}>💻 {model}</option>
                    ))}
                  </optgroup>
                )}
              </select>
            </div>
            
            <div className="control-group">
              <label htmlFor="chunk-size">Chunk Size:</label>
              <input 
                id="chunk-size"
                type="number" 
                className="chunk-size-input"
                value={chunkSize}
                onChange={handleChunkSizeChange}
                min="1"
                max="20"
                disabled={isLoading || isChatLoading}
              />
            </div>
            
            <div className="control-group">
              <label htmlFor="line-constraint">Line Limit:</label>
              <select
                id="line-constraint"
                className="line-constraint-selector"
                value={lineConstraint === null ? "null" : lineConstraint.toString()}
                onChange={handleLineConstraintChange}
                disabled={isLoading || isChatLoading}
              >
                <option value="null">No limit</option>
                <option value="100">100 lines</option>
                <option value="500">500 lines</option>
                <option value="1000">1000 lines</option>
                <option value="5000">5000 lines</option>
              </select>
            </div>
          </>
        )}
      </div>
      
      <div className="viewer-layout">
        <div 
          ref={contentContainerRef}
          className={`content-container ${isChatVisible ? 'with-chat' : ''}`}
          onClick={() => {
            if (isPopupVisible || isGraphVisible) {
              // Close popup when clicking outside
              if (!isInteractingWithGraph) {
                closePopup();
              }
            }
          }}
        >
          <div
            ref={markdownContentRef}
            className="markdown-content"
            onMouseUp={handleMouseUp}
            onClick={handleClick}
            dangerouslySetInnerHTML={{ __html: marked(markdown) }}
          />
          
          {/* Layer for custom highlights */}
          <div ref={highlightLayerRef} className="highlight-layer"></div>
          
          {/* Loading overlay */}
          {isLoading && (
            <div className="loading-overlay" style={{ zIndex: 9999 }}>
              <div className="spinner"></div>
              <p>Analyzing sources and generating alternative remarks...</p>
            </div>
          )}
        </div>
        
        {/* Chat sidebar */}
        {isChatVisible && (
          <div className="chat-container">
            {renderChatInterface()}
          </div>
        )}
      </div>
      
      <AnimatePresence>
        {isPopupVisible && popupPosition && characterCount !== null && mode === 'char-counter' && (
          <motion.div 
            className="character-popup"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0, transition: { duration: 0.3 } }}
            exit={{ opacity: 0, y: -10 }}
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
        
        {isGraphVisible && graphData && mode === 'sample-graph' && (
          <motion.div 
            className="graph-popup"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1, transition: { duration: 0.4 } }}
            exit={{ opacity: 0, scale: 0.9 }}
            onClick={(e) => {
              // Prevent clicks inside the popup from closing it
              e.stopPropagation();
            }}
          >
            <div className="graph-header">
              <h3>Source Analysis</h3>
              <p className="selected-text" title={graphData.selected_text}>"{graphData.selected_text}"</p>
              {graphData.selected_text.length > 80 && (
                <details className="full-text-details">
                  <summary>Show full text</summary>
                  <p className="full-selected-text">"{graphData.selected_text}"</p>
                </details>
              )}
              
              {graphData.model_used && (
                <div className="model-info">
                  <span className="model-label">AI Model:</span> 
                  <span className="model-name">{graphData.model_used}</span>
                </div>
              )}
              
              {graphData.error && (
                <div className="graph-error">
                  <h4>Error</h4>
                  <p>{graphData.error}</p>
                </div>
              )}
            </div>
            
            {graphData.error ? (
              <div className="error-message">
                <p>An error occurred while processing your request. You can try:</p>
                <ul>
                  <li>Using a different AI model</li>
                  <li>Selecting a shorter text passage</li>
                  <li>Trying again later</li>
                </ul>
              </div>
            ) : (
              render3DGraph()
            )}
            
            <button 
              className="close-popup" 
              onClick={(e) => {
                e.stopPropagation();
                closePopup();
              }}
              aria-label="Close"
            >
              ✕
            </button>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Add relevance legend when graph data is available */}
      {graphData && isGraphVisible && mode === 'sample-graph' && (
        <RelevanceLegend />
      )}
    </div>
  );
};

export default MarkdownViewer; 