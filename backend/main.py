from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import random
import glob
import uuid
import json
import datetime
import requests
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from typing import Optional

# Load environment variables from .env file
load_dotenv(dotenv_path="../.env")

# Create logs directory if it doesn't exist
LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://host.docker.internal:11434")  # Default to Docker host for local Ollama

def count_tokens(text):
    """Count the number of tokens in a string."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except:
        # Fallback simple estimation (not as accurate)
        return len(text.split())

def log_gpt_interaction(prompt, response, prefix="gpt"):
    """Log GPT prompt and response to text files"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare prompt content
    if isinstance(prompt, dict):
        # Convert dict to JSON string
        prompt_content = json.dumps(prompt, indent=2)
    elif isinstance(prompt, list):
        # For message format
        prompt_content = "\n---\n".join([f"ROLE: {msg['role']}\nCONTENT: {msg['content']}" for msg in prompt])
    else:
        # For string format
        prompt_content = str(prompt)
    
    # Calculate token count
    token_count = count_tokens(str(prompt_content))
    
    # Create datetime-prefixed filename
    prompt_filename = f"{LOGS_DIR}/{timestamp}_{prefix}_prompt.txt"
    with open(prompt_filename, "w") as f:
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"TOKEN COUNT: {token_count}\n\n")
        f.write(prompt_content)
    
    # Save response with same datetime prefix
    response_filename = f"{LOGS_DIR}/{timestamp}_{prefix}_response.txt"
    with open(response_filename, "w") as f:
        f.write(f"TIMESTAMP: {timestamp}\n\n")
        if isinstance(response, dict):
            f.write(json.dumps(response, indent=2))
        else:
            f.write(str(response))
    
    print(f"Logged GPT interaction to {prompt_filename} and {response_filename}")
    
    return prompt_filename, response_filename

class TextSelection(BaseModel):
    text: str
    document_content: Optional[str] = None  # Full document content for context
    model: str = "gpt-4o"  # Default model
    chunk_size: int = 5  # Default chunk size for document processing
    line_constraint: Optional[int] = None  # Optional line constraint

class GraphResponse(BaseModel):
    sources: list
    opacities: list
    content: list

class ModelList(BaseModel):
    openai_models: list
    ollama_models: list

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://frontend:5173"],  # Both local and Docker
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/hello")
def read_root():
    return {"message": "Hello from FastAPI"}

@app.post("/api/count-characters")
def count_characters(selection: TextSelection):
    char_count = len(selection.text)
    return {"count": char_count}

@app.get("/api/models")
def list_models():
    """List available AI models from OpenAI and Ollama"""
    models = get_available_models()
    return models

@app.post("/api/sample-graph")
def sample_graph(selection: TextSelection):
    # Directory where structured input files are stored
    structured_input_dir = "/app/structured_input"
    
    # Fallback to local development path if Docker path doesn't exist
    if not os.path.exists(structured_input_dir):
        structured_input_dir = "structured_input"
        # Additional fallback in case we're running in a different working directory
        if not os.path.exists(structured_input_dir):
            structured_input_dir = "../structured_input"
            if not os.path.exists(structured_input_dir):
                print(f"Warning: structured_input directory not found. Tried: /app/structured_input, structured_input, ../structured_input")
    
    print(f"Using structured_input directory: {structured_input_dir}")
    print(f"Using AI model: {selection.model}")
    print(f"Using chunk size: {selection.chunk_size}")
    if selection.line_constraint:
        print(f"Limiting each file to {selection.line_constraint} lines")
    else:
        print("No line limit applied - using full files")
    
    # Find all markdown files in the directory
    md_files = glob.glob(f"{structured_input_dir}/**/*.md", recursive=True)
    
    # Generate nodes and links for force graph
    nodes = []
    links = []
    
    # Add the selected text as the central node
    selected_node_id = "selected-text"
    nodes.append({
        "id": selected_node_id,
        "name": selection.text[:50] + ("..." if len(selection.text) > 50 else ""),
        "val": 20,  # Size of the node
        "color": "#888888",  # Gray color for the central node
        "group": "selected",
        "fullText": selection.text
    })
    
    # If no files found, show an error
    if not md_files:
        return {
            "sources": ["Error: No structured data files found"],
            "opacities": [0.5],
            "content": ["Could not locate any .md files in the structured_input directory"],
            "selected_text": selection.text,
            "graph_data": {
                "nodes": nodes,
                "links": []
            }
        }
    
    # Check if we're using OpenAI or Ollama
    is_ollama = is_ollama_model(selection.model)
    
    # Initialize AI client based on the selected model
    if not is_ollama:
        # OpenAI models
        if not OPENAI_API_KEY:
            print("Warning: OpenAI API key not found. Check your .env file.")
            return {
                "sources": ["Error: OpenAI API key not found"],
                "opacities": [0.5],
                "content": ["Please check the .env file for a valid OpenAI API key"],
                "selected_text": selection.text,
                "graph_data": {
                    "nodes": nodes,
                    "links": []
                }
            }
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        # For Ollama models, we don't need a client instance
        # Just verify Ollama server is reachable
        try:
            response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=2)
            if response.status_code != 200:
                return {
                    "sources": ["Error: Ollama server not available"],
                    "opacities": [0.5],
                    "content": [f"Could not connect to Ollama server at {OLLAMA_API_URL}"],
                    "selected_text": selection.text,
                    "graph_data": {
                        "nodes": nodes,
                        "links": []
                    }
                }
        except Exception as e:
            return {
                "sources": ["Error: Ollama server not available"],
                "opacities": [0.5],
                "content": [f"Could not connect to Ollama server: {str(e)}"],
                "selected_text": selection.text,
                "graph_data": {
                    "nodes": nodes,
                    "links": []
                }
            }
    
    # Read file contents - provide full content for all files
    file_contents = []
    for file in md_files:
        try:
            with open(file, 'r') as f:
                content = f.read()
                
                # Apply line constraint if specified
                if selection.line_constraint is not None:
                    lines = content.split('\n')
                    # Take only the specified number of lines
                    limited_lines = lines[:selection.line_constraint]
                    content = '\n'.join(limited_lines)
                
                if content:  # Only add non-empty files
                    file_contents.append({"filename": file, "content": content})
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}")
            continue
    
    # Use AI to find relevant files and paragraphs
    relevant_files = []
    full_prompt_content = []
    full_response_content = []
    
    # Unique session ID for this request
    session_id = f"ai_session_{selection.model}_{uuid.uuid4().hex[:8]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # First find the most relevant files
        # Modified approach: Use all files with chunking to avoid token limits
        # The maximum number of files to include
        max_files = len(file_contents)  # Use all files

        # If we have a lot of documents, we need to process them in chunks
        # to avoid hitting context limits
        # Each chunk will have a set of documents that we'll analyze separately
        chunk_size = selection.chunk_size
        total_chunks = (max_files + chunk_size - 1) // chunk_size  # Ceiling division

        # Store all ranked documents across chunks
        all_ranked_documents = []

        # Process each chunk of documents
        for chunk_index in range(total_chunks):
            start_idx = chunk_index * chunk_size
            end_idx = min((chunk_index + 1) * chunk_size, max_files)
            
            # Get the current chunk of documents
            current_chunk = file_contents[start_idx:end_idx]
            
            # Prepare document summaries with full content for this chunk
            file_summaries = []
            for fc in current_chunk:
                # Use full content
                file_summaries.append(f"FILENAME: {fc['filename']}\nCONTENT: {fc['content']}")
            
            # Join summaries
            file_summaries_text = "\n\n".join(file_summaries)
            
            system_msg = """
            You are a specialized document retrieval system. Given a user query (which is a highlighted excerpt from a document) and a list of documents, 
            rank the documents based on semantic similarity and relevance to the highlighted excerpt.
            Respond with a JSON object with a 'ranked_documents' array containing objects with 'filename' and 'relevance_score' (0.0-1.0) properties.
            """
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"HIGHLIGHTED EXCERPT: {selection.text}\n\nAVAILABLE DOCUMENTS ({start_idx+1}-{end_idx} of {max_files}):\n{file_summaries_text}"}
            ]
            
            # Generate a unique prefix for this chunk's logs
            chunk_log_prefix = f"chunk_{chunk_index+1}_of_{total_chunks}"
            
            # Store this part of the conversation for full logging
            full_prompt_content.append({
                "section": f"document_ranking_{chunk_log_prefix}",
                "messages": messages
            })
            
            # Call the appropriate AI service
            if is_ollama:
                response = ollama_generate(messages, selection.model)
            else:
                response = client.chat.completions.create(
                    model=selection.model,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
            
            # Log this specific interaction directly
            log_gpt_interaction(
                messages, 
                response if is_ollama else response.model_dump(),
                prefix=f"{session_id}_{chunk_log_prefix}"
            )
            
            # Store this response for full logging
            full_response_content.append({
                "section": f"document_ranking_{chunk_log_prefix}",
                "response": response if is_ollama else response.model_dump()
            })
            
            # Extract the response content
            if is_ollama:
                response_content = response["choices"][0]["message"]["content"]
            else:
                response_content = response.choices[0].message.content
                
            # Extract the JSON response
            try:
                chunk_result = json.loads(response_content)
                if "ranked_documents" in chunk_result:
                    all_ranked_documents.extend(chunk_result["ranked_documents"])
            except json.JSONDecodeError:
                # If response isn't valid JSON, try to extract filenames using simple text analysis
                print(f"Warning: Failed to parse JSON response for chunk {chunk_index+1}")

        # After processing all chunks, sort all ranked documents by relevance
        if all_ranked_documents:
            all_ranked_documents.sort(key=lambda x: float(x["relevance_score"]), reverse=True)
            
            # Take the top 3 most relevant documents
            top_documents = all_ranked_documents[:3]
            
            # Create the list of relevant files from the top documents
            for doc in top_documents:
                filename = doc["filename"]
                relevance = float(doc["relevance_score"])
                # Find the matching file content
                for fc in file_contents:
                    if fc["filename"] == filename or filename in fc["filename"] or os.path.basename(fc["filename"]) == filename:
                        relevant_files.append((fc["filename"], relevance))
                        break
        
        # If no files found or error occurred, fall back to random selection
        if not relevant_files:
            relevant_files = [(file["filename"], random.uniform(0.3, 0.7)) for file in random.sample(file_contents, min(3, len(file_contents)))]
    
    except Exception as e:
        print(f"Error using AI API: {str(e)}")
        # Return the error to the frontend
        return {
            "error": str(e),
            "sources": [f"Error: {str(e)}"],
            "opacities": [0.5],
            "content": [f"An error occurred: {str(e)}"],
            "selected_text": selection.text,
            "graph_data": {
                "nodes": nodes,
                "links": []
            }
        }
    
    # Sort by relevance and take top 3
    relevant_files.sort(key=lambda x: x[1], reverse=True)
    selected_files = relevant_files[:3]
    
    # Now for each selected file, find the most relevant chunks within it
    sources = [os.path.basename(file) for file, _ in selected_files]
    influences = [min(score, 0.9) for _, score in selected_files]  # Convert relevance to opacity
    
    # Define color schemes for the nodes based on relevance
    color_schemes = [
        {"main": "rgba(255, 70, 70, ", "detail": "rgba(255, 120, 120, "},  # Red (most relevant)
        {"main": "rgba(255, 220, 70, ", "detail": "rgba(255, 230, 120, "},  # Yellow (second)
        {"main": "rgba(70, 200, 70, ", "detail": "rgba(120, 230, 120, "}    # Green (third)
    ]
    
    # Read content from files and create nodes/links
    for i, ((file_path, relevance), influence) in enumerate(zip(selected_files, influences)):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Use AI to find the most relevant paragraphs within the file
                try:
                    paragraphs = [p for p in content.split('\n\n') if p.strip()]
                    # Use all paragraphs without sampling
                    
                    # Calculate paragraph positions for highlighting
                    paragraph_positions = []
                    current_position = 0
                    for p in paragraphs:
                        start_pos = content.find(p, current_position)
                        if start_pos != -1:
                            end_pos = start_pos + len(p)
                            paragraph_positions.append((start_pos, end_pos))
                            current_position = end_pos
                        else:
                            paragraph_positions.append((-1, -1))  # Not found (shouldn't happen)
                    
                    # Join paragraphs with index markers - include all paragraphs
                    paragraphs_text = "\n".join([f"[{idx}] {p}" for idx, p in enumerate(paragraphs)])
                    
                    # Ask the AI to identify the most relevant paragraphs with document context
                    para_system_msg = """
                    You are a specialized text analysis system. Given a user query, full document content, and a highlighted excerpt from that document,
                    provide 2-5 alternative remarks or observations that can be made about the highlighted excerpt given the context of the full document.
                    
                    Each remark should offer a different perspective, insight, or alternative interpretation regarding the highlighted excerpt.
                    
                    For each alternative remark, also provide a brief reason explaining why this perspective is valuable 
                    or how it differs from the obvious interpretation.
                    
                    Respond with a JSON object with an 'alternative_remarks' property containing an array of objects, 
                    each with 'remark', 'reason', and 'relevance_score' (0.0-1.0) properties.
                    """
                    
                    # Use the document content provided from the frontend if available
                    document_content = selection.document_content if selection.document_content else content
                    
                    para_messages = [
                        {"role": "system", "content": para_system_msg},
                        {"role": "user", "content": f"QUERY: {selection.text}\n\nDocument: {os.path.basename(file_path)}\n\nFULL DOCUMENT CONTENT:\n{document_content}\n\nHIGHLIGHTED EXCERPT: {selection.text}\n\nProvide alternative remarks about the highlighted excerpt in the context of the document."}
                    ]
                    
                    # Generate a unique prefix for this paragraph analysis
                    para_log_prefix = f"para_file_{i}"
                    
                    # Store this part of the conversation for full logging
                    full_prompt_content.append({
                        "section": f"paragraph_ranking_{para_log_prefix}",
                        "file": os.path.basename(file_path),
                        "messages": para_messages
                    })
                    
                    # Call the appropriate AI service
                    if is_ollama:
                        para_response = ollama_generate(para_messages, selection.model)
                    else:
                        para_response = client.chat.completions.create(
                            model=selection.model,
                            messages=para_messages,
                            response_format={"type": "json_object"}
                        )
                    
                    # Log this specific paragraph analysis interaction
                    log_gpt_interaction(
                        para_messages,
                        para_response if is_ollama else para_response.model_dump(),
                        prefix=f"{session_id}_{para_log_prefix}"
                    )
                    
                    # Store this response for full logging
                    full_response_content.append({
                        "section": f"paragraph_ranking_{para_log_prefix}",
                        "file": os.path.basename(file_path),
                        "response": para_response if is_ollama else para_response.model_dump()
                    })
                    
                    # Extract the response content
                    if is_ollama:
                        para_response_content = para_response["choices"][0]["message"]["content"]
                    else:
                        para_response_content = para_response.choices[0].message.content
                    
                    # Extract the JSON response for paragraphs
                    try:
                        para_result = json.loads(para_response_content)
                    except json.JSONDecodeError:
                        # Fallback if JSON is invalid
                        para_result = {"alternative_remarks": []}
                        
                    relevant_paras = []
                    
                    if "alternative_remarks" in para_result:
                        for remark_info in para_result["alternative_remarks"]:
                            remark_text = remark_info["remark"]
                            remark_score = float(remark_info["relevance_score"])
                            remark_reason = remark_info.get("reason", "")
                            # Include reason in the tuple
                            relevant_paras.append((remark_text, remark_score, (-1, -1), remark_reason))
                    
                    # If no remarks found in the structured format, create some generic ones
                    if not relevant_paras:
                        relevant_paras = [
                            (f"This document appears to be related to the topic '{selection.text}'.", 0.8, (-1, -1), "Document content matches query topic."),
                            (f"Consider analyzing this document for additional context.", 0.6, (-1, -1), "This source might contain related information."),
                            (f"This source might provide relevant information about the query.", 0.7, (-1, -1), "Contains keywords related to the query.")
                        ]
                except Exception as e:
                    print(f"Error using AI API for paragraph selection: {str(e)}")
                    # Fallback - use first few paragraphs
                    paragraphs = [p for p in content.split('\n\n') if p.strip()]
                    paragraph_positions = []
                    current_position = 0
                    for p in paragraphs:
                        start_pos = content.find(p, current_position)
                        if start_pos != -1:
                            end_pos = start_pos + len(p)
                            paragraph_positions.append((start_pos, end_pos))
                            current_position = end_pos
                        else:
                            paragraph_positions.append((-1, -1))
                            
                    relevant_paras = [(p, (1.0 - idx*0.1), paragraph_positions[idx] if idx < len(paragraph_positions) else (-1, -1), f"Fallback reason {idx+1}: Could not analyze using AI.") 
                                     for idx, p in enumerate(paragraphs[:4]) if p.strip()]
                
                # Get the color scheme for this node based on relevance ranking
                color_scheme = color_schemes[min(i, len(color_schemes)-1)]
                
                # Create a source node for the file
                source_id = f"source-{i}"
                nodes.append({
                    "id": source_id,
                    "name": os.path.basename(file_path),
                    "val": 15,
                    "color": f"{color_scheme['main']}{influence + 0.3})",
                    "group": "source",
                    "content": content  # Full content, not truncated
                })
                
                # Link to selected text
                links.append({
                    "source": source_id,
                    "target": selected_node_id,
                    "value": influence,
                    "color": f"{color_scheme['main']}{influence})"
                })
                
                # Create paragraph sections as detail nodes
                for j, (para, para_relevance, para_position, para_reason) in enumerate(relevant_paras[:5]):  # Take up to 5 remarks
                    if para.strip():
                        detail_id = f"detail-{i}-{j}"
                        detail_influence = para_relevance * influence  # Scale by file relevance
                        
                        nodes.append({
                            "id": detail_id,
                            "name": f"Alternative Remark {j+1}",
                            "val": 8 + (para_relevance * 5),  # Size based on relevance
                            "color": f"{color_scheme['detail']}{detail_influence + 0.2})",
                            "group": "detail",
                            "content": para,  # Full remark text
                            "position": para_position,  # Not used for remarks but kept for compatibility
                            "relevance": para_relevance,  # Add raw relevance score
                            "reason": para_reason  # Add reason for the alternative remark
                        })
                        
                        # Link detail to source
                        links.append({
                            "source": detail_id,
                            "target": source_id,
                            "value": detail_influence,
                            "color": f"{color_scheme['main']}{detail_influence})"
                        })
        except Exception as e:
            # Add error node
            error_id = f"error-{i}"
            nodes.append({
                "id": error_id,
                "name": f"Error: {os.path.basename(file_path)}",
                "val": 10,
                "color": "rgba(200, 50, 50, 0.7)",
                "group": "error",
                "content": f"Error processing file: {str(e)}"
            })
            
            links.append({
                "source": error_id,
                "target": selected_node_id,
                "value": 0.1,
                "color": "rgba(200, 50, 50, 0.3)"
            })
    
    # Create a single log for the entire input and output
    # Combine all prompts into a single document
    combined_prompt = {
        "query": selection.text,
        "model": selection.model,
        "timestamp": datetime.datetime.now().isoformat(),
        "interactions": full_prompt_content
    }
    
    # Combine all responses into a single document
    combined_response = {
        "timestamp": datetime.datetime.now().isoformat(),
        "interactions": full_response_content
    }
    
    # Log the combined prompt and response
    log_gpt_interaction(combined_prompt, combined_response, prefix=session_id)
    
    # Prepare the response data
    return {
        "sources": sources,
        "opacities": influences,
        "content": [node["content"] for node in nodes if node["group"] == "source"],
        "selected_text": selection.text,  # Full text, not truncated
        "model_used": selection.model,
        "graph_data": {
            "nodes": nodes,
            "links": links
        }
    }

def get_available_models():
    """Get available models from both OpenAI and Ollama"""
    openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o-mini"]
    ollama_models = []
    
    # Try to fetch models from Ollama
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_data = response.json()
            if "models" in ollama_data:
                ollama_models = [model["name"] for model in ollama_data["models"]]
            else:
                # For newer Ollama versions
                ollama_models = [model["name"] for model in ollama_data.get("models", [])]
    except Exception as e:
        print(f"Error fetching Ollama models: {str(e)}")
    
    return {"openai_models": openai_models, "ollama_models": ollama_models}

def is_ollama_model(model_name):
    """Check if model is from Ollama (not starting with 'gpt')"""
    return not model_name.startswith("gpt")

def ollama_generate(messages, model, json_response=True):
    """Generate text using Ollama API"""
    # Convert OpenAI format messages to Ollama prompt
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    
    prompt += "<|assistant|>\n"
    
    # Prepare request for Ollama
    api_url = f"{OLLAMA_API_URL}/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 2048,
            "num_ctx": 128000  # Increase context window to maximum (gemma3 supports up to 128k tokens)
        }
    }
    
    if json_response:
        data["options"]["format"] = "json"
    
    # Make the API call
    response = requests.post(api_url, json=data)
    
    if response.status_code != 200:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
    
    # Parse the response
    result = response.json()
    
    # Format like OpenAI response for compatibility
    formatted_response = {
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": result.get("response", "")
                },
                "finish_reason": "stop"
            }
        ]
    }
    
    return formatted_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)