@app.post("/api/chat-rfq")
async def chat_rfq(request: Request):
    """
    Chat with the document content.
    """
    try:
        data = await request.json()
        
        # Get request parameters
        message = data.get("message", "")
        chat_history = data.get("chat_history", [])
        document_content = data.get("document_content", "")
        model = data.get("model", "gpt-4o")
        chunk_size = data.get("chunk_size", 5)
        line_constraint = data.get("line_constraint")
        
        logger.info(f"Chat RfQ request with model: {model}, chunk_size: {chunk_size}, line_constraint: {line_constraint}")
        
        # Prepare document content: limit lines if specified
        if line_constraint:
            document_content = "\n".join(document_content.split("\n")[:line_constraint])
            
        # Initialize LLM client based on model type
        client = None
        if model.startswith("gpt-"):
            from openai import OpenAI
            client = OpenAI()
            
            # Create system prompt
            system_prompt = f"""You are an expert assistant helping to analyze a Request for Quotation (RfQ) document.
            You should review the document carefully and answer questions about its content, requirements, specifications, deadlines, and other relevant details.
            Provide clear, concise, and accurate responses. When information is not available in the document, be honest about it.
            
            Document Content:
            ---
            {document_content}
            ---
            
            When analyzing this document:
            1. Identify key requirements, specifications, and constraints
            2. Consider important dates, deadlines, and milestones
            3. Look for technical specifications that need to be met
            4. Note any compliance and regulatory requirements
            5. Be attentive to evaluation criteria and selection processes
            """
            
            # Format chat history in OpenAI format
            messages = [{"role": "system", "content": system_prompt}]
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": message})
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2
            )
            
            response_text = response.choices[0].message.content
            
        else:  # Assume Ollama model
            import ollama
            
            # Create detailed prompt with document content
            context_prompt = f"""You are an expert assistant helping to analyze a Request for Quotation (RfQ) document.
            Document Content:
            ---
            {document_content}
            ---
            
            Chat History:
            """
            
            # Add chat history to the context
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                context_prompt += f"\n{role}: {msg['content']}"
            
            # Add the current message
            context_prompt += f"\n\nUser: {message}\n\nAssistant: "
            
            # Call Ollama API
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": context_prompt}],
                stream=False
            )
            
            response_text = response["message"]["content"]
        
        # Return the response
        return {"response": response_text, "model_used": model}
        
    except Exception as e:
        logger.error(f"Error in chat_rfq: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}") 