"""
Web Application Examples for Ollama
Chapter 12: Building Applications
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Web framework imports
try:
    from flask import Flask, request, jsonify, render_template_string, session
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("httpx not available. Install with: pip install httpx")


class OllamaWebClient:
    """Web client for interacting with Ollama API."""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.logger = logging.getLogger(__name__)
    
    async def generate_response(self, model: str, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """Generate response from Ollama."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": stream
                    }
                )
                response.raise_for_status()
                
                if stream:
                    return {"stream": response.iter_lines()}
                else:
                    return response.json()
                    
            except Exception as e:
                self.logger.error(f"Error generating response: {e}")
                raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.ollama_host}/api/tags")
                response.raise_for_status()
                data = response.json()
                return data.get("models", [])
            except Exception as e:
                self.logger.error(f"Error listing models: {e}")
                return []
    
    async def check_health(self) -> bool:
        """Check if Ollama is healthy."""
        try:
            models = await self.list_models()
            return True
        except Exception:
            return False


# Flask Web Application
if FLASK_AVAILABLE:
    
    class ChatWebApp:
        """Flask web application for Ollama chat interface."""
        
        def __init__(self, ollama_client: OllamaWebClient):
            self.app = Flask(__name__)
            self.app.secret_key = "ollama-chat-secret-key-change-in-production"
            self.ollama_client = ollama_client
            self.setup_routes()
        
        def setup_routes(self):
            """Setup Flask routes."""
            
            @self.app.route('/')
            def index():
                """Main chat interface."""
                return render_template_string(self.get_chat_template())
            
            @self.app.route('/api/models')
            async def get_models():
                """Get available models."""
                try:
                    models = await self.ollama_client.list_models()
                    return jsonify({"models": models})
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @self.app.route('/api/chat', methods=['POST'])
            async def chat():
                """Chat endpoint."""
                try:
                    data = request.get_json()
                    
                    if not data or 'message' not in data:
                        return jsonify({"error": "Message is required"}), 400
                    
                    message = data['message']
                    model = data.get('model', 'llama2')
                    
                    # Get conversation history from session
                    if 'conversation' not in session:
                        session['conversation'] = []
                    
                    # Build context from conversation history
                    context = ""
                    for msg in session['conversation'][-5:]:  # Last 5 messages
                        context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"
                    
                    full_prompt = f"{context}User: {message}\nAssistant:"
                    
                    # Generate response
                    response = await self.ollama_client.generate_response(
                        model=model,
                        prompt=full_prompt
                    )
                    
                    assistant_response = response.get('response', '')
                    
                    # Update conversation history
                    session['conversation'].append({
                        'user': message,
                        'assistant': assistant_response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Keep only last 10 exchanges
                    session['conversation'] = session['conversation'][-10:]
                    
                    return jsonify({
                        "response": assistant_response,
                        "model": model
                    })
                    
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            
            @self.app.route('/api/clear', methods=['POST'])
            def clear_conversation():
                """Clear conversation history."""
                session['conversation'] = []
                return jsonify({"status": "cleared"})
            
            @self.app.route('/api/health')
            async def health_check():
                """Health check endpoint."""
                healthy = await self.ollama_client.check_health()
                return jsonify({"healthy": healthy})
        
        def get_chat_template(self) -> str:
            """Get the HTML template for the chat interface."""
            return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama Chat</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
            padding: 1rem;
            box-sizing: border-box;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            background: white;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .message {
            margin-bottom: 1rem;
            padding: 0.5rem;
        }
        
        .user-message {
            text-align: right;
        }
        
        .user-message .content {
            background: #007bff;
            color: white;
            padding: 0.75rem 1rem;
            border-radius: 18px 18px 4px 18px;
            display: inline-block;
            max-width: 70%;
            word-wrap: break-word;
        }
        
        .assistant-message .content {
            background: #f1f3f4;
            color: #333;
            padding: 0.75rem 1rem;
            border-radius: 18px 18px 18px 4px;
            display: inline-block;
            max-width: 70%;
            word-wrap: break-word;
        }
        
        .input-container {
            display: flex;
            gap: 0.5rem;
            padding: 1rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .message-input {
            flex: 1;
            padding: 0.75rem;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .message-input:focus {
            border-color: #007bff;
        }
        
        .send-button {
            padding: 0.75rem 1.5rem;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            transition: background-color 0.3s;
        }
        
        .send-button:hover {
            background: #0056b3;
        }
        
        .send-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        
        .model-selector {
            margin-bottom: 1rem;
        }
        
        .model-selector select {
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #ddd;
            font-size: 1rem;
        }
        
        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .clear-button {
            padding: 0.5rem 1rem;
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        
        .clear-button:hover {
            background: #c82333;
        }
        
        .loading {
            text-align: center;
            color: #6c757d;
            font-style: italic;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 0.75rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🦙 Ollama Chat</h1>
        <p>Chat with your local AI models</p>
    </div>
    
    <div class="chat-container">
        <div class="controls">
            <div class="model-selector">
                <label for="model-select">Model: </label>
                <select id="model-select">
                    <option value="llama2">llama2</option>
                </select>
            </div>
            <button class="clear-button" onclick="clearConversation()">Clear Chat</button>
        </div>
        
        <div class="messages" id="messages">
            <div class="message assistant-message">
                <div class="content">
                    Hello! I'm your AI assistant. How can I help you today?
                </div>
            </div>
        </div>
        
        <div class="input-container">
            <input 
                type="text" 
                class="message-input" 
                id="message-input" 
                placeholder="Type your message..."
                onkeypress="handleKeyPress(event)"
            >
            <button class="send-button" id="send-button" onclick="sendMessage()">
                Send
            </button>
        </div>
    </div>

    <script>
        let isLoading = false;

        // Load available models
        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const data = await response.json();
                
                const select = document.getElementById('model-select');
                select.innerHTML = '';
                
                if (data.models && data.models.length > 0) {
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.name;
                        option.textContent = model.name;
                        select.appendChild(option);
                    });
                } else {
                    const option = document.createElement('option');
                    option.value = 'llama2';
                    option.textContent = 'llama2 (default)';
                    select.appendChild(option);
                }
            } catch (error) {
                console.error('Error loading models:', error);
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        async function sendMessage() {
            if (isLoading) return;

            const messageInput = document.getElementById('message-input');
            const message = messageInput.value.trim();
            
            if (!message) return;

            const messagesContainer = document.getElementById('messages');
            const selectedModel = document.getElementById('model-select').value;

            // Add user message
            addMessage(message, 'user');
            messageInput.value = '';

            // Show loading indicator
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message assistant-message loading';
            loadingDiv.innerHTML = '<div class="content">Thinking...</div>';
            messagesContainer.appendChild(loadingDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;

            // Disable input
            isLoading = true;
            document.getElementById('send-button').disabled = true;
            messageInput.disabled = true;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        model: selectedModel
                    })
                });

                const data = await response.json();

                // Remove loading indicator
                messagesContainer.removeChild(loadingDiv);

                if (response.ok) {
                    addMessage(data.response, 'assistant');
                } else {
                    addMessage(`Error: ${data.error}`, 'assistant', true);
                }
            } catch (error) {
                // Remove loading indicator
                messagesContainer.removeChild(loadingDiv);
                addMessage(`Error: ${error.message}`, 'assistant', true);
            } finally {
                // Re-enable input
                isLoading = false;
                document.getElementById('send-button').disabled = false;
                messageInput.disabled = false;
                messageInput.focus();
            }
        }

        function addMessage(content, sender, isError = false) {
            const messagesContainer = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'content';
            if (isError) {
                contentDiv.style.background = '#f8d7da';
                contentDiv.style.color = '#721c24';
            }
            
            // Handle line breaks in content
            contentDiv.innerHTML = content.replace(/\\n/g, '<br>');
            
            messageDiv.appendChild(contentDiv);
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        async function clearConversation() {
            try {
                await fetch('/api/clear', { method: 'POST' });
                const messagesContainer = document.getElementById('messages');
                messagesContainer.innerHTML = `
                    <div class="message assistant-message">
                        <div class="content">
                            Hello! I'm your AI assistant. How can I help you today?
                        </div>
                    </div>
                `;
            } catch (error) {
                console.error('Error clearing conversation:', error);
            }
        }

        // Load models when page loads
        document.addEventListener('DOMContentLoaded', () => {
            loadModels();
            document.getElementById('message-input').focus();
        });
    </script>
</body>
</html>
            """
        
        def run(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
            """Run the Flask application."""
            self.app.run(host=host, port=port, debug=debug)


# Simple HTTP Server (without Flask)
class SimpleHTTPChatServer:
    """Simple HTTP server for Ollama chat without Flask dependency."""
    
    def __init__(self, ollama_client: OllamaWebClient, port: int = 8000):
        self.ollama_client = ollama_client
        self.port = port
        self.conversations = {}  # Simple in-memory storage
    
    def get_simple_html(self) -> str:
        """Get a simple HTML page."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Ollama Chat</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .messages { border: 1px solid #ddd; height: 400px; overflow-y: scroll; padding: 10px; margin-bottom: 10px; }
        .input-area { display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 10px; }
        .input-area button { padding: 10px 20px; }
        .user-msg { background: #e3f2fd; padding: 5px; margin: 5px 0; border-radius: 5px; }
        .ai-msg { background: #f3e5f5; padding: 5px; margin: 5px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Simple Ollama Chat</h1>
    <div class="messages" id="messages"></div>
    <div class="input-area">
        <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
        <button onclick="sendMessage()">Send</button>
        <button onclick="clearMessages()">Clear</button>
    </div>
    
    <script>
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                addMessage(data.response, 'ai');
            } catch (error) {
                addMessage('Error: ' + error.message, 'ai');
            }
        }
        
        function addMessage(content, sender) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = sender + '-msg';
            div.textContent = (sender === 'user' ? 'You: ' : 'AI: ') + content;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function clearMessages() {
            document.getElementById('messages').innerHTML = '';
        }
    </script>
</body>
</html>
        """
    
    async def start_server(self):
        """Start the simple HTTP server."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse
        
        class ChatHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(self.server.chat_server.get_simple_html().encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_POST(self):
                if self.path == '/chat':
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                    
                    # This is a simplified version - in reality you'd need async handling
                    response_data = {"response": f"Echo: {data.get('message', '')}"}
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response_data).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
        
        server = HTTPServer(('localhost', self.port), ChatHandler)
        server.chat_server = self
        
        print(f"Simple chat server running on http://localhost:{self.port}")
        server.serve_forever()


# Example usage and main function
async def main():
    """Main function to run the web application examples."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize Ollama client
    ollama_client = OllamaWebClient()
    
    # Check if Ollama is available
    if not await ollama_client.check_health():
        print("Warning: Ollama is not available. Make sure it's running on localhost:11434")
        return
    
    print("Available web application examples:")
    print("1. Flask web application (requires Flask)")
    print("2. Simple HTTP server (no external dependencies)")
    
    if FLASK_AVAILABLE:
        print("\nStarting Flask web application...")
        
        # Create and run Flask app
        web_app = ChatWebApp(ollama_client)
        
        print("Flask app starting on http://127.0.0.1:5000")
        print("Press Ctrl+C to stop the server")
        
        # Note: In production, you'd want to use an ASGI server like Uvicorn
        web_app.run(host="127.0.0.1", port=5000, debug=True)
    
    else:
        print("\nFlask not available. You can install it with: pip install flask")
        print("Running simple HTTP server instead...")
        
        simple_server = SimpleHTTPChatServer(ollama_client, port=8000)
        await simple_server.start_server()


if __name__ == "__main__":
    if not HTTPX_AVAILABLE:
        print("httpx is required. Install with: pip install httpx")
        exit(1)
    
    asyncio.run(main())
