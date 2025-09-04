"""
Mobile Development Examples for Ollama
Chapter 13: Mobile Development
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("httpx not available. Install with: pip install httpx")


class MobileOllamaClient:
    """Mobile-optimized client for Ollama API."""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.logger = logging.getLogger(__name__)
        self.timeout_config = httpx.Timeout(
            connect=10.0,  # 10 seconds for connection
            read=60.0,     # 1 minute for reading response
            write=10.0,    # 10 seconds for writing request
            pool=5.0       # 5 seconds for acquiring connection from pool
        )
    
    async def generate_response(
        self,
        model: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate response optimized for mobile usage."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required")
        
        # Prepare request data
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or 150,  # Limit tokens for mobile
                "top_k": 20,  # Reduce for faster inference
                "top_p": 0.9
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout_config) as client:
            try:
                if stream:
                    return await self._handle_streaming_response(client, request_data)
                else:
                    response = await client.post(
                        f"{self.ollama_host}/api/generate",
                        json=request_data
                    )
                    response.raise_for_status()
                    return response.json()
                    
            except httpx.TimeoutException:
                raise Exception("Request timed out - check your internet connection")
            except httpx.ConnectError:
                raise Exception("Cannot connect to Ollama server")
            except Exception as e:
                self.logger.error(f"Error generating response: {e}")
                raise
    
    async def _handle_streaming_response(self, client: httpx.AsyncClient, request_data: Dict) -> Dict[str, Any]:
        """Handle streaming response for real-time updates."""
        full_response = ""
        
        async with client.stream('POST', f"{self.ollama_host}/api/generate", json=request_data) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            full_response += chunk['response']
                            
                        if chunk.get('done', False):
                            return {
                                "response": full_response,
                                "model": request_data['model'],
                                "done": True
                            }
                    except json.JSONDecodeError:
                        continue
        
        return {"response": full_response, "model": request_data['model'], "done": True}
    
    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a model."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required")
        
        async with httpx.AsyncClient(timeout=self.timeout_config) as client:
            try:
                response = await client.post(
                    f"{self.ollama_host}/api/show",
                    json={"name": model}
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                self.logger.error(f"Error getting model info: {e}")
                return {}
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models with mobile-relevant info."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required")
        
        async with httpx.AsyncClient(timeout=self.timeout_config) as client:
            try:
                response = await client.get(f"{self.ollama_host}/api/tags")
                response.raise_for_status()
                data = response.json()
                
                # Add mobile-friendly information
                models = data.get("models", [])
                for model in models:
                    # Add size category for mobile consideration
                    size_bytes = model.get("size", 0)
                    size_gb = size_bytes / (1024**3)
                    
                    if size_gb < 1:
                        model["mobile_category"] = "small"
                    elif size_gb < 4:
                        model["mobile_category"] = "medium"
                    else:
                        model["mobile_category"] = "large"
                    
                    model["size_gb"] = round(size_gb, 2)
                
                return models
                
            except Exception as e:
                self.logger.error(f"Error listing models: {e}")
                return []
    
    async def check_connection(self) -> Dict[str, Any]:
        """Check connection status and performance."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                response.raise_for_status()
                
                end_time = asyncio.get_event_loop().time()
                response_time = round((end_time - start_time) * 1000, 2)  # milliseconds
                
                return {
                    "connected": True,
                    "response_time_ms": response_time,
                    "status": "good" if response_time < 1000 else "slow"
                }
                
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "status": "error"
            }


class MobileChatSession:
    """Manage chat sessions optimized for mobile apps."""
    
    def __init__(self, ollama_client: MobileOllamaClient, max_history: int = 10):
        self.ollama_client = ollama_client
        self.max_history = max_history
        self.conversation_history = []
        self.current_model = "llama2"
        self.session_id = None
        self.logger = logging.getLogger(__name__)
    
    def start_session(self, model: str = "llama2") -> str:
        """Start a new chat session."""
        self.current_model = model
        self.conversation_history = []
        self.session_id = f"mobile_session_{asyncio.get_event_loop().time()}"
        
        self.logger.info(f"Started new session: {self.session_id}")
        return self.session_id
    
    async def send_message(self, message: str, use_context: bool = True) -> Dict[str, Any]:
        """Send a message in the current session."""
        try:
            # Build context from conversation history
            context = ""
            if use_context and self.conversation_history:
                # Use only recent messages to keep context manageable
                recent_history = self.conversation_history[-3:]  # Last 3 exchanges
                for exchange in recent_history:
                    context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"
            
            # Create the full prompt
            if context:
                full_prompt = f"{context}User: {message}\nAssistant:"
            else:
                full_prompt = message
            
            # Generate response
            response = await self.ollama_client.generate_response(
                model=self.current_model,
                prompt=full_prompt,
                max_tokens=200,  # Limit for mobile
                temperature=0.7
            )
            
            assistant_response = response.get("response", "").strip()
            
            # Add to conversation history
            self.conversation_history.append({
                "user": message,
                "assistant": assistant_response,
                "timestamp": asyncio.get_event_loop().time(),
                "model": self.current_model
            })
            
            # Trim history if too long
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            return {
                "response": assistant_response,
                "session_id": self.session_id,
                "model": self.current_model,
                "message_count": len(self.conversation_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return {
                "error": str(e),
                "session_id": self.session_id
            }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        if not self.conversation_history:
            return {
                "message_count": 0,
                "session_id": self.session_id,
                "model": self.current_model
            }
        
        return {
            "session_id": self.session_id,
            "model": self.current_model,
            "message_count": len(self.conversation_history),
            "first_message": self.conversation_history[0]["user"][:50] + "..." if len(self.conversation_history[0]["user"]) > 50 else self.conversation_history[0]["user"],
            "last_message": self.conversation_history[-1]["user"][:50] + "..." if len(self.conversation_history[-1]["user"]) > 50 else self.conversation_history[-1]["user"],
            "start_time": self.conversation_history[0]["timestamp"],
            "last_activity": self.conversation_history[-1]["timestamp"]
        }
    
    def export_conversation(self) -> List[Dict[str, Any]]:
        """Export conversation for saving or sharing."""
        return [
            {
                "user": exchange["user"],
                "assistant": exchange["assistant"],
                "timestamp": exchange["timestamp"],
                "model": exchange["model"]
            }
            for exchange in self.conversation_history
        ]
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.logger.info("Conversation cleared")


class MobileAppSimulator:
    """Simulate a mobile app interface for testing."""
    
    def __init__(self, ollama_client: MobileOllamaClient):
        self.ollama_client = ollama_client
        self.chat_session = MobileChatSession(ollama_client)
        self.logger = logging.getLogger(__name__)
    
    async def startup_sequence(self) -> Dict[str, Any]:
        """Simulate app startup sequence."""
        startup_info = {
            "app_name": "Ollama Mobile Chat",
            "version": "1.0.0",
            "startup_time": asyncio.get_event_loop().time()
        }
        
        print("🚀 Starting Ollama Mobile App Simulator...")
        print("=" * 50)
        
        # Check connection
        print("📡 Checking connection to Ollama server...")
        connection_status = await self.ollama_client.check_connection()
        
        if not connection_status["connected"]:
            print(f"❌ Connection failed: {connection_status.get('error', 'Unknown error')}")
            return {"startup_success": False, "error": connection_status.get('error')}
        
        print(f"✅ Connected! Response time: {connection_status['response_time_ms']}ms")
        startup_info["connection_status"] = connection_status
        
        # Load available models
        print("📱 Loading available models...")
        models = await self.ollama_client.list_models()
        
        if not models:
            print("⚠️ No models available")
        else:
            print(f"✅ Found {len(models)} models:")
            for model in models[:3]:  # Show first 3
                size = model.get('size_gb', 0)
                category = model.get('mobile_category', 'unknown')
                print(f"   • {model['name']} ({size}GB - {category})")
        
        startup_info["available_models"] = models
        startup_info["startup_success"] = True
        
        return startup_info
    
    async def run_interactive_demo(self):
        """Run an interactive demo of the mobile app."""
        # Startup
        startup_result = await self.startup_sequence()
        if not startup_result["startup_success"]:
            return
        
        # Start chat session
        models = startup_result["available_models"]
        if models:
            default_model = models[0]["name"]
        else:
            default_model = "llama2"
        
        session_id = self.chat_session.start_session(default_model)
        print(f"\n💬 Chat session started with {default_model}")
        print("📱 Mobile Chat Simulator - Type your messages below")
        print("   Commands: /models, /switch <model>, /clear, /summary, /quit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n📱 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    await self.handle_command(user_input)
                    continue
                
                # Send regular message
                print("🤖 Thinking...")
                
                result = await self.chat_session.send_message(user_input)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"🤖 Assistant: {result['response']}")
                    print(f"   (Model: {result['model']}, Messages: {result['message_count']})")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
    
    async def handle_command(self, command: str):
        """Handle app commands."""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/models":
            models = await self.ollama_client.list_models()
            if models:
                print("\n📋 Available models:")
                for model in models:
                    size = model.get('size_gb', 0)
                    category = model.get('mobile_category', 'unknown')
                    current = "← current" if model['name'] == self.chat_session.current_model else ""
                    print(f"   • {model['name']} ({size}GB - {category}) {current}")
            else:
                print("No models available")
        
        elif cmd == "/switch":
            if len(parts) > 1:
                new_model = parts[1]
                self.chat_session.current_model = new_model
                print(f"✅ Switched to model: {new_model}")
            else:
                print("Usage: /switch <model_name>")
        
        elif cmd == "/clear":
            self.chat_session.clear_conversation()
            print("✅ Conversation cleared")
        
        elif cmd == "/summary":
            summary = self.chat_session.get_conversation_summary()
            print(f"\n📊 Session Summary:")
            print(f"   Session ID: {summary['session_id']}")
            print(f"   Model: {summary['model']}")
            print(f"   Messages: {summary['message_count']}")
            if summary['message_count'] > 0:
                print(f"   First: {summary['first_message']}")
                print(f"   Last: {summary['last_message']}")
        
        elif cmd == "/quit":
            raise KeyboardInterrupt
        
        else:
            print(f"Unknown command: {cmd}")
            print("Available commands: /models, /switch, /clear, /summary, /quit")


def create_mobile_api_example():
    """Create example mobile API integration code."""
    
    api_example = '''
"""
Mobile API Integration Example
For React Native, Flutter, or native mobile apps
"""

// JavaScript/React Native Example
class OllamaMobileAPI {
    constructor(baseURL = 'http://localhost:11434') {
        this.baseURL = baseURL;
        this.timeout = 30000; // 30 seconds
    }
    
    async generateResponse(model, prompt, options = {}) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);
        
        try {
            const response = await fetch(`${this.baseURL}/api/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: model,
                    prompt: prompt,
                    stream: false,
                    options: {
                        num_predict: options.maxTokens || 150,
                        temperature: options.temperature || 0.7,
                        top_k: 20,
                        top_p: 0.9
                    }
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
            
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                throw new Error('Request timeout - check your connection');
            }
            
            throw error;
        }
    }
    
    async listModels() {
        try {
            const response = await fetch(`${this.baseURL}/api/tags`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            return data.models || [];
            
        } catch (error) {
            console.error('Error listing models:', error);
            return [];
        }
    }
    
    async checkHealth() {
        try {
            const models = await this.listModels();
            return { healthy: true, modelCount: models.length };
        } catch (error) {
            return { healthy: false, error: error.message };
        }
    }
}

// Usage in React Native component
const ChatComponent = () => {
    const [ollama] = useState(new OllamaMobileAPI());
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    
    const sendMessage = async (text) => {
        setIsLoading(true);
        
        try {
            const result = await ollama.generateResponse('llama2', text);
            
            setMessages(prev => [
                ...prev,
                { type: 'user', text: text },
                { type: 'assistant', text: result.response }
            ]);
            
        } catch (error) {
            console.error('Error sending message:', error);
            // Show error to user
        } finally {
            setIsLoading(false);
        }
    };
    
    // Component render logic...
};

// Flutter/Dart Example
class OllamaMobileAPI {
    final String baseURL;
    final Duration timeout;
    
    OllamaMobileAPI({
        this.baseURL = 'http://localhost:11434',
        this.timeout = const Duration(seconds: 30),
    });
    
    Future<Map<String, dynamic>> generateResponse(
        String model, 
        String prompt, 
        {int? maxTokens, double? temperature}
    ) async {
        final client = http.Client();
        
        try {
            final response = await client.post(
                Uri.parse('$baseURL/api/generate'),
                headers: {'Content-Type': 'application/json'},
                body: jsonEncode({
                    'model': model,
                    'prompt': prompt,
                    'stream': false,
                    'options': {
                        'num_predict': maxTokens ?? 150,
                        'temperature': temperature ?? 0.7,
                        'top_k': 20,
                        'top_p': 0.9,
                    }
                }),
            ).timeout(timeout);
            
            if (response.statusCode == 200) {
                return jsonDecode(response.body);
            } else {
                throw Exception('HTTP ${response.statusCode}: ${response.body}');
            }
            
        } on TimeoutException {
            throw Exception('Request timeout - check your connection');
        } finally {
            client.close();
        }
    }
    
    Future<List<dynamic>> listModels() async {
        final client = http.Client();
        
        try {
            final response = await client.get(
                Uri.parse('$baseURL/api/tags'),
            ).timeout(timeout);
            
            if (response.statusCode == 200) {
                final data = jsonDecode(response.body);
                return data['models'] ?? [];
            } else {
                throw Exception('HTTP ${response.statusCode}: ${response.body}');
            }
            
        } catch (e) {
            print('Error listing models: $e');
            return [];
        } finally {
            client.close();
        }
    }
}
'''
    
    with open("mobile_api_examples.txt", "w") as f:
        f.write(api_example)
    
    print("📱 Created mobile API integration examples: mobile_api_examples.txt")


# Main function for demonstration
async def main():
    """Main function to run mobile development examples."""
    logging.basicConfig(level=logging.INFO)
    
    if not HTTPX_AVAILABLE:
        print("httpx is required. Install with: pip install httpx")
        return
    
    # Initialize mobile client
    ollama_client = MobileOllamaClient()
    
    # Create mobile app simulator
    app_simulator = MobileAppSimulator(ollama_client)
    
    # Create mobile API examples
    create_mobile_api_example()
    
    print("📱 Ollama Mobile Development Examples")
    print("=" * 50)
    print("1. Run interactive mobile app simulator")
    print("2. Check connection and models")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        await app_simulator.run_interactive_demo()
    elif choice == "2":
        connection = await ollama_client.check_connection()
        print(f"Connection status: {connection}")
        
        if connection["connected"]:
            models = await ollama_client.list_models()
            print(f"Available models: {len(models)}")
            for model in models:
                print(f"  - {model['name']} ({model.get('size_gb', 0)}GB)")
    else:
        print("👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
