#!/usr/bin/env python3
"""
flask_api_server.py - Flask API server for AI services

This module provides a complete REST API server that wraps Ollama functionality
with features like rate limiting, request tracking, streaming, and error handling.

Usage:
    python flask_api_server.py

Endpoints:
    GET  /v1/health          - Health check
    GET  /v1/models          - List available models
    POST /v1/generate        - Generate text
    POST /v1/chat            - Chat completion
    GET  /v1/stats           - Request statistics

Author: Book Example
License: MIT
"""

import json
import logging
import queue
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from flask import Flask, request, jsonify, stream_with_context, Response
    from flask_cors import CORS
except ImportError:
    print("Error: Flask not found. Install with: pip install flask flask-cors")
    raise

try:
    from ollama_client import OllamaClient, OllamaAPIError
except ImportError:
    print("Error: ollama_client module not found. Ensure it's in the same directory.")
    raise

# Configuration
API_VERSION = "v1"
DEFAULT_MODEL = "llama3.2:3b"
MAX_REQUESTS_PER_MINUTE = 100
MAX_CONCURRENT_REQUESTS = 10

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Ollama client
ollama_client = OllamaClient()

# Request tracking
request_counter = 0
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_response_time": 0.0,
    "current_concurrent": 0,
    "start_time": datetime.now().isoformat()
}

class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, max_requests: int, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is within rate limit"""
        now = time.time()
        
        with self.lock:
            # Clean old requests
            cutoff = now - self.window_seconds
            self.requests = {
                cid: [t for t in timestamps if t > cutoff]
                for cid, timestamps in self.requests.items()
            }
            
            # Check current client
            client_requests = self.requests.get(client_id, [])
            
            if len(client_requests) < self.max_requests:
                client_requests.append(now)
                self.requests[client_id] = client_requests
                return True
            
            return False

class RequestTracker:
    """Track active requests and provide statistics"""
    
    def __init__(self):
        self.active_requests = {}
        self.lock = threading.Lock()
    
    def start_request(self, request_id: str, model: str, prompt: str):
        """Start tracking a request"""
        with self.lock:
            self.active_requests[request_id] = {
                "model": model,
                "prompt_length": len(prompt),
                "start_time": time.time(),
                "status": "processing"
            }
            request_stats["current_concurrent"] = len(self.active_requests)
    
    def complete_request(self, request_id: str, success: bool = True):
        """Complete tracking a request"""
        with self.lock:
            if request_id in self.active_requests:
                request_info = self.active_requests.pop(request_id)
                duration = time.time() - request_info["start_time"]
                
                # Update global stats
                request_stats["total_requests"] += 1
                if success:
                    request_stats["successful_requests"] += 1
                else:
                    request_stats["failed_requests"] += 1
                
                # Update average response time
                total = request_stats["total_requests"]
                current_avg = request_stats["average_response_time"]
                request_stats["average_response_time"] = (
                    (current_avg * (total - 1) + duration) / total
                )
                
                request_stats["current_concurrent"] = len(self.active_requests)
    
    def get_active_requests(self) -> Dict[str, Any]:
        """Get currently active requests"""
        with self.lock:
            return {
                k: v for k, v in self.active_requests.items() 
                if v["status"] == "processing"
            }

# Global instances
rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)
request_tracker = RequestTracker()

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist",
        "code": 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "code": 500
    }), 500

@app.route(f"/{API_VERSION}/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    ollama_status = ollama_client.health_check()
    
    return jsonify({
        "status": "healthy" if ollama_status else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "ollama_server": "running" if ollama_status else "not responding",
        "api_version": API_VERSION
    }), 200 if ollama_status else 503

@app.route(f"/{API_VERSION}/models", methods=["GET"])
def list_models():
    """List available models"""
    try:
        models = ollama_client.list_models()
        return jsonify({
            "models": models,
            "count": len(models),
            "default_model": DEFAULT_MODEL
        })
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return jsonify({
            "error": "Failed to retrieve models",
            "message": str(e)
        }), 500

@app.route(f"/{API_VERSION}/generate", methods=["POST"])
def generate_text():
    """Generate text using AI model"""
    global request_counter
    request_counter += 1
    request_id = f"req_{int(time.time())}_{request_counter}"
    
    # Get client IP for rate limiting
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    
    # Check rate limit
    if not rate_limiter.is_allowed(client_ip):
        return jsonify({
            "error": "Rate limit exceeded",
            "message": f"Maximum {MAX_REQUESTS_PER_MINUTE} requests per minute allowed"
        }), 429
    
    try:
        data = request.json
        if not data:
            return jsonify({
                "error": "Invalid request",
                "message": "JSON data required"
            }), 400
        
        # Validate required fields
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({
                "error": "Missing prompt",
                "message": "The 'prompt' field is required"
            }), 400
        
        model = data.get("model", DEFAULT_MODEL)
        system = data.get("system")
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", 1000)
        stream = data.get("stream", False)
        
        # Start request tracking
        request_tracker.start_request(request_id, model, prompt)
        
        # Generate response
        if stream:
            return Response(
                stream_with_context(
                    generate_streaming_response(
                        request_id, model, prompt, system, temperature
                    )
                ),
                mimetype="text/plain"
            )
        else:
            response = ollama_client.generate(
                model=model,
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            request_tracker.complete_request(request_id, True)
            
            return jsonify({
                "id": request_id,
                "model": model,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
    except OllamaAPIError as e:
        request_tracker.complete_request(request_id, False)
        logger.error(f"Ollama API error: {e}")
        return jsonify({
            "error": "AI service error",
            "message": str(e)
        }), 502
        
    except Exception as e:
        request_tracker.complete_request(request_id, False)
        logger.error(f"Generation error: {e}")
        return jsonify({
            "error": "Generation failed",
            "message": str(e)
        }), 500

def generate_streaming_response(request_id: str, model: str, prompt: str, 
                              system: Optional[str], temperature: float):
    """Generate streaming response"""
    try:
        for chunk in ollama_client.generate(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            stream=True
        ):
            yield f"data: {json.dumps({'chunk': chunk, 'id': request_id})}\n\n"
        
        # End stream
        yield f"data: {json.dumps({'done': True, 'id': request_id})}\n\n"
        request_tracker.complete_request(request_id, True)
        
    except Exception as e:
        error_response = {
            "error": True,
            "message": str(e),
            "id": request_id
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        request_tracker.complete_request(request_id, False)

@app.route(f"/{API_VERSION}/chat", methods=["POST"])
def chat_completion():
    """Chat completion endpoint"""
    global request_counter
    request_counter += 1
    request_id = f"chat_{int(time.time())}_{request_counter}"
    
    # Get client IP for rate limiting
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    
    # Check rate limit
    if not rate_limiter.is_allowed(client_ip):
        return jsonify({
            "error": "Rate limit exceeded",
            "message": f"Maximum {MAX_REQUESTS_PER_MINUTE} requests per minute allowed"
        }), 429
    
    try:
        data = request.json
        if not data:
            return jsonify({
                "error": "Invalid request",
                "message": "JSON data required"
            }), 400
        
        # Validate required fields
        messages = data.get("messages")
        if not messages or not isinstance(messages, list):
            return jsonify({
                "error": "Missing messages",
                "message": "The 'messages' field is required and must be a list"
            }), 400
        
        model = data.get("model", DEFAULT_MODEL)
        temperature = data.get("temperature", 0.7)
        stream = data.get("stream", False)
        
        # Convert to ChatMessage objects
        from ollama_client import ChatMessage
        chat_messages = []
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                return jsonify({
                    "error": "Invalid message format",
                    "message": "Each message must have 'role' and 'content' fields"
                }), 400
            chat_messages.append(ChatMessage(msg['role'], msg['content']))
        
        # Start request tracking
        request_tracker.start_request(request_id, model, str(messages))
        
        # Generate response
        if stream:
            return Response(
                stream_with_context(
                    chat_streaming_response(
                        request_id, model, chat_messages, temperature
                    )
                ),
                mimetype="text/plain"
            )
        else:
            response = ollama_client.chat(
                model=model,
                messages=chat_messages,
                temperature=temperature
            )
            
            request_tracker.complete_request(request_id, True)
            
            return jsonify({
                "id": request_id,
                "model": model,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "timestamp": datetime.now().isoformat()
            })
            
    except OllamaAPIError as e:
        request_tracker.complete_request(request_id, False)
        logger.error(f"Ollama API error: {e}")
        return jsonify({
            "error": "AI service error",
            "message": str(e)
        }), 502
        
    except Exception as e:
        request_tracker.complete_request(request_id, False)
        logger.error(f"Chat error: {e}")
        return jsonify({
            "error": "Chat failed",
            "message": str(e)
        }), 500

def chat_streaming_response(request_id: str, model: str, messages, temperature: float):
    """Generate streaming chat response"""
    try:
        for chunk in ollama_client.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        ):
            response_data = {
                "id": request_id,
                "model": model,
                "delta": {"content": chunk},
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(response_data)}\n\n"
        
        # End stream
        yield f"data: {json.dumps({'done': True, 'id': request_id})}\n\n"
        request_tracker.complete_request(request_id, True)
        
    except Exception as e:
        error_response = {
            "error": True,
            "message": str(e),
            "id": request_id
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        request_tracker.complete_request(request_id, False)

@app.route(f"/{API_VERSION}/stats", methods=["GET"])
def get_stats():
    """Get API statistics"""
    active_requests = request_tracker.get_active_requests()
    
    return jsonify({
        **request_stats,
        "active_requests": len(active_requests),
        "rate_limit": {
            "max_requests_per_minute": MAX_REQUESTS_PER_MINUTE,
            "max_concurrent": MAX_CONCURRENT_REQUESTS
        },
        "uptime_seconds": (
            datetime.now() - datetime.fromisoformat(request_stats["start_time"])
        ).total_seconds()
    })

@app.route(f"/{API_VERSION}/debug", methods=["GET"])
def debug_info():
    """Debug information endpoint"""
    return jsonify({
        "active_requests": request_tracker.get_active_requests(),
        "ollama_health": ollama_client.health_check(),
        "server_time": datetime.now().isoformat(),
        "api_version": API_VERSION
    })

# WebSocket support for real-time updates
try:
    from flask_socketio import SocketIO, emit
    
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    @socketio.on('connect')
    def on_connect():
        emit('status', {'connected': True, 'timestamp': datetime.now().isoformat()})
    
    @socketio.on('request_stats')
    def on_request_stats():
        stats = {
            **request_stats,
            "active_requests": len(request_tracker.get_active_requests())
        }
        emit('stats_update', stats)
    
except ImportError:
    print("SocketIO not available. Install with: pip install flask-socketio")
    socketio = None

def main():
    """Main function to run the server"""
    # Check Ollama connection
    if not ollama_client.health_check():
        logger.warning("Ollama server not responding. Some endpoints may not work.")
    else:
        logger.info("Connected to Ollama server successfully")
    
    # Run the Flask app
    if socketio:
        socketio.run(
            app,
            host="0.0.0.0",
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
    else:
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True
        )

if __name__ == "__main__":
    main()
