#!/usr/bin/env python3
"""
ollama_client.py - Python client for Ollama automation

This module provides a robust Python client for interacting with Ollama's API.
It includes error handling, retry logic, and comprehensive functionality.

Usage:
    from ollama_client import OllamaClient, ChatMessage
    
    client = OllamaClient()
    response = client.generate("llama3.2:3b", "What is AI?")
    print(response)

Author: Book Example
License: MIT
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Generator, Union
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    raise

@dataclass
class ModelConfig:
    """Configuration for model interactions"""
    name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 60
    
@dataclass
class ChatMessage:
    """Chat message structure"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: Optional[str] = None

class OllamaAPIError(Exception):
    """Custom exception for Ollama API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

class OllamaClient:
    """
    Python client for interacting with Ollama API
    
    This client provides comprehensive functionality for:
    - Text generation and chat
    - Model management
    - Health monitoring
    - Error handling and retries
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60, max_retries: int = 3):
        """
        Initialize the Ollama client
        
        Args:
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ollama_client')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def health_check(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('models', [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to list models: {e}")
            raise OllamaAPIError(f"Failed to list models: {e}")
    
    def generate(
        self, 
        model: str, 
        prompt: str, 
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text using specified model
        
        Args:
            model: Model name to use
            prompt: Input prompt
            system: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Generated text or generator for streaming
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            }
        }
        
        if system:
            payload["system"] = system
            
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._stream_response(response)
            else:
                data = response.json()
                return data.get('response', '')
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Generation failed: {e}")
            raise OllamaAPIError(f"Generation failed: {e}")
    
    def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Chat with model using message history
        
        Args:
            model: Model name to use
            messages: List of chat messages
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Response message or generator for streaming
        """
        payload = {
            "model": model,
            "messages": [
                {"role": msg.role, "content": msg.content} 
                for msg in messages
            ],
            "stream": stream,
            "options": {
                "temperature": temperature,
                **kwargs
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._stream_response(response, message_key='message')
            else:
                data = response.json()
                return data.get('message', {}).get('content', '')
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Chat failed: {e}")
            raise OllamaAPIError(f"Chat failed: {e}")
    
    def _stream_response(self, response: requests.Response, message_key: str = 'response') -> Generator[str, None, None]:
        """Handle streaming response"""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if message_key == 'message':
                        content = data.get('message', {}).get('content', '')
                    else:
                        content = data.get('response', '')
                    
                    if content:
                        yield content
                        
                    if data.get('done', False):
                        break
                        
                except json.JSONDecodeError:
                    continue
    
    def pull_model(self, model: str) -> bool:
        """
        Pull/download a model
        
        Args:
            model: Model name to pull
            
        Returns:
            True if successful
        """
        payload = {"name": model}
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=300  # Longer timeout for model downloads
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to pull model {model}: {e}")
            raise OllamaAPIError(f"Failed to pull model {model}: {e}")
    
    def delete_model(self, model: str) -> bool:
        """
        Delete a model
        
        Args:
            model: Model name to delete
            
        Returns:
            True if successful
        """
        try:
            response = self.session.delete(f"{self.base_url}/api/delete", json={"name": model})
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to delete model {model}: {e}")
            raise OllamaAPIError(f"Failed to delete model {model}: {e}")
    
    def get_embeddings(self, model: str, prompt: str) -> List[float]:
        """
        Get embeddings for text
        
        Args:
            model: Model name to use for embeddings
            prompt: Text to get embeddings for
            
        Returns:
            List of embedding values
        """
        payload = {
            "model": model,
            "prompt": prompt
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get('embedding', [])
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get embeddings: {e}")
            raise OllamaAPIError(f"Failed to get embeddings: {e}")

class AITaskProcessor:
    """
    High-level processor for common AI tasks
    """
    
    def __init__(self, client: OllamaClient):
        self.client = client
        
    def summarize_text(self, text: str, model: str = "llama3.2:3b", max_length: int = 150) -> str:
        """Summarize long text"""
        prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"
        return self.client.generate(model, prompt, temperature=0.3)
    
    def translate_text(self, text: str, target_language: str, model: str = "llama3.2:3b") -> str:
        """Translate text to target language"""
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
        return self.client.generate(model, prompt, temperature=0.1)
    
    def analyze_sentiment(self, text: str, model: str = "llama3.2:3b") -> str:
        """Analyze sentiment of text"""
        prompt = f"Analyze the sentiment of this text (positive, negative, or neutral):\n\n{text}"
        return self.client.generate(model, prompt, temperature=0.1)
    
    def extract_keywords(self, text: str, model: str = "llama3.2:3b", count: int = 10) -> str:
        """Extract keywords from text"""
        prompt = f"Extract the top {count} keywords from this text:\n\n{text}"
        return self.client.generate(model, prompt, temperature=0.2)

def main():
    """Example usage of the Ollama client"""
    client = OllamaClient()
    
    # Check health
    if not client.health_check():
        print("❌ Ollama server is not running")
        return
    
    print("✅ Ollama server is running")
    
    # List available models
    try:
        models = client.list_models()
        print(f"Available models: {[m['name'] for m in models]}")
    except OllamaAPIError as e:
        print(f"Failed to list models: {e}")
        return
    
    # Simple generation example
    try:
        response = client.generate(
            model="llama3.2:3b",
            prompt="Explain Python decorators in simple terms",
            temperature=0.5
        )
        print(f"Response: {response}")
    except OllamaAPIError as e:
        print(f"Generation failed: {e}")
    
    # Chat example
    try:
        messages = [
            ChatMessage("user", "What is machine learning?"),
        ]
        
        chat_response = client.chat("llama3.2:3b", messages)
        print(f"Chat response: {chat_response}")
    except OllamaAPIError as e:
        print(f"Chat failed: {e}")
    
    # Task processor example
    processor = AITaskProcessor(client)
    
    # Analyze some text
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on 
    creating systems that can learn and improve from experience without being 
    explicitly programmed. It uses algorithms to analyze data, identify 
    patterns, and make predictions or decisions.
    """
    
    try:
        summary = processor.summarize_text(sample_text)
        print(f"Summary: {summary}")
        
        sentiment = processor.analyze_sentiment(sample_text)
        print(f"Sentiment: {sentiment}")
    except OllamaAPIError as e:
        print(f"Task processing failed: {e}")

if __name__ == "__main__":
    main()
