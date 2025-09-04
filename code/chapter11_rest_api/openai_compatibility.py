#!/usr/bin/env python3
"""
openai_compatibility.py - OpenAI API compatibility examples

This module demonstrates how to use Ollama with OpenAI's Python library
for seamless integration with existing OpenAI-based applications.

Usage:
    python openai_compatibility.py

Features:
- Chat completions
- Streaming responses  
- Structured outputs
- Vision capabilities
- Model management
- Embeddings

Author: Book Example
License: MIT
"""

import base64
import json
from typing import List, Dict, Any
from pydantic import BaseModel

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not found. Install with: pip install openai")
    raise

# Initialize OpenAI client for Ollama
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',  # Required but ignored
)

class BookRecommendation(BaseModel):
    """Structured output model for book recommendations"""
    title: str
    author: str
    genre: str
    description: str
    rating: float

class LocationInfo(BaseModel):
    """Structured output for location analysis"""
    city: str
    country: str
    description: str
    interesting_facts: List[str]

def basic_chat_example():
    """Basic chat completion using OpenAI client."""
    print("=== Basic Chat Completion ===")
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain quantum computing in simple terms",
            }
        ],
        model="llama3.2",
        max_tokens=200,
        temperature=0.7
    )
    
    print("Response:", chat_completion.choices[0].message.content)

def structured_outputs_example():
    """Example using structured outputs with Pydantic models."""
    print("\n=== Structured Outputs Example ===")
    
    try:
        completion = client.beta.chat.completions.parse(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "You are a helpful book recommendation assistant."},
                {"role": "user", "content": "Recommend a science fiction book"}
            ],
            response_format=BookRecommendation,
        )
        
        book = completion.choices[0].message.parsed
        print(f"Title: {book.title}")
        print(f"Author: {book.author}")
        print(f"Genre: {book.genre}")
        print(f"Description: {book.description}")
        print(f"Rating: {book.rating}/5")
        
    except Exception as e:
        print(f"Structured output not supported: {e}")
        # Fallback to regular completion
        completion = client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "You are a helpful book recommendation assistant. Respond in JSON format with title, author, genre, description, and rating fields."},
                {"role": "user", "content": "Recommend a science fiction book"}
            ],
            temperature=0.3
        )
        print("Fallback response:", completion.choices[0].message.content)

def vision_example():
    """Example using vision capabilities (requires multimodal model)."""
    print("\n=== Vision Example ===")
    
    # Create a simple test image (you would normally load an actual image)
    try:
        # Note: This would work with an actual image file
        # with open("image.jpg", "rb") as image_file:
        #     base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # For demo purposes, we'll show the structure
        print("Vision API structure (requires actual image):")
        example_structure = {
            "model": "llama3.2-vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        print(json.dumps(example_structure, indent=2))
        
    except Exception as e:
        print(f"Vision example failed: {e}")

def streaming_example():
    """Example of streaming chat completion."""
    print("\n=== Streaming Example ===")
    
    stream = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": "Write a short story about a robot"}],
        stream=True,
        max_tokens=300
    )
    
    print("Streaming response:")
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    
    print(f"\n\nTotal response length: {len(full_response)} characters")

def completions_example():
    """Example using the completions endpoint."""
    print("\n=== Completions Example ===")
    
    completion = client.completions.create(
        model="llama3.2",
        prompt="The future of artificial intelligence is",
        max_tokens=100,
        temperature=0.8
    )
    
    print("Completion:", completion.choices[0].text)

def models_example():
    """Example of listing and retrieving model information."""
    print("\n=== Models Example ===")
    
    models = client.models.list()
    print("Available models:")
    for model in models.data:
        print(f"- {model.id} (created: {model.created})")
    
    if models.data:
        # Get details of first model
        model = client.models.retrieve(models.data[0].id)
        print(f"\nModel details: {model.id}")
        print(f"Object type: {model.object}")

def embeddings_example():
    """Example of generating embeddings."""
    print("\n=== Embeddings Example ===")
    
    try:
        embedding = client.embeddings.create(
            model="all-minilm",
            input=["Hello world", "How are you?", "Machine learning is fascinating"]
        )
        
        print(f"Generated {len(embedding.data)} embeddings")
        for i, emb in enumerate(embedding.data):
            print(f"Embedding {i}: {len(emb.embedding)} dimensions")
            # Show first 5 values as sample
            print(f"Sample values: {emb.embedding[:5]}")
            
    except Exception as e:
        print(f"Embeddings failed: {e}")

def advanced_chat_example():
    """Advanced chat example with system prompt and parameters."""
    print("\n=== Advanced Chat Example ===")
    
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a pirate captain explaining complex topics in pirate speak."},
            {"role": "user", "content": "Explain machine learning"}
        ],
        temperature=0.8,
        max_tokens=200,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["\n\n"]
    )
    
    print("Pirate ML explanation:", response.choices[0].message.content)

def function_calling_example():
    """Example of function calling (if supported)."""
    print("\n=== Function Calling Example ===")
    
    # Define a function
    functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
            functions=functions,
            function_call="auto"
        )
        
        print("Function calling response:", response.choices[0].message)
        
    except Exception as e:
        print(f"Function calling not supported: {e}")

def batch_processing_example():
    """Example of processing multiple requests efficiently."""
    print("\n=== Batch Processing Example ===")
    
    prompts = [
        "What is Python?",
        "What is JavaScript?", 
        "What is Rust?",
        "What is Go?"
    ]
    
    responses = []
    for i, prompt in enumerate(prompts):
        try:
            response = client.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3
            )
            responses.append({
                "prompt": prompt,
                "response": response.choices[0].message.content,
                "index": i
            })
            print(f"✅ Completed prompt {i+1}/{len(prompts)}")
            
        except Exception as e:
            print(f"❌ Failed prompt {i+1}: {e}")
            responses.append({
                "prompt": prompt,
                "error": str(e),
                "index": i
            })
    
    print(f"\nProcessed {len(responses)} requests")
    successful = len([r for r in responses if 'response' in r])
    print(f"Successful: {successful}/{len(prompts)}")

def setup_openai_aliases():
    """Instructions for setting up OpenAI model aliases."""
    print("\n=== Setting up OpenAI Model Aliases ===")
    
    print("To set up aliases for OpenAI compatibility, run these commands:")
    print("ollama cp llama3.2 gpt-3.5-turbo")
    print("ollama cp llama3.2:70b gpt-4") 
    print("ollama cp all-minilm text-embedding-ada-002")
    print("\nThis allows you to use familiar OpenAI model names")

def error_handling_example():
    """Example of proper error handling."""
    print("\n=== Error Handling Example ===")
    
    try:
        # Try with non-existent model
        response = client.chat.completions.create(
            model="non-existent-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("Response:", response.choices[0].message.content)
        
    except Exception as e:
        print(f"Expected error: {type(e).__name__}: {e}")
        
        # Fallback to working model
        try:
            response = client.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": "Hello"}]
            )
            print("Fallback response:", response.choices[0].message.content)
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")

def main():
    """Run all examples."""
    print("OpenAI Compatibility Examples with Ollama")
    print("=" * 50)
    
    try:
        basic_chat_example()
        structured_outputs_example()
        vision_example()
        streaming_example()
        completions_example()
        models_example()
        embeddings_example()
        advanced_chat_example()
        function_calling_example()
        batch_processing_example()
        error_handling_example()
        setup_openai_aliases()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running and models are available")

if __name__ == "__main__":
    main()
