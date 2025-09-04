"""
Multimodal Chat Application with Ollama
Chapter 16: Multimodal Models Examples
"""

import asyncio
import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any
import io

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("httpx not installed. Install with: pip install httpx")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Pillow not installed. Install with: pip install Pillow")


class ImageProcessor:
    """Handle image processing and encoding for multimodal models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    def is_supported_image(self, file_path: str) -> bool:
        """Check if the file is a supported image format."""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode an image file to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def resize_image_if_needed(self, image_path: str, max_size: int = 1024) -> str:
        """Resize image if it's too large, return path to processed image."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available, returning original image path")
            return image_path
        
        try:
            with Image.open(image_path) as img:
                # Check if resizing is needed
                if max(img.size) <= max_size:
                    return image_path
                
                # Calculate new size maintaining aspect ratio
                ratio = min(max_size / img.width, max_size / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                
                # Resize image
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save to temporary file
                temp_path = Path(image_path).parent / f"resized_{Path(image_path).name}"
                resized_img.save(temp_path, optimize=True, quality=85)
                
                self.logger.info(f"Resized image from {img.size} to {new_size}")
                return str(temp_path)
                
        except Exception as e:
            self.logger.error(f"Error resizing image {image_path}: {e}")
            return image_path
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get information about an image file."""
        try:
            file_path = Path(image_path)
            file_size = file_path.stat().st_size
            
            info = {
                "path": str(file_path),
                "name": file_path.name,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "format": file_path.suffix.lower(),
                "mime_type": mimetypes.guess_type(str(file_path))[0]
            }
            
            if PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    info.update({
                        "dimensions": img.size,
                        "mode": img.mode,
                        "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                    })
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting image info for {image_path}: {e}")
            return {"error": str(e)}


class MultimodalChat:
    """Multimodal chat interface for Ollama vision models."""
    
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model_name: str = "llava",
        max_image_size: int = 1024
    ):
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.max_image_size = max_image_size
        self.image_processor = ImageProcessor()
        self.logger = logging.getLogger(__name__)
        self.conversation_history = []
    
    async def check_model_availability(self) -> bool:
        """Check if the specified model is available."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for API calls")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                response.raise_for_status()
                
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]
                
                is_available = any(self.model_name in model for model in available_models)
                
                if not is_available:
                    self.logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                
                return is_available
                
        except Exception as e:
            self.logger.error(f"Error checking model availability: {e}")
            return False
    
    async def send_text_message(self, message: str) -> Dict[str, Any]:
        """Send a text-only message to the model."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for API calls")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": message,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                # Store in conversation history
                self.conversation_history.append({
                    "type": "text",
                    "user_message": message,
                    "assistant_response": result.get("response", ""),
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                return {
                    "response": result.get("response", ""),
                    "model": self.model_name,
                    "type": "text_only"
                }
                
        except Exception as e:
            self.logger.error(f"Error sending text message: {e}")
            raise
    
    async def send_image_message(self, message: str, image_path: str) -> Dict[str, Any]:
        """Send a message with an image to the model."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for API calls")
        
        if not self.image_processor.is_supported_image(image_path):
            raise ValueError(f"Unsupported image format: {Path(image_path).suffix}")
        
        try:
            # Get image information
            image_info = self.image_processor.get_image_info(image_path)
            self.logger.info(f"Processing image: {image_info}")
            
            # Resize image if needed
            processed_image_path = self.image_processor.resize_image_if_needed(
                image_path, self.max_image_size
            )
            
            # Encode image to base64
            image_base64 = self.image_processor.encode_image_to_base64(processed_image_path)
            
            # Send request to Ollama
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": message,
                        "images": [image_base64],
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                # Store in conversation history
                self.conversation_history.append({
                    "type": "multimodal",
                    "user_message": message,
                    "image_path": image_path,
                    "image_info": image_info,
                    "assistant_response": result.get("response", ""),
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                # Clean up temporary resized image if created
                if processed_image_path != image_path:
                    try:
                        Path(processed_image_path).unlink()
                    except Exception:
                        pass
                
                return {
                    "response": result.get("response", ""),
                    "model": self.model_name,
                    "type": "multimodal",
                    "image_info": image_info
                }
                
        except Exception as e:
            self.logger.error(f"Error sending image message: {e}")
            raise
    
    async def analyze_image(self, image_path: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze an image with predefined analysis prompts."""
        analysis_prompts = {
            "general": "Describe what you see in this image in detail.",
            "objects": "List all the objects you can identify in this image.",
            "people": "Describe the people in this image, including their appearance and activities.",
            "scene": "Describe the scene, setting, and context of this image.",
            "colors": "Describe the colors, lighting, and visual style of this image.",
            "text": "Read and transcribe any text visible in this image.",
            "technical": "Analyze the technical aspects of this image, such as composition, quality, and camera settings if apparent."
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        
        return await self.send_image_message(prompt, image_path)
    
    async def compare_images(self, image1_path: str, image2_path: str, comparison_prompt: str = None) -> Dict[str, Any]:
        """Compare two images (requires sending them separately as most models don't support multiple images)."""
        if not comparison_prompt:
            comparison_prompt = "Describe this image in detail."
        
        # Analyze first image
        result1 = await self.send_image_message(
            f"{comparison_prompt} This is the first image.",
            image1_path
        )
        
        # Analyze second image
        result2 = await self.send_image_message(
            f"{comparison_prompt} This is the second image. Compare it with the previous image I showed you.",
            image2_path
        )
        
        return {
            "image1_analysis": result1,
            "image2_analysis": result2,
            "comparison_type": "sequential"
        }
    
    def get_conversation_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get conversation history."""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared")
    
    def export_conversation(self, file_path: str):
        """Export conversation history to a JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Conversation exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Error exporting conversation: {e}")
            raise


class MultimodalChatCLI:
    """Command-line interface for multimodal chat."""
    
    def __init__(self, chat: MultimodalChat):
        self.chat = chat
        self.logger = logging.getLogger(__name__)
    
    def print_help(self):
        """Print help information."""
        print("""
Multimodal Chat Commands:
  /help                 - Show this help message
  /image <path>         - Send an image with optional message
  /analyze <path> <type> - Analyze image (types: general, objects, people, scene, colors, text, technical)
  /compare <path1> <path2> - Compare two images
  /history [n]          - Show conversation history (last n messages)
  /clear                - Clear conversation history
  /export <path>        - Export conversation to JSON file
  /model                - Show current model info
  /quit                 - Exit the chat
  
  For regular text messages, just type your message and press Enter.
  For image messages, use: /image /path/to/image.jpg Your question about the image
        """)
    
    async def process_command(self, user_input: str) -> bool:
        """Process user commands. Returns True to continue, False to quit."""
        parts = user_input.strip().split()
        command = parts[0].lower()
        
        try:
            if command == "/help":
                self.print_help()
            
            elif command == "/image":
                if len(parts) < 2:
                    print("Usage: /image <path> [message]")
                    return True
                
                image_path = parts[1]
                message = " ".join(parts[2:]) if len(parts) > 2 else "Describe this image."
                
                if not Path(image_path).exists():
                    print(f"Error: Image file not found: {image_path}")
                    return True
                
                print(f"Analyzing image: {image_path}")
                result = await self.chat.send_image_message(message, image_path)
                print(f"\nAssistant: {result['response']}")
                
                if 'image_info' in result:
                    info = result['image_info']
                    print(f"[Image: {info.get('name', 'unknown')} - {info.get('size_mb', 0)}MB]")
            
            elif command == "/analyze":
                if len(parts) < 2:
                    print("Usage: /analyze <path> [type]")
                    return True
                
                image_path = parts[1]
                analysis_type = parts[2] if len(parts) > 2 else "general"
                
                if not Path(image_path).exists():
                    print(f"Error: Image file not found: {image_path}")
                    return True
                
                print(f"Analyzing image ({analysis_type}): {image_path}")
                result = await self.chat.analyze_image(image_path, analysis_type)
                print(f"\nAssistant: {result['response']}")
            
            elif command == "/compare":
                if len(parts) < 3:
                    print("Usage: /compare <path1> <path2>")
                    return True
                
                image1_path, image2_path = parts[1], parts[2]
                
                for path in [image1_path, image2_path]:
                    if not Path(path).exists():
                        print(f"Error: Image file not found: {path}")
                        return True
                
                print(f"Comparing images: {image1_path} and {image2_path}")
                result = await self.chat.compare_images(image1_path, image2_path)
                
                print(f"\nFirst image analysis: {result['image1_analysis']['response']}")
                print(f"\nSecond image analysis: {result['image2_analysis']['response']}")
            
            elif command == "/history":
                limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                history = self.chat.get_conversation_history(limit)
                
                if not history:
                    print("No conversation history.")
                    return True
                
                print(f"\nConversation History (last {len(history)} messages):")
                for i, entry in enumerate(history, 1):
                    print(f"\n{i}. [{entry['type']}] User: {entry['user_message']}")
                    if 'image_path' in entry:
                        print(f"   Image: {entry['image_path']}")
                    print(f"   Assistant: {entry['assistant_response'][:100]}...")
            
            elif command == "/clear":
                self.chat.clear_conversation_history()
                print("Conversation history cleared.")
            
            elif command == "/export":
                if len(parts) < 2:
                    print("Usage: /export <path>")
                    return True
                
                export_path = parts[1]
                self.chat.export_conversation(export_path)
                print(f"Conversation exported to {export_path}")
            
            elif command == "/model":
                print(f"Current model: {self.chat.model_name}")
                print(f"Ollama host: {self.chat.ollama_host}")
                is_available = await self.chat.check_model_availability()
                print(f"Model available: {is_available}")
            
            elif command == "/quit":
                return False
            
            else:
                print(f"Unknown command: {command}. Type /help for available commands.")
        
        except Exception as e:
            print(f"Error executing command: {e}")
        
        return True
    
    async def run(self):
        """Run the interactive chat."""
        print("Multimodal Chat with Ollama")
        print("Type /help for commands or just start chatting!")
        print(f"Using model: {self.chat.model_name}")
        
        # Check if model is available
        if not await self.chat.check_model_availability():
            print("Warning: Model may not be available. Make sure it's installed in Ollama.")
        
        print("\nChat started. Type /quit to exit.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    should_continue = await self.process_command(user_input)
                    if not should_continue:
                        break
                else:
                    # Regular text message
                    result = await self.chat.send_text_message(user_input)
                    print(f"\nAssistant: {result['response']}")
            
            except KeyboardInterrupt:
                print("\nExiting chat...")
                break
            except Exception as e:
                print(f"Error: {e}")


# Example usage and main function
async def main():
    """Main function to run the multimodal chat example."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the multimodal chat
    chat = MultimodalChat(
        model_name="llava",  # Change to your preferred vision model
        max_image_size=1024
    )
    
    # Create CLI interface
    cli = MultimodalChatCLI(chat)
    
    # Run the interactive chat
    await cli.run()


if __name__ == "__main__":
    if not HTTPX_AVAILABLE:
        print("Error: httpx is required. Install with: pip install httpx")
        exit(1)
    
    asyncio.run(main())
