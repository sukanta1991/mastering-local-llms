# Ollama Book - Code Examples and Applications

This repository contains all the code examples, applications, and scripts from the comprehensive Ollama book. The examples are organized by chapter and demonstrate various aspects of working with Ollama, from basic CLI operations to advanced applications with RAG, multimodal capabilities, and production deployment.

## 📚 Book Structure

This code repository accompanies a comprehensive book about Ollama that covers:

- **Chapters 1-6**: Getting started with Ollama, model management, and basic conversations
- **Chapters 7-10**: CLI fundamentals, advanced operations, and scripting automation  
- **Chapters 11-14**: REST API programming, building applications, mobile development, and advanced AI features
- **Chapters 15-17**: RAG implementation, multimodal models, and performance optimization
- **Chapters 18-24**: Docker deployment, scaling solutions, monitoring, enterprise deployment, and troubleshooting

## 🗂️ Repository Structure

```
code/
├── chapter10_scripting_automation/     # Python automation scripts
├── chapter11_rest_api/                 # REST API examples
├── chapter15_rag_implementation/       # RAG system implementations
├── chapter16_multimodal/               # Multimodal model examples
├── chapter18_docker_deployment/        # Docker deployment scripts
├── deployment/                         # Production deployment configs
├── mobile_examples/                    # Mobile development examples
├── shell_scripts/                      # Shell scripts for automation
└── web_applications/                   # Web application examples
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher (for web applications)
- Docker (for deployment examples)
- Ollama installed and running

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ollama-book/examples.git
   cd examples
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Node.js dependencies (for web examples):**
   ```bash
   npm install
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Ensure Ollama is running:**
   ```bash
   ollama serve
   ```

## 📖 Chapter Examples

### Chapter 10: Scripting Automation

**Ollama Client (`ollama_client.py`)**
```bash
python chapter10_scripting_automation/ollama_client.py
```

**AI Workflow Manager (`ai_workflow_manager.py`)**
```bash
python chapter10_scripting_automation/ai_workflow_manager.py
```

### Chapter 11: REST API Programming

**Flask API Server (`flask_api_server.py`)**
```bash
python chapter11_rest_api/flask_api_server.py
```

**OpenAI Compatibility Layer (`openai_compatibility.py`)**
```bash
python chapter11_rest_api/openai_compatibility.py
```

### Chapter 15: RAG Implementation

**Advanced RAG System (`rag_system.py`)**
```bash
python chapter15_rag_implementation/rag_system.py
```

**Simple RAG System (`simple_rag.py`)**
```bash
python chapter15_rag_implementation/simple_rag.py
```

### Chapter 16: Multimodal Models

**Multimodal Chat Application (`multimodal_chat.py`)**
```bash
python chapter16_multimodal/multimodal_chat.py
```

### Chapter 18: Docker Deployment

**Docker Manager (`docker_manager.py`)**
```bash
python chapter18_docker_deployment/docker_manager.py
```

### Web Applications

**Flask Chat Web App (`chat_web_app.py`)**
```bash
python web_applications/chat_web_app.py
```

### Shell Scripts

**CLI Fundamentals (`ollama-cli-fundamentals.sh`)**
```bash
chmod +x shell_scripts/ollama-cli-fundamentals.sh
./shell_scripts/ollama-cli-fundamentals.sh
```

**Model Management (`model-management.sh`)**
```bash
chmod +x shell_scripts/model-management.sh
./shell_scripts/model-management.sh help
```

**Performance Testing (`performance-test.sh`)**
```bash
chmod +x shell_scripts/performance-test.sh
./shell_scripts/performance-test.sh full-test llama2
```

### Mobile Examples

**Mobile Client (`mobile_client.py`)**
```bash
python mobile_examples/mobile_client.py
```

## 🛠️ Key Features

### Core Python Client
- Asynchronous Ollama API client
- Error handling and retry logic
- Model management utilities
- Conversation history tracking

### Web Applications
- Flask-based chat interface
- REST API endpoints
- WebSocket support for real-time chat
- Mobile-responsive design

### RAG Implementation
- Document processing (PDF, DOCX, TXT)
- Vector storage with ChromaDB
- Similarity search and retrieval
- Context-aware response generation

### Multimodal Support
- Image processing and analysis
- Vision model integration
- Multiple image format support
- Interactive CLI interface

### Automation Scripts
- Shell scripts for common operations
- Performance testing utilities
- Model management automation
- Docker deployment helpers

## 📋 Dependencies

### Python Dependencies
- `httpx`: Async HTTP client for API calls
- `numpy`, `pandas`: Data processing
- `sentence-transformers`: Text embeddings
- `chromadb`: Vector database
- `flask`: Web framework
- `pillow`: Image processing
- `PyPDF2`, `python-docx`: Document processing

### Node.js Dependencies
- `express`: Web server framework
- `axios`: HTTP client
- `socket.io`: WebSocket support
- `cors`: Cross-origin resource sharing

## 🔧 Configuration

Key configuration options in `.env`:

```env
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

# Web Application
WEB_PORT=5000
WEB_SECRET_KEY=your-secret-key

# RAG Settings
RAG_CHUNK_SIZE=1000
RAG_MAX_DOCS=5

# Performance Settings
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=150
```

## 🧪 Testing

Run the test suite:
```bash
# Python tests
python -m pytest tests/

# Node.js tests  
npm test

# Integration tests
./shell_scripts/run-integration-tests.sh
```

## 🐳 Docker Deployment

Build and run with Docker:
```bash
# Build image
docker build -t ollama-examples .

# Run container
docker run -p 5000:5000 -e OLLAMA_HOST=http://host.docker.internal:11434 ollama-examples

# Or use Docker Compose
docker-compose up -d
```

## 📱 Mobile Development

The repository includes examples for:
- React Native integration
- Flutter/Dart client
- Mobile-optimized API endpoints
- Offline capability patterns

## ⚡ Performance Optimization

Performance testing and optimization tools:
- Benchmark scripts for model comparison
- Memory usage monitoring
- Response time analysis
- Concurrent request testing

## 🔐 Security Considerations

- Environment variable configuration
- API rate limiting
- Input validation and sanitization
- CORS configuration
- Authentication examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Related Resources

- [Ollama Official Documentation](https://ollama.ai/docs)
- [Ollama GitHub Repository](https://github.com/jmorganca/ollama)
- [Model Library](https://ollama.ai/library)

## 🆘 Support

If you encounter any issues:

1. Check the [troubleshooting guide](docs/TROUBLESHOOTING.md)
2. Review the [FAQ](docs/FAQ.md)
3. Search existing [issues](https://github.com/ollama-book/examples/issues)
4. Create a new issue with detailed information

## 🏷️ Version Information

- **Version**: 1.0.0
- **Ollama Compatibility**: 0.1.0+
- **Python**: 3.8+
- **Node.js**: 16+

---

**Happy coding with Ollama! 🦙**
