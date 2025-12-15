"""
Simple RAG Example with Basic Vector Search
Chapter 15: RAG Implementation - Simplified Version
"""

import os
import json
import asyncio
import logging
import sqlite3
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import re

try:
    import httpx
except ImportError:
    print("httpx not installed. Install with: pip install httpx")
    httpx = None

try:
    import numpy as np
except ImportError:
    print("numpy not installed. Install with: pip install numpy")
    np = None


class SimpleDocumentProcessor:
    """Basic document processor without external dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Error reading text file {file_path}: {e}")
            return ""
    
    def process_document(self, file_path: str) -> str:
        """Process a document (currently supports only text files)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension in ['.txt', '.md']:
            return self.extract_text_from_txt(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {extension}. Use .txt or .md files.")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence or paragraph boundaries
            if end < len(text):
                # Look for sentence ending within the last 100 characters
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                
                break_point = max(last_period, last_newline)
                if break_point > start + chunk_size // 2:
                    end = break_point + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks


class SimpleVectorStore:
    """Simple vector store using basic keyword matching."""
    
    def __init__(self, db_path: str = "simple_vector_store.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for document storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                keywords TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple regex."""
        # Remove punctuation and convert to lowercase
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words and filter out short words and common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }
        
        words = clean_text.split()
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Return unique keywords
        return list(set(keywords))
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to the vector store."""
        if ids is None:
            ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for i, text in enumerate(texts):
                keywords = self.extract_keywords(text)
                keyword_string = " ".join(keywords)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO documents (id, content, metadata, keywords)
                    VALUES (?, ?, ?, ?)
                """, (
                    ids[i],
                    text,
                    json.dumps(metadatas[i]),
                    keyword_string
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added {len(texts)} documents to vector store")
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar documents using keyword matching."""
        try:
            query_keywords = self.extract_keywords(query)
            
            if not query_keywords:
                return []
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create a query that counts keyword matches
            keyword_conditions = []
            params = []
            
            for keyword in query_keywords:
                keyword_conditions.append("keywords LIKE ?")
                params.append(f"%{keyword}%")
            
            # Build query to find documents with keyword matches
            where_clause = " OR ".join(keyword_conditions)
            
            cursor.execute(f"""
                SELECT id, content, metadata, keywords,
                       ({" + ".join([f"CASE WHEN keywords LIKE ? THEN 1 ELSE 0 END" for _ in query_keywords])}) as score
                FROM documents
                WHERE {where_clause}
                ORDER BY score DESC, id
                LIMIT ?
            """, params + params + [n_results])
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                doc_id, content, metadata_json, keywords, score = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                results.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "score": score,
                    "keywords": keywords
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching vector store: {e}")
            return []


class SimpleRAGSystem:
    """Simple RAG system using basic keyword matching."""
    
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model_name: str = "llama2",
        vector_store_path: str = "simple_rag.db"
    ):
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.document_processor = SimpleDocumentProcessor()
        self.vector_store = SimpleVectorStore(vector_store_path)
        self.logger = logging.getLogger(__name__)
        
        # Setup database for conversation history
        self.setup_conversation_db()
    
    def setup_conversation_db(self):
        """Setup SQLite database for conversation history."""
        self.conversation_db_path = "simple_rag_conversations.db"
        conn = sqlite3.connect(self.conversation_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                retrieved_docs TEXT,
                response TEXT,
                model_used TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_documents_from_directory(self, directory_path: str):
        """Add all supported documents from a directory to the RAG system."""
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        supported_extensions = ['.txt', '.md']
        files = []
        
        for ext in supported_extensions:
            files.extend(directory.glob(f"**/*{ext}"))
        
        if not files:
            self.logger.warning(f"No supported files found in {directory}")
            return
        
        self.logger.info(f"Processing {len(files)} files from {directory}")
        
        all_chunks = []
        all_metadata = []
        
        for file_path in files:
            try:
                self.logger.info(f"Processing: {file_path}")
                text = self.document_processor.process_document(str(file_path))
                
                if not text.strip():
                    self.logger.warning(f"No text extracted from {file_path}")
                    continue
                
                chunks = self.document_processor.chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": str(file_path),
                        "chunk_index": i,
                        "file_size": file_path.stat().st_size,
                        "processed_at": datetime.now().isoformat()
                    })
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if all_chunks:
            self.vector_store.add_documents(all_chunks, all_metadata)
            self.logger.info(f"Added {len(all_chunks)} chunks to vector store")
        else:
            self.logger.warning("No chunks were successfully processed")
    
    async def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama API."""
        if httpx is None:
            raise ImportError("httpx is required. Install with: pip install httpx")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
                
            except Exception as e:
                self.logger.error(f"Error generating response: {e}")
                raise
    
    def retrieve_relevant_docs(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        try:
            results = self.vector_store.search(query, n_results)
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []
    
    def create_rag_prompt(self, query: str, relevant_docs: List[Dict]) -> str:
        """Create a prompt that includes the query and relevant context."""
        if not relevant_docs:
            return f"Question: {query}\nAnswer:"
        
        context = "\n\n".join([
            f"Document {i+1} (from {doc['metadata'].get('source', 'unknown')}):\n{doc['content']}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        prompt = f"""Based on the following context documents, please answer the question. If the answer cannot be found in the provided context, please say so.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    async def query(self, question: str, n_docs: int = 3) -> Dict[str, Any]:
        """Main RAG query method."""
        try:
            # Retrieve relevant documents
            self.logger.info(f"Retrieving relevant documents for: {question}")
            relevant_docs = self.retrieve_relevant_docs(question, n_docs)
            
            # Create RAG prompt
            rag_prompt = self.create_rag_prompt(question, relevant_docs)
            
            # Generate response
            response = await self.generate_response(rag_prompt)
            
            # Store conversation
            self.store_conversation(question, relevant_docs, response)
            
            return {
                "question": question,
                "answer": response,
                "retrieved_docs": relevant_docs,
                "model_used": self.model_name,
                "num_docs_found": len(relevant_docs)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise
    
    def store_conversation(self, query: str, retrieved_docs: List[Dict], response: str):
        """Store conversation in database."""
        try:
            conn = sqlite3.connect(self.conversation_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO conversations (query, retrieved_docs, response, model_used)
                VALUES (?, ?, ?, ?)
            """, (
                query,
                json.dumps(retrieved_docs),
                response,
                self.model_name
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing conversation: {e}")


# Example usage
async def main():
    """Example usage of the simple RAG system."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAG system
    rag = SimpleRAGSystem(
        model_name="llama2"
    )
    
    # Create sample documents for testing
    sample_docs_dir = Path("sample_documents")
    sample_docs_dir.mkdir(exist_ok=True)
    
    # Create sample text files
    sample_content = {
        "ai_basics.txt": """
        Artificial Intelligence (AI) is a branch of computer science that aims to create 
        intelligent machines that can perform tasks that typically require human intelligence.
        These tasks include learning, reasoning, problem-solving, perception, and language understanding.
        
        Machine Learning is a subset of AI that focuses on the development of algorithms and 
        statistical models that enable computers to improve their performance on a specific 
        task through experience.
        """,
        "ollama_info.txt": """
        Ollama is a tool that allows you to run large language models locally on your machine.
        It supports various models including Llama 2, Code Llama, and Mistral.
        
        Ollama provides a simple API for generating text and can be used for various applications
        including chatbots, content generation, and code assistance.
        
        The tool is designed to be easy to use and requires minimal setup to get started.
        """
    }
    
    # Write sample files
    for filename, content in sample_content.items():
        sample_file = sample_docs_dir / filename
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # Add documents to RAG system
    print("Adding documents to RAG system...")
    rag.add_documents_from_directory(str(sample_docs_dir))
    
    # Example queries
    queries = [
        "What is Artificial Intelligence?",
        "Tell me about Ollama",
        "How does machine learning work?",
        "What models does Ollama support?"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        try:
            result = await rag.query(query)
            print(f"Answer: {result['answer']}")
            print(f"Documents found: {result['num_docs_found']}")
            
            if result['retrieved_docs']:
                print("\nRelevant documents:")
                for i, doc in enumerate(result['retrieved_docs']):
                    print(f"  {i+1}. {doc['metadata'].get('source', 'Unknown')} (score: {doc.get('score', 0)})")
            
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
