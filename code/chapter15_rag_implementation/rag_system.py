"""
Advanced Ollama Client with RAG Implementation
Chapter 15: RAG Implementation Examples
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import sqlite3
from datetime import datetime

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document
import chromadb
from chromadb.config import Settings


class DocumentProcessor:
    """Handles document processing and text extraction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Error reading text file {file_path}: {e}")
            return ""
    
    def process_document(self, file_path: str) -> str:
        """Process a document based on its file extension."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.extract_text_from_pdf(str(file_path))
        elif extension == '.docx':
            return self.extract_text_from_docx(str(file_path))
        elif extension in ['.txt', '.md']:
            return self.extract_text_from_txt(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
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


class EmbeddingManager:
    """Manages text embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(__name__)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.model.encode(texts)
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.generate_embeddings([text])[0]


class VectorStore:
    """Vector store using ChromaDB for similarity search."""
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.logger = logging.getLogger(__name__)
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to the vector store."""
        if ids is None:
            ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            self.logger.info(f"Added {len(texts)} documents to vector store")
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar documents."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            self.logger.error(f"Error searching vector store: {e}")
            raise
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
            raise


class RAGSystem:
    """Retrieval-Augmented Generation system using Ollama."""
    
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model_name: str = "llama2",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "rag_documents"
    ):
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.vector_store = VectorStore(collection_name)
        self.logger = logging.getLogger(__name__)
        
        # Setup database for conversation history
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for conversation history."""
        self.db_path = "rag_conversations.db"
        conn = sqlite3.connect(self.db_path)
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
        
        supported_extensions = ['.pdf', '.docx', '.txt', '.md']
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
    
    def add_single_document(self, file_path: str):
        """Add a single document to the RAG system."""
        try:
            text = self.document_processor.process_document(file_path)
            
            if not text.strip():
                self.logger.warning(f"No text extracted from {file_path}")
                return
            
            chunks = self.document_processor.chunk_text(text)
            metadata = [{
                "source": file_path,
                "chunk_index": i,
                "processed_at": datetime.now().isoformat()
            } for i in range(len(chunks))]
            
            self.vector_store.add_documents(chunks, metadata)
            self.logger.info(f"Added {len(chunks)} chunks from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error adding document {file_path}: {e}")
            raise
    
    async def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama API."""
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
            
            relevant_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    relevant_docs.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance  # Convert distance to similarity
                    })
            
            return relevant_docs
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []
    
    def create_rag_prompt(self, query: str, relevant_docs: List[Dict]) -> str:
        """Create a prompt that includes the query and relevant context."""
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
            
            if not relevant_docs:
                self.logger.warning("No relevant documents found")
                response = await self.generate_response(f"Question: {question}\nAnswer:")
            else:
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
                "model_used": self.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise
    
    def store_conversation(self, query: str, retrieved_docs: List[Dict], response: str):
        """Store conversation in database."""
        try:
            conn = sqlite3.connect(self.db_path)
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
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, query, response, model_used
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                "timestamp": row[0],
                "query": row[1],
                "response": row[2],
                "model_used": row[3]
            } for row in rows]
            
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {e}")
            return []


# Example usage and testing
async def main():
    """Example usage of the RAG system."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAG system
    rag = RAGSystem(
        model_name="llama2",
        collection_name="example_docs"
    )
    
    # Add documents (uncomment to use)
    # rag.add_documents_from_directory("./documents")
    
    # Example queries
    queries = [
        "What is the main topic discussed in the documents?",
        "Can you summarize the key points?",
        "What are the recommendations mentioned?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        try:
            result = await rag.query(query)
            print(f"Answer: {result['answer']}")
            print(f"Sources used: {len(result['retrieved_docs'])} documents")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
