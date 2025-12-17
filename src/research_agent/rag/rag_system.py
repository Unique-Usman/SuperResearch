"""
RAG (Retrieval Augmented Generation) System
Handles document embedding, storage, and retrieval
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
from ..tools.search_tools import SearchResult


@dataclass
class Document:
    """Document with embedding"""
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingModel:
    """Handles text embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Sentence transformer model name
                       all-MiniLM-L6-v2: Fast, lightweight (384 dim)
                       all-mpnet-base-v2: Higher quality (768 dim)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed batch of texts"""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Embed documents in place"""
        texts = [doc.content for doc in documents]
        embeddings = self.embed_batch(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        return documents


class VectorStore:
    """FAISS-based vector store for document retrieval"""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize vector store
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[Document] = []
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the store"""
        # Extract embeddings
        embeddings = np.array([doc.embedding for doc in documents]).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (document, distance) tuples
        """
        if len(self.documents) == 0:
            return []
        
        # Ensure correct shape
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding, k)
        
        # Return documents with scores
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                # Convert L2 distance to similarity score (0-1, higher is better)
                score = 1 / (1 + distance)
                results.append((self.documents[idx], score))
        
        return results
    
    def clear(self):
        """Clear all documents"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []


class RAGSystem:
    """Complete RAG system with embedding and retrieval"""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize RAG system
        
        Args:
            embedding_model: Name of sentence transformer model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model = EmbeddingModel(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = VectorStore(self.embedding_model.dimension)
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """
        Split text into chunks
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of Document chunks
        """
        if metadata is None:
            metadata = {}
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunks.append(Document(
                content=chunk_text,
                metadata={**metadata, "chunk_start": start, "chunk_end": end}
            ))
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def add_search_results(self, results: List[SearchResult]):
        """
        Add search results to RAG system
        
        Args:
            results: List of search results
        """
        all_documents = []
        
        for result in results:
            # Chunk each result
            chunks = self.chunk_text(
                result.content,
                metadata={
                    "title": result.title,
                    "url": result.url,
                    "source": result.source,
                    "score": result.score,
                    **result.metadata
                }
            )
            all_documents.extend(chunks)
        
        # Embed documents
        self.embedding_model.embed_documents(all_documents)
        
        # Add to vector store
        self.vector_store.add_documents(all_documents)
        
        print(f"Added {len(results)} search results ({len(all_documents)} chunks)")
    
    def retrieve(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        # Embed query
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search
        return self.vector_store.search(query_embedding, k=k)
    
    def retrieve_with_multiple_queries(
        self,
        queries: List[str],
        k_per_query: int = 10,
        combine_strategy: str = "union"
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using multiple queries
        
        Args:
            queries: List of query strings
            k_per_query: Number of docs to retrieve per query
            combine_strategy: "union" or "intersection"
            
        Returns:
            Combined list of (document, score) tuples
        """
        all_results = []
        seen_content = set()
        
        for query in queries:
            results = self.retrieve(query, k=k_per_query)
            
            for doc, score in results:
                content_hash = hash(doc.content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_results.append((doc, score))
        
        # Sort by score
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return all_results
    
    def get_context_for_generation(
        self,
        queries: List[str],
        max_tokens: int = 4000,
        tokens_per_char: float = 0.25
    ) -> Tuple[str, List[Dict]]:
        """
        Get context string for LLM generation
        
        Args:
            queries: List of queries
            max_tokens: Maximum tokens for context
            tokens_per_char: Estimated tokens per character
            
        Returns:
            Tuple of (context_string, source_citations)
        """
        max_chars = int(max_tokens / tokens_per_char)
        
        # Retrieve documents
        results = self.retrieve_with_multiple_queries(queries, k_per_query=5)
        
        # Build context
        context_parts = []
        citations = []
        current_length = 0
        
        for i, (doc, score) in enumerate(results):
            chunk_text = f"[Source {i+1}]\n{doc.content}\n"
            
            if current_length + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
            
            # Add citation
            citations.append({
                "id": i + 1,
                "title": doc.metadata.get("title", "Unknown"),
                "url": doc.metadata.get("url", ""),
                "source": doc.metadata.get("source", "unknown"),
                "score": score
            })
        
        context = "\n".join(context_parts)
        
        return context, citations
    
    def clear(self):
        """Clear all stored documents"""
        self.vector_store.clear()


# Utility function
def create_rag_system(
    search_results: List[SearchResult],
    embedding_model: str = "all-MiniLM-L6-v2"
) -> RAGSystem:
    """
    Create and populate RAG system from search results
    
    Args:
        search_results: List of search results
        embedding_model: Embedding model to use
        
    Returns:
        Populated RAG system
    """
    rag = RAGSystem(embedding_model=embedding_model)
    rag.add_search_results(search_results)
    return rag
