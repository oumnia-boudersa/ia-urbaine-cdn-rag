"""
Dense index module - builds dense index using vector embeddings
"""
import pickle
import os
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss


class DenseIndexer:
    """Dense indexer - uses vector embeddings and FAISS"""
    
    def __init__(
        self,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        index_type: str = 'flat'
    ):
        """
        Initialize dense indexer
        
        Args:
            model_name: sentence-transformers model name
            index_type: FAISS index type ('flat' or 'ivf')
        """
        self.model_name = model_name
        self.index_type = index_type
        self.model = None
        self.index = None
        self.chunks = []
        self.embeddings = None
    
    def _load_model(self):
        """Load embedding model"""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded")
    
    def build_index(self, chunks: List[Dict[str, any]], batch_size: int = 32):
        """
        Build dense index
        
        Args:
            chunks: List of document chunks
            batch_size: Batch size for processing
        """
        self.chunks = chunks
        texts = [chunk['content'] for chunk in chunks]
        
        # Load model
        self._load_model()
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        
        if self.index_type == 'flat':
            # Use Flat index with L2 distance (exact search)
            self.index = faiss.IndexFlatL2(dimension)
        elif self.index_type == 'ivf':
            # Use IVF index (approximate search, faster)
            nlist = min(100, len(chunks) // 10)  # Number of cluster centers
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(self.embeddings)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Add vectors to index
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Index built, dimension: {dimension}, vectors: {len(self.embeddings)}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, any], float]]:
        """
        Search for relevant documents
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (document chunk, similarity score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built, please call build_index first")
        
        # Load model (if not loaded)
        self._load_model()
        
        # Generate query vector
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                # Convert distance to similarity (smaller L2 distance = higher similarity)
                similarity = 1 / (1 + distance)
                results.append((self.chunks[idx], float(similarity)))
        
        return results
    
    def save(self, file_path: str):
        """Save index to file"""
        if self.index is None:
            raise ValueError("Index not built, cannot save")
        
        # Save FAISS index
        index_path = file_path.replace('.pkl', '.faiss')
        faiss.write_index(self.index, index_path)
        
        # Save other data
        save_data = {
            'model_name': self.model_name,
            'index_type': self.index_type,
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'faiss_index_path': index_path
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Dense index saved to: {file_path}")
        print(f"FAISS index saved to: {index_path}")
    
    def load(self, file_path: str):
        """Load index from file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Index file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model_name = save_data['model_name']
        self.index_type = save_data['index_type']
        self.chunks = save_data['chunks']
        self.embeddings = save_data['embeddings']
        
        # Load FAISS index
        index_path = save_data['faiss_index_path']
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        
        print(f"Dense index loaded from {file_path}")
