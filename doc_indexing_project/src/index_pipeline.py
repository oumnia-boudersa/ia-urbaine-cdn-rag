"""
Complete index building pipeline
"""
import os
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm

import sys
from pathlib import Path

# Ensure modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

from document_loader import DocumentLoader
from chunker import DocumentChunker
from sparse_index import SparseIndexer
from dense_index import DenseIndexer
from chunk_loader import ChunkLoader


class IndexPipeline:
    """Complete document indexing pipeline"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        sparse_method: str = 'bm25',
        dense_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        language: str = 'zh',
        chunk_method: str = 'recursive',
        split_by_organization: bool = False
    ):
        """
        Initialize indexing pipeline
        
        Args:
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap size
            sparse_method: Sparse indexing method ('bm25' or 'tfidf')
            dense_model: Dense indexing model name
            language: Language setting ('zh', 'en', 'fr', 'auto')
            chunk_method: Chunking method ('recursive', 'token', 'organization')
            split_by_organization: Whether to split by organization boundaries
        """
        self.document_loader = DocumentLoader()
        self.chunk_loader = ChunkLoader()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            method=chunk_method,
            split_by_organization=split_by_organization
        )
        self.sparse_indexer = SparseIndexer(
            method=sparse_method,
            language=language
        )
        self.dense_indexer = DenseIndexer(model_name=dense_model)
        
        self.chunks = []
        self.stats = {}
        self.language = language
    
    def process_documents(
        self,
        input_path: str,
        recursive: bool = True,
        is_prechunked: bool = False
    ) -> List[Dict[str, any]]:
        """
        Process documents: load and chunk
        
        Args:
            input_path: Input path (file or directory)
            recursive: Whether to recursively process subdirectories
            is_prechunked: If True, load pre-chunked JSON files instead of raw documents
            
        Returns:
            List of chunked documents
        """
        input_path = Path(input_path)
        
        # Load pre-chunked JSON data
        if is_prechunked:
            print("=" * 50)
            print("Step 1: Loading pre-chunked JSON data")
            print("=" * 50)
            
            if input_path.is_file() and input_path.suffix.lower() == '.json':
                self.chunks = self.chunk_loader.load_json_file(str(input_path))
            else:
                self.chunks = self.chunk_loader.load_json_directory(
                    str(input_path),
                    recursive=recursive
                )
            
            print(f"Loaded {len(self.chunks)} pre-chunked chunks")
            
            # Auto-detect language for pre-chunked data (if set to auto)
            # Note: Language detection from content is not implemented for pre-chunked data
            # Default to English if auto is selected
            if self.language == 'auto':
                self.sparse_indexer.language = 'en'  # Default for pre-chunked data
                self.sparse_indexer.tokenizer = self.sparse_indexer._get_tokenizer()
            
            # Get chunk statistics
            self.stats['chunk_stats'] = self.chunker.get_chunk_stats(self.chunks)
            print(f"Average chunk size: {self.stats['chunk_stats']['avg_chunk_size']:.0f} characters")
            
            return self.chunks
        
        # Load raw documents and chunk them
        print("=" * 50)
        print("Step 1: Loading documents")
        print("=" * 50)
        
        if input_path.is_file():
            documents = [self.document_loader.load_file(str(input_path))]
        else:
            documents = self.document_loader.load_directory(
                str(input_path),
                recursive=recursive
            )
        
        print(f"Loaded {len(documents)} documents")
        
        # Auto-detect language (if set to auto)
        if self.language == 'auto' and documents:
            languages = [doc.get('metadata', {}).get('language', 'unknown') for doc in documents]
            detected_lang = max(set(languages), key=languages.count) if languages else 'en'
            if detected_lang != 'unknown':
                print(f"Detected primary language: {detected_lang}")
                self.sparse_indexer.language = detected_lang
                self.sparse_indexer.tokenizer = self.sparse_indexer._get_tokenizer()
        
        # Display document statistics
        if documents:
            lang_stats = {}
            topic_stats = {}
            for doc in documents:
                lang = doc.get('metadata', {}).get('language', 'unknown')
                topic = doc.get('metadata', {}).get('topic', 'unknown')
                lang_stats[lang] = lang_stats.get(lang, 0) + 1
                topic_stats[topic] = topic_stats.get(topic, 0) + 1
            
            if lang_stats:
                print(f"\nLanguage distribution: {lang_stats}")
            if topic_stats and len(topic_stats) < 20:  # Only show if not too many topics
                print(f"Topic distribution: {dict(list(topic_stats.items())[:10])}")
        
        # Chunking
        print("\n" + "=" * 50)
        print("Step 2: Document chunking")
        print("=" * 50)
        
        self.chunks = self.chunker.chunk_documents(documents)
        self.stats['chunk_stats'] = self.chunker.get_chunk_stats(self.chunks)
        
        print(f"Generated {len(self.chunks)} chunks")
        print(f"Average chunk size: {self.stats['chunk_stats']['avg_chunk_size']:.0f} characters")
        
        return self.chunks
    
    def build_sparse_index(self):
        """Build sparse index"""
        if not self.chunks:
            raise ValueError("Please process documents first (call process_documents)")
        
        print("\n" + "=" * 50)
        print("Step 3: Building sparse index")
        print("=" * 50)
        
        self.sparse_indexer.build_index(self.chunks)
        print("Sparse index built")
    
    def build_dense_index(self):
        """Build dense index"""
        if not self.chunks:
            raise ValueError("Please process documents first (call process_documents)")
        
        print("\n" + "=" * 50)
        print("Step 4: Building dense index")
        print("=" * 50)
        
        self.dense_indexer.build_index(self.chunks)
        print("Dense index built")
    
    def build_all_indices(self):
        """Build all indices"""
        self.build_sparse_index()
        self.build_dense_index()
    
    def save_indices(self, output_dir: str):
        """
        Save all indices
        
        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 50)
        print("Step 5: Saving indices")
        print("=" * 50)
        
        # Save sparse index
        sparse_path = output_dir / 'sparse_index.pkl'
        self.sparse_indexer.save(str(sparse_path))
        
        # Save dense index
        dense_path = output_dir / 'dense_index.pkl'
        self.dense_indexer.save(str(dense_path))
        
        # Save chunks and statistics
        import pickle
        chunks_path = output_dir / 'chunks.pkl'
        with open(chunks_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'stats': self.stats
            }, f)
        
        print(f"\nAll indices saved to: {output_dir}")
        print(f"  - Sparse index: {sparse_path}")
        print(f"  - Dense index: {dense_path}")
        print(f"  - Chunks data: {chunks_path}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_sparse: bool = True,
        use_dense: bool = True
    ) -> Dict[str, List]:
        """
        Hybrid search
        
        Args:
            query: Query text
            top_k: Number of results to return from each index
            use_sparse: Whether to use sparse index
            use_dense: Whether to use dense index
            
        Returns:
            Dictionary of search results
        """
        results = {}
        
        if use_sparse:
            sparse_results = self.sparse_indexer.search(query, top_k=top_k)
            results['sparse'] = sparse_results
        
        if use_dense:
            dense_results = self.dense_indexer.search(query, top_k=top_k)
            results['dense'] = dense_results
        
        return results
