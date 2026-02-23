"""
Sparse index module - builds sparse index using BM25 algorithm
"""
import pickle
import os
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba


class SparseIndexer:
    """Sparse indexer - supports BM25 and TF-IDF"""
    
    def __init__(self, method: str = 'bm25', language: str = 'zh'):
        """
        Initialize sparse indexer
        
        Args:
            method: Indexing method ('bm25' or 'tfidf')
            language: Language ('zh' Chinese, 'en' English, 'fr' French)
        """
        self.method = method
        self.language = language
        self.index = None
        self.chunks = []
        self.tokenizer = self._get_tokenizer()
    
    def _get_tokenizer(self):
        """Get tokenizer"""
        if self.language == 'zh':
            try:
                import jieba
                return lambda text: jieba.lcut(text)
            except ImportError:
                print("Warning: jieba not installed, using simple tokenization")
                return lambda text: list(text)
        elif self.language == 'fr':
            # French tokenization: use simple space/punctuation splitting, or spacy
            try:
                import spacy
                try:
                    nlp = spacy.load("fr_core_news_sm")
                    return lambda text: [token.text.lower() for token in nlp(text) if not token.is_punct and not token.is_space]
                except OSError:
                    print("Warning: spacy French model not installed, using simple tokenization")
                    print("Installation: python -m spacy download fr_core_news_sm")
                    return lambda text: text.lower().replace(',', ' ').replace('.', ' ').replace(';', ' ').split()
            except ImportError:
                # If spacy not available, use simple tokenization
                return lambda text: text.lower().replace(',', ' ').replace('.', ' ').replace(';', ' ').replace(':', ' ').split()
        else:  # 'en' or default
            return lambda text: text.lower().split()
    
    def build_index(self, chunks: List[Dict[str, any]]):
        """
        Build sparse index
        
        Args:
            chunks: List of document chunks
        """
        self.chunks = chunks
        texts = [chunk['content'] for chunk in chunks]
        
        # Tokenize
        tokenized_texts = [self.tokenizer(text) for text in texts]
        
        if self.method == 'bm25':
            self.index = BM25Okapi(tokenized_texts)
        elif self.method == 'tfidf':
            # Rejoin tokenized texts
            texts_for_tfidf = [' '.join(tokens) for tokens in tokenized_texts]
            self.vectorizer = TfidfVectorizer(max_features=10000)
            self.index = self.vectorizer.fit_transform(texts_for_tfidf)
        else:
            raise ValueError(f"Unsupported indexing method: {self.method}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, any], float]]:
        """
        Search for relevant documents
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (document chunk, relevance score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built, please call build_index first")
        
        query_tokens = self.tokenizer(query)
        
        if self.method == 'bm25':
            scores = self.index.get_scores(query_tokens)
        elif self.method == 'tfidf':
            query_vector = self.vectorizer.transform([' '.join(query_tokens)])
            scores = np.array((self.index * query_vector.T).toarray()).flatten()
        
        # Get top_k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return results with scores
                results.append((self.chunks[idx], float(scores[idx])))
        
        return results
    
    def save(self, file_path: str):
        """Save index to file"""
        save_data = {
            'method': self.method,
            'language': self.language,
            'chunks': self.chunks,
            'index': self.index
        }
        
        if self.method == 'tfidf':
            save_data['vectorizer'] = self.vectorizer
        
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Sparse index saved to: {file_path}")
    
    def load(self, file_path: str):
        """Load index from file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Index file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.method = save_data['method']
        self.language = save_data['language']
        self.chunks = save_data['chunks']
        self.index = save_data['index']
        
        if self.method == 'tfidf':
            self.vectorizer = save_data['vectorizer']
        
        self.tokenizer = self._get_tokenizer()
        print(f"Sparse index loaded from {file_path}")
