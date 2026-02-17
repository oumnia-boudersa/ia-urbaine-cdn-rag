#!/usr/bin/env python3
"""
Script for searching index
"""
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sparse_index import SparseIndexer
from dense_index import DenseIndexer


def main():
    parser = argparse.ArgumentParser(description='Search document index')
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Query text'
    )
    parser.add_argument(
        '--index-dir',
        type=str,
        default='indexes',
        help='Index directory (default: indexes)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top results to return (default: 5)'
    )
    parser.add_argument(
        '--sparse-only',
        action='store_true',
        help='Use only sparse index'
    )
    parser.add_argument(
        '--dense-only',
        action='store_true',
        help='Use only dense index'
    )
    
    args = parser.parse_args()
    
    index_dir = Path(args.index_dir)
    
    results = {}
    
    # Load and search sparse index
    if not args.dense_only:
        sparse_path = index_dir / 'sparse_index.pkl'
        if sparse_path.exists():
            print("Loading sparse index...")
            sparse_indexer = SparseIndexer()
            sparse_indexer.load(str(sparse_path))
            
            print(f"\nSparse index search results (query: '{args.query}'):")
            print("-" * 50)
            sparse_results = sparse_indexer.search(args.query, top_k=args.top_k)
            results['sparse'] = sparse_results
            
            for i, (chunk, score) in enumerate(sparse_results, 1):
                print(f"\nResult {i} (score: {score:.4f}):")
                print(f"File: {chunk['file_name']}")
                print(f"Chunk ID: {chunk['chunk_id']}")
                print(f"Content preview: {chunk['content'][:200]}...")
        else:
            print(f"Warning: Sparse index file not found: {sparse_path}")
    
    # Load and search dense index
    if not args.sparse_only:
        dense_path = index_dir / 'dense_index.pkl'
        if dense_path.exists():
            print("\nLoading dense index...")
            dense_indexer = DenseIndexer()
            dense_indexer.load(str(dense_path))
            
            print(f"\nDense index search results (query: '{args.query}'):")
            print("-" * 50)
            dense_results = dense_indexer.search(args.query, top_k=args.top_k)
            results['dense'] = dense_results
            
            for i, (chunk, score) in enumerate(dense_results, 1):
                print(f"\nResult {i} (similarity: {score:.4f}):")
                print(f"File: {chunk['file_name']}")
                print(f"Chunk ID: {chunk['chunk_id']}")
                print(f"Content preview: {chunk['content'][:200]}...")
        else:
            print(f"Warning: Dense index file not found: {dense_path}")
    
    if not results:
        print("Error: No index files found")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
