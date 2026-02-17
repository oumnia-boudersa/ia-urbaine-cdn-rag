#!/usr/bin/env python3
"""
Example script for loading existing index and searching
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from sparse_index import SparseIndexer
    from dense_index import DenseIndexer
except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease install dependencies first:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def main():
    index_dir = Path(__file__).parent.parent / 'indexes'
    
    if not index_dir.exists():
        print(f"Error: Index directory not found: {index_dir}")
        print("Please run build_index.py to build index first")
        return
    
    # Load indices
    print("Loading indices...")
    
    sparse_indexer = None
    dense_indexer = None
    
    sparse_path = index_dir / 'sparse_index.pkl'
    if sparse_path.exists():
        print(f"  Loading sparse index: {sparse_path}")
        sparse_indexer = SparseIndexer()
        sparse_indexer.load(str(sparse_path))
    
    dense_path = index_dir / 'dense_index.pkl'
    if dense_path.exists():
        print(f"  Loading dense index: {dense_path}")
        dense_indexer = DenseIndexer()
        dense_indexer.load(str(dense_path))
    
    if not sparse_indexer and not dense_indexer:
        print("Error: No index files found")
        return
    
    # Interactive search
    print("\n" + "=" * 60)
    print("Interactive Search (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        query = input("\nEnter query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        if sparse_indexer:
            print("\nSparse index results:")
            results = sparse_indexer.search(query, top_k=5)
            for i, (chunk, score) in enumerate(results, 1):
                print(f"\n  {i}. [{score:.4f}] {chunk['file_name']}")
                print(f"     Chunk ID: {chunk['chunk_id']}")
                print(f"     Content: {chunk['content'][:150]}...")
        
        if dense_indexer:
            print("\nDense index results:")
            results = dense_indexer.search(query, top_k=5)
            for i, (chunk, score) in enumerate(results, 1):
                print(f"\n  {i}. [{score:.4f}] {chunk['file_name']}")
                print(f"     Chunk ID: {chunk['chunk_id']}")
                print(f"     Content: {chunk['content'][:150]}...")


if __name__ == '__main__':
    main()
