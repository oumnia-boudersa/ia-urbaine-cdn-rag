#!/usr/bin/env python3
"""
Script specifically for building index for macommunaute data
"""
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from index_pipeline import IndexPipeline


def main():
    parser = argparse.ArgumentParser(description='Build index for macommunaute data')
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/macommunaute',
        help='Input path (default: data/raw/macommunaute)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='indexes',
        help='Output directory (default: indexes)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=800,
        help='Chunk size (default: 800, suitable for organization info)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Chunk overlap size (default: 50)'
    )
    parser.add_argument(
        '--sparse-method',
        type=str,
        default='bm25',
        choices=['bm25', 'tfidf'],
        help='Sparse indexing method (default: bm25)'
    )
    parser.add_argument(
        '--dense-model',
        type=str,
        default='paraphrase-multilingual-MiniLM-L12-v2',
        help='Dense indexing model (default: paraphrase-multilingual-MiniLM-L12-v2, multilingual support)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='auto',
        choices=['fr', 'en', 'auto'],
        help='Language setting (default: auto, auto-detect)'
    )
    parser.add_argument(
        '--use-org-chunking',
        action='store_true',
        default=True,
        help='Split by organization boundaries (default: True)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Macommunaute Data Index Building")
    print("=" * 60)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Chunking method: {'Organization-based' if args.use_org_chunking else 'Standard'}")
    print(f"Language setting: {args.language}")
    print("=" * 60)
    
    # Create indexing pipeline
    pipeline = IndexPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sparse_method=args.sparse_method,
        dense_model=args.dense_model,
        language=args.language,
        chunk_method='organization' if args.use_org_chunking else 'recursive',
        split_by_organization=args.use_org_chunking
    )
    
    # Process documents
    pipeline.process_documents(
        args.input,
        recursive=True
    )
    
    # Build indices
    pipeline.build_all_indices()
    
    # Save indices
    pipeline.save_indices(args.output)
    
    print("\n" + "=" * 60)
    print("Index building completed!")
    print("=" * 60)
    print(f"\nIndex files saved to: {args.output}/")
    print("\nUse the following command to search:")
    print(f"  python scripts/search_index.py --query 'your query' --index-dir {args.output}")


if __name__ == '__main__':
    main()
