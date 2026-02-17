#!/usr/bin/env python3
"""
Main script for building index
"""
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from index_pipeline import IndexPipeline


def main():
    parser = argparse.ArgumentParser(description='Build document index')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input path (file or directory)'
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
        default=500,
        help='Chunk size (default: 500)'
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
        help='Dense indexing model (default: paraphrase-multilingual-MiniLM-L12-v2)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='auto',
        choices=['zh', 'en', 'fr', 'auto'],
        help='Language setting (default: auto, auto-detect)'
    )
    parser.add_argument(
        '--chunk-method',
        type=str,
        default='recursive',
        choices=['recursive', 'token', 'organization'],
        help='Chunking method (default: recursive)'
    )
    parser.add_argument(
        '--split-by-org',
        action='store_true',
        help='Split by organization boundaries (for macommunaute format)'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not recursively process subdirectories'
    )
    
    args = parser.parse_args()
    
    # Create indexing pipeline
    pipeline = IndexPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sparse_method=args.sparse_method,
        dense_model=args.dense_model,
        language=args.language,
        chunk_method=args.chunk_method,
        split_by_organization=args.split_by_org
    )
    
    # Process documents
    pipeline.process_documents(
        args.input,
        recursive=not args.no_recursive
    )
    
    # Build indices
    pipeline.build_all_indices()
    
    # Save indices
    pipeline.save_indices(args.output)
    
    print("\n" + "=" * 50)
    print("Index building completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
