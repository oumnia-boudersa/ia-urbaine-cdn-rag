#!/usr/bin/env python3
"""
Build index from pre-chunked JSON files
"""
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from index_pipeline import IndexPipeline


def main():
    parser = argparse.ArgumentParser(description='Build index from pre-chunked JSON files')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input path (JSON file or directory containing JSON files)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='indexes',
        help='Output directory (default: indexes)'
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
        '--no-recursive',
        action='store_true',
        help='Do not recursively process subdirectories'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Building Index from Pre-chunked JSON Files")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Language: {args.language}")
    print("=" * 60)
    
    # Create indexing pipeline
    pipeline = IndexPipeline(
        sparse_method=args.sparse_method,
        dense_model=args.dense_model,
        language=args.language
    )
    
    # Load pre-chunked data (skip chunking step)
    pipeline.process_documents(
        args.input,
        recursive=not args.no_recursive,
        is_prechunked=True
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
