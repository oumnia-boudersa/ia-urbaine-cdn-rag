"""
Convert BRIGHT dataset to BEIR-like format for QRHead retrieval.

This script converts BRIGHT data to the same JSON format used by BEIR in QRHead:
[
  {
    "idx": "query_id",
    "question": "query_text",
    "paragraphs": [
      {
        "idx": "doc_id",
        "title": null,
        "paragraph_text": "doc_text",
        "is_supporting": true/false
      },
      ...
    ]
  },
  ...
]
"""
import json
import os
import argparse
from collections import defaultdict


def load_queries(queries_path):
    """Load queries from TSV file."""
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                query_id, query_text = parts[0], parts[1]
                queries[query_id] = query_text
    return queries


def load_qrels(qrels_path):
    """Load relevance judgments from TSV file."""
    qrels = defaultdict(dict)
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                query_id, _, doc_id, relevance = parts[0], parts[1], parts[2], int(parts[3])
                qrels[query_id][doc_id] = relevance
    return qrels


def load_corpus(corpus_path):
    """Load corpus from JSONL file."""
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_id = doc['id']
            doc_text = doc['contents']
            corpus[doc_id] = doc_text
    return corpus


def load_bm25_run(run_path, top_k=200):
    """Load BM25 run file (TREC format)."""
    runs = defaultdict(list)
    with open(run_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id = parts[0]
                doc_id = parts[2]
                rank = int(parts[3])
                if rank <= top_k:
                    runs[query_id].append(doc_id)
    return runs


def convert_bright_to_beir_format(
    dataset_name,
    queries_path,
    qrels_path,
    corpus_path,
    run_path,
    output_path,
    top_k=200
):
    """Convert BRIGHT dataset to BEIR-like JSON format."""
    
    print(f"Loading queries from {queries_path}...")
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} queries")
    
    print(f"Loading qrels from {qrels_path}...")
    qrels = load_qrels(qrels_path)
    print(f"Loaded qrels for {len(qrels)} queries")
    
    print(f"Loading corpus from {corpus_path}...")
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} documents")
    
    print(f"Loading BM25 run from {run_path}...")
    runs = load_bm25_run(run_path, top_k)
    print(f"Loaded runs for {len(runs)} queries")
    
    # Convert to BEIR format
    data = []
    for query_id, query_text in queries.items():
        if query_id not in runs:
            print(f"Warning: No BM25 results for query {query_id}, skipping...")
            continue
        
        paragraphs = []
        for doc_id in runs[query_id]:
            if doc_id not in corpus:
                print(f"Warning: Document {doc_id} not found in corpus, skipping...")
                continue
            
            # Check if document is relevant
            is_supporting = False
            if query_id in qrels and doc_id in qrels[query_id]:
                is_supporting = qrels[query_id][doc_id] > 0
            
            paragraphs.append({
                "idx": doc_id,
                "title": None,
                "paragraph_text": corpus[doc_id],
                "is_supporting": is_supporting
            })
        
        if len(paragraphs) > 0:
            data.append({
                "idx": query_id,
                "question": query_text,
                "paragraphs": paragraphs
            })
    
    print(f"Converted {len(data)} queries to BEIR format")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Convert BRIGHT dataset to BEIR format")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset name (e.g., aops, biology, economics)")
    parser.add_argument("--data_dir", type=str, default="/u/tianyuxi/ReasoningRerank/Datasets/bright/data",
                        help="Path to BRIGHT data directory")
    parser.add_argument("--runs_dir", type=str, default="/u/tianyuxi/ReasoningRerank/Datasets/bright/runs_bm25",
                        help="Path to BM25 runs directory")
    parser.add_argument("--output_dir", type=str, default="/u/tianyuxi/ReasoningRerank/Datasets/bright/beir_format",
                        help="Output directory for converted data")
    parser.add_argument("--top_k", type=int, default=200,
                        help="Number of top documents to include per query")
    parser.add_argument("--use_filtered", action="store_true",
                        help="Use filtered BM25 results instead of raw results")
    
    args = parser.parse_args()
    
    # Construct paths
    queries_path = os.path.join(args.data_dir, "pyserini_queries", f"{args.dataset}.tsv")
    qrels_path = os.path.join(args.data_dir, "pyserini_qrels", f"{args.dataset}.tsv")
    corpus_path = os.path.join(args.data_dir, "pyserini_corpus", args.dataset, f"{args.dataset}.jsonl")
    
    if args.use_filtered:
        run_path = os.path.join(args.runs_dir, f"bm25.{args.dataset}.filtered.trec")
    else:
        run_path = os.path.join(args.runs_dir, f"bm25.{args.dataset}.trec")
    
    output_path = os.path.join(args.output_dir, f"{args.dataset}_bm25_top_{args.top_k}.json")
    
    # Check if files exist
    for path, name in [(queries_path, "queries"), (qrels_path, "qrels"), 
                       (corpus_path, "corpus"), (run_path, "run")]:
        if not os.path.exists(path):
            print(f"Error: {name} file not found: {path}")
            return
    
    convert_bright_to_beir_format(
        args.dataset,
        queries_path,
        qrels_path,
        corpus_path,
        run_path,
        output_path,
        args.top_k
    )


if __name__ == "__main__":
    main()

