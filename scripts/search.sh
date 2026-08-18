#!/bin/bash
# Adapted from reference code for dynamic_rags project
# Search indexes using Pyserini BM25

echo "Searching Pyserini indexes..."
echo "=============================="

# Define datasets to search
DATASETS=("211data" "macommunaute" "combined")

# Example queries (replace with your actual queries)
# You'll need to create query files or provide queries programmatically
QUERY="Where can I find housing support?"

# Create output directory
mkdir -p runs_bm25

for dataset in "${DATASETS[@]}"
do
  echo ""
  echo "Searching $dataset..."
  
  # Check if index exists
  if [ ! -d "data/pyserini_indexes/$dataset" ]; then
    echo "⚠ Index not found for $dataset. Run index_corpus.sh first."
    continue
  fi
  
  # Run search
  # Note: --topics expects a TSV file. For single query, use --query flag instead
  python -m pyserini.search.lucene \
    --index data/pyserini_indexes/$dataset \
    --query "$QUERY" \
    --bm25 \
    --hits 10
  
  if [ $? -eq 0 ]; then
    echo "✓ Search completed for $dataset"
  else
    echo "✗ Search failed for $dataset"
  fi
done

echo ""
echo "=============================="
echo "✓ Search complete!"
echo "=============================="
echo ""
echo "To search with query files (TSV format):"
echo "  python -m pyserini.search.lucene \\"
echo "    --index data/pyserini_indexes/combined \\"
echo "    --topics queries.tsv \\"
echo "    --output runs_bm25/bm25.combined.trec \\"
echo "    --bm25 \\"
echo "    --hits 100"
