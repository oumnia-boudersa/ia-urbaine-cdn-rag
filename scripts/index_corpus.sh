#!/bin/bash
# Adapted from reference code for dynamic_rags project
# Index corpus files using Pyserini

echo "Creating Pyserini BM25 indexes..."
echo "=================================="

# Define datasets
DATASETS=("211data" "macommunaute" "combined")

# Create indexes for each dataset
for dataset in "${DATASETS[@]}"
do
  echo ""
  echo "Indexing $dataset..."
  
  python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input data/pyserini_corpus/$dataset \
    --index data/pyserini_indexes/$dataset \
    --generator DefaultLuceneDocumentGenerator \
    --storePositions --storeDocvectors --storeRaw \
    --threads 4
  
  if [ $? -eq 0 ]; then
    echo "✓ Successfully indexed $dataset"
  else
    echo "✗ Failed to index $dataset"
  fi
done

echo ""
echo "=================================="
echo "✓ Indexing complete!"
echo "=================================="
echo ""
echo "Indexes created in: data/pyserini_indexes/"
echo "Next step: Use search.sh to query the indexes"
