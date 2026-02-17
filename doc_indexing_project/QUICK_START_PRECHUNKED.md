# Quick Start Guide for Pre-chunked JSON Data

## Your Data Format

Your JSON file (`chunks_211data.json`) contains pre-chunked data with the following structure:

```json
[
  {
    "chunk_id": 1,
    "page_number": 1,
    "text": "chunk content...",
    "category_id": 17,
    "category_name": "Category Name",
    "subcategory": "Subcategory"
  }
]
```

## Quick Start

### Step 1: Build Index from Your Pre-chunked Data

```bash
python scripts/build_from_chunks.py \
    --input data/chuncked/211data/chunks_211data.json \
    --output indexes \
    --language en
```

This will:
- Load all chunks from your JSON file
- Build sparse index (BM25)
- Build dense index (vector embeddings)
- Save all indices to `indexes/` directory

### Step 2: Search Your Index

```bash
python scripts/search_index.py \
    --query "social services" \
    --index-dir indexes \
    --top-k 10
```

### Step 3: Interactive Search

```bash
python scripts/load_and_search.py
```

Then enter your queries interactively.

## Command Options

### build_from_chunks.py

```bash
python scripts/build_from_chunks.py \
    --input <JSON_FILE_OR_DIR> \    # Required: path to JSON file or directory
    --output indexes \               # Optional: output directory (default: indexes)
    --sparse-method bm25 \          # Optional: bm25 or tfidf (default: bm25)
    --dense-model paraphrase-multilingual-MiniLM-L12-v2 \  # Optional: model name
    --language en \                  # Optional: en, fr, zh, or auto (default: auto)
    --no-recursive                   # Optional: don't search subdirectories
```

## Example Workflow

```bash
# 1. Build index
python scripts/build_from_chunks.py \
    --input data/chuncked/211data/chunks_211data.json \
    --output indexes \
    --language en

# 2. Search
python scripts/search_index.py \
    --query "community resources" \
    --index-dir indexes \
    --top-k 5

# 3. Search with specific index type
python scripts/search_index.py \
    --query "health services" \
    --sparse-only \
    --top-k 10
```

## Python API Usage

```python
from src.index_pipeline import IndexPipeline

# Create pipeline
pipeline = IndexPipeline(
    sparse_method='bm25',
    dense_model='paraphrase-multilingual-MiniLM-L12-v2',
    language='en'
)

# Load pre-chunked JSON data
pipeline.process_documents(
    'data/chuncked/211data/chunks_211data.json',
    is_prechunked=True
)

# Build indices
pipeline.build_all_indices()

# Save indices
pipeline.save_indices('indexes/')

# Search
results = pipeline.search("your query", top_k=5)
for chunk, score in results['sparse']:
    print(f"Score: {score:.4f}")
    print(f"Category: {chunk['metadata'].get('category_name')}")
    print(f"Content: {chunk['content'][:200]}...")
```

## Notes

- Your metadata (category_id, category_name, subcategory, page_number) will be preserved
- The `text` field from JSON becomes the `content` field in chunks
- All metadata is accessible via `chunk['metadata']` in search results
- Language detection is not automatic for pre-chunked data, specify `--language en` explicitly
