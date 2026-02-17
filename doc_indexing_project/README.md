# Document Indexing System

A complete document indexing system that supports document loading, chunking, sparse indexing (BM25/TF-IDF), and dense indexing (vector search).

## Features

- 📄 **Multi-format support**: PDF, Word, Excel, Markdown, TXT, HTML
- ✂️ **Smart chunking**: recursive character splitting and token-based splitting
- 🔍 **Sparse indexing**: BM25 or TF-IDF
- 🎯 **Dense indexing**: Sentence Transformers + FAISS
- 💾 **Persistence**: save/load indices for easy sharing
- 🔎 **Hybrid search**: query both sparse and dense indices

## Installation

### 1. Clone or download the project

```bash
cd doc_indexing_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Note: If you work with Chinese documents, installing `jieba` is recommended:

```bash
pip install jieba
```

## Quick start

### 1. Prepare your documents

Put your documents under `data/` (or point the scripts to another directory).

### 2. Build indices

**Option A: Build from raw documents (requires chunking)**

```bash
# Basic usage
python scripts/build_index.py --input data/

# Custom parameters
python scripts/build_index.py \
    --input data/ \
    --output indexes \
    --chunk-size 500 \
    --chunk-overlap 50 \
    --sparse-method bm25 \
    --dense-model paraphrase-multilingual-MiniLM-L12-v2 \
    --language zh
```

**Option B: Build from pre-chunked JSON (recommended if you already have chunked data)**

```bash
# Build from a single JSON file
python scripts/build_from_chunks.py --input data/chuncked/211data/chunks_211data.json

# Build from a directory containing multiple JSON files
python scripts/build_from_chunks.py \
    --input data/chuncked/ \
    --output indexes \
    --language auto
```

**Pre-chunked JSON format**:
```json
[
  {
    "chunk_id": 1,
    "page_number": 1,
    "text": "chunk content here...",
    "category_id": 17,
    "category_name": "Category Name",
    "subcategory": "Subcategory"
  }
]
```

### 3. Search indices

```bash
# Search using both indices
python scripts/search_index.py --query "your query" --top-k 5

# Sparse-only
python scripts/search_index.py --query "your query" --sparse-only

# Dense-only
python scripts/search_index.py --query "your query" --dense-only
```

### 4. Run tests

```bash
python tests/test_pipeline.py
```

## Project structure

```
doc_indexing_project/
├── data/                   # Document data
├── indexes/                # Index outputs
│   ├── sparse_index.pkl         # Sparse index
│   ├── dense_index.pkl          # Dense index metadata
│   ├── dense_index.faiss        # FAISS index file
│   └── chunks.pkl               # Chunk data
├── src/                    # Source code
│   ├── document_loader.py       # Document loader
│   ├── chunker.py               # Document chunker
│   ├── sparse_index.py          # Sparse indexer
│   ├── dense_index.py           # Dense indexer
│   └── index_pipeline.py        # End-to-end pipeline
├── scripts/                # CLI scripts
│   ├── build_index.py           # Build indices
│   └── search_index.py          # Search
├── tests/                  # Tests
│   └── test_pipeline.py         # Pipeline test
├── requirements.txt         # Dependency list
└── README.md                # Documentation
```

## Python API

### Basic usage

```python
from src.index_pipeline import IndexPipeline

# Create the indexing pipeline
pipeline = IndexPipeline(
    chunk_size=500,
    chunk_overlap=50,
    sparse_method='bm25',
    dense_model='paraphrase-multilingual-MiniLM-L12-v2',
    language='zh'
)

# Process documents
pipeline.process_documents('data/')

# Build indices
pipeline.build_all_indices()

# Save indices
pipeline.save_indices('indexes/')

# Search
results = pipeline.search("your query", top_k=5)
print(results['sparse'])  # Sparse results
print(results['dense'])   # Dense results
```

### Load existing indices

```python
from src.sparse_index import SparseIndexer
from src.dense_index import DenseIndexer

# Load sparse index
sparse_indexer = SparseIndexer()
sparse_indexer.load('indexes/sparse_index.pkl')
results = sparse_indexer.search("query", top_k=5)

# Load dense index
dense_indexer = DenseIndexer()
dense_indexer.load('indexes/dense_index.pkl')
results = dense_indexer.search("query", top_k=5)
```

## Share indices

Index files can be shared easily:

1. **Package the index files**:
   ```bash
   cd indexes
   tar -czf indexes.tar.gz *.pkl *.faiss
   ```

2. **Share with others**:
   - Send `indexes.tar.gz` and `chunks.pkl`
   - After extracting, they can search with `search_index.py`

3. **Notes**:
   - Make sure they install compatible dependencies
   - Dense indices require the same embedding model to work properly
   - If the model differs, rebuild the index

## CLI parameters

### Build parameters

- `--input`: input path (file or directory)
- `--output`: output directory (default: `indexes`)
- `--chunk-size`: chunk size in characters (default: 500)
- `--chunk-overlap`: overlap size (default: 50)
- `--sparse-method`: sparse method, `bm25` or `tfidf` (default: `bm25`)
- `--dense-model`: Sentence Transformers model name (default: `paraphrase-multilingual-MiniLM-L12-v2`)
- `--language`: language setting, `zh` or `en` (default: `zh`)
- `--no-recursive`: do not recurse into subdirectories

### Search parameters

- `--query`: query text (required)
- `--index-dir`: index directory (default: `indexes`)
- `--top-k`: return top-k results (default: 5)
- `--sparse-only`: sparse-only search
- `--dense-only`: dense-only search

## Recommended models

### For Chinese documents
- `paraphrase-multilingual-MiniLM-L12-v2` (default; multilingual; fast)
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (more accurate, slower)
- `shibing624/text2vec-base-chinese` (optimized for Chinese)

### For English documents
- `all-MiniLM-L6-v2` (fast)
- `all-mpnet-base-v2` (more accurate)

## FAQ

### Q: How do I choose a chunk size?
A: It depends on your document type and use case:
- Short docs (e.g., FAQs): 100–200 characters
- Medium docs (e.g., articles): 300–500 characters
- Long docs (e.g., books): 500–1000 characters

### Q: What's the difference between sparse and dense indices?
A:
- **Sparse (BM25/TF-IDF)**: keyword-based matching; good for exact queries
- **Dense (vectors)**: semantic similarity; good for semantic search and similar-content retrieval

### Q: What if the index files are huge?
A:
- Use a smaller embedding model
- Reduce the number of chunks
- Use IVF index types (approximate search)

## License

MIT License

## Contributing

Issues and pull requests are welcome!
