# Using Pre-chunked JSON Data

## Overview

If your documents are already pre-processed and chunked into JSON format, you can use the `build_from_chunks.py` script to build indices directly from the JSON files without re-chunking.

## JSON Format

Your JSON file should be an array of chunk objects. Each chunk should have:

```json
[
  {
    "chunk_id": 1,
    "page_number": 1,
    "text": "The actual chunk content text here...",
    "category_id": 17,
    "category_name": "Category Name",
    "subcategory": "Subcategory Name"
  },
  {
    "chunk_id": 2,
    "page_number": 2,
    "text": "Another chunk...",
    "category_id": 17,
    "category_name": "Category Name",
    "subcategory": "Subcategory Name"
  }
]
```

### Required Fields

- `text` or `content`: The chunk text content (required)
- `chunk_id`: Unique identifier for the chunk (optional, will be auto-generated if missing)

### Optional Fields

- `page_number`: Page number where chunk appears
- `category_id`: Category ID
- `category_name`: Category name
- `subcategory`: Subcategory name
- Any other metadata fields you want to preserve

## Usage

### Build from Single JSON File

```bash
python scripts/build_from_chunks.py \
    --input data/chuncked/211data/chunks_211data.json \
    --output indexes
```

### Build from Directory of JSON Files

```bash
python scripts/build_from_chunks.py \
    --input data/chuncked/ \
    --output indexes \
    --language auto
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | **required** | Input path (JSON file or directory) |
| `--output` | str | `indexes` | Output directory |
| `--sparse-method` | str | `bm25` | Sparse method: `bm25` or `tfidf` |
| `--dense-model` | str | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence transformer model |
| `--language` | str | `auto` | Language: `zh`, `en`, `fr`, or `auto` |
| `--no-recursive` | flag | `False` | Don't recursively process subdirectories |

## Example

```bash
# Build index from your pre-chunked data
python scripts/build_from_chunks.py \
    --input data/chuncked/211data/chunks_211data.json \
    --output indexes \
    --sparse-method bm25 \
    --language auto

# Search the built index
python scripts/search_index.py \
    --query "your search query" \
    --index-dir indexes \
    --top-k 10
```

## Advantages

1. **Faster**: Skip document loading and chunking steps
2. **Preserve Metadata**: Keep your original chunk metadata (category, page number, etc.)
3. **Flexible**: Use your own chunking strategy
4. **Efficient**: Directly build indices from prepared data

## Python API

You can also use the Python API:

```python
from src.index_pipeline import IndexPipeline

# Create pipeline
pipeline = IndexPipeline(
    sparse_method='bm25',
    dense_model='paraphrase-multilingual-MiniLM-L12-v2',
    language='auto'
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
```

## Notes

- The script automatically detects if input is a JSON file or directory
- All metadata fields from your JSON will be preserved in the chunk metadata
- The `text` field is used as chunk content (falls back to `content` if `text` doesn't exist)
- Chunk IDs are auto-generated if not provided in the JSON
