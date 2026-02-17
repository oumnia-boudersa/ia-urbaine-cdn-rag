# Macommunaute Data Guide

## Overview

The Macommunaute dataset contains community service organization information in two languages: French (`fr`) and English (`en`).

### Directory structure

```
data/raw/macommunaute/
├── fr/          # French version
│   ├── sante.txt
│   ├── sante.pdf
│   ├── emploi.txt
│   └── ...
└── en/          # English version
    ├── sante_en.txt
    ├── sante_en.pdf
    └── ...
```

### File format

Each file contains multiple organizations, formatted like this:

```
===== Organisation 1: Centre Philou =====
URL: https://...
Site internet: http://...
Description: À PROPOS DE L'ORGANISME
Coordonnées
...
Mission
...
Description
...
```

## Quick start

### 1. Build indices (recommended: dedicated script)

```bash
# Default settings (chunk by organization, auto-detect language)
python scripts/build_macommunaute_index.py

# Custom settings
python scripts/build_macommunaute_index.py \
    --input data/raw/macommunaute \
    --output indexes \
    --chunk-size 800 \
    --language auto \
    --use-org-chunking
```

### 2. Build with the standard script

```bash
python scripts/build_index.py \
    --input data/raw/macommunaute \
    --output indexes \
    --chunk-method organization \
    --split-by-org \
    --language auto
```

### 3. Search

```bash
# French query
python scripts/search_index.py --query "santé mentale" --top-k 5

# English query
python scripts/search_index.py --query "mental health" --top-k 5

# Mixed query
python scripts/search_index.py --query "aide aux personnes âgées" --top-k 5
```

## Recommended configuration

### For Macommunaute

1. **Chunking strategy**: Use `organization` chunking (split by organization boundaries)
   - Pros: preserves complete organization records
   - Each organization becomes one chunk

2. **Chunk size**: 800–1000 characters
   - Most organization records fit in a single chunk
   - Very long records will be split automatically

3. **Language**: `auto` (auto-detect)
   - Language is detected from the file path (`/fr/` or `/en/`)
   - The sparse indexer selects an appropriate tokenizer based on the detected language

4. **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
   - Multilingual (French/English)
   - Performs well for semantic search

## French tokenization

### Option 1: Use spaCy (recommended)

```bash
# Install spaCy
pip install spacy

# Download the French model
python -m spacy download fr_core_news_sm
```

### Option 2: Simple tokenization (default)

If you don't install spaCy, the system falls back to splitting on whitespace and punctuation, which is sufficient for many cases.

## Search examples

### Example 1: Find health-related services

```bash
python scripts/search_index.py --query "santé mentale" --top-k 5
```

### Example 2: Find employment services

```bash
python scripts/search_index.py --query "emploi" --top-k 5
```

### Example 3: Find a specific organization

```bash
python scripts/search_index.py --query "Centre Philou" --top-k 3
```

## Python API

```python
from src.index_pipeline import IndexPipeline

# Create a pipeline (tuned for Macommunaute)
pipeline = IndexPipeline(
    chunk_size=800,
    chunk_overlap=50,
    sparse_method='bm25',
    dense_model='paraphrase-multilingual-MiniLM-L12-v2',
    language='auto',  # Auto-detect language
    chunk_method='organization',  # Chunk by organization
    split_by_organization=True
)

# Process documents
pipeline.process_documents('data/raw/macommunaute', recursive=True)

# Build indices
pipeline.build_all_indices()

# Save indices
pipeline.save_indices('indexes/')

# Search
results = pipeline.search("santé mentale", top_k=5)

# Print results
for chunk, score in results['sparse']:
    print(f"Organization: {chunk['metadata'].get('organization_name', 'N/A')}")
    print(f"File: {chunk['file_name']}")
    print(f"Content: {chunk['content'][:200]}...")
    print()
```

## Notes

1. **Language detection**: Language is detected from the file path (containing `/fr/` or `/en/`).

2. **Organization chunking**: When using organization chunking, each chunk's metadata includes an `organization_name` field.

3. **Bilingual search**: With a multilingual model, you can often query in one language and retrieve results from the other language.

4. **File formats**: `.txt` and `.pdf` are supported. `.txt` is recommended (faster processing).

5. **Index size**: Because the dataset is bilingual, index files may be large; leave enough disk space.
