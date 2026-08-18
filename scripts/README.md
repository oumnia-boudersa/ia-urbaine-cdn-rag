# Pyserini Indexing and Search Scripts

**Adapted from reference code for dynamic_rags project**

This directory contains scripts to create and search Pyserini BM25 indexes for the 211data and macommunaute datasets.

## Quick Start - Run These Commands Sequentially

From the project root directory, run these commands in order:

### Step 1: Navigate to scripts directory
```powershell
cd D:\mscudemproject\main\dynamic_rags\scripts
```

### Step 2: Convert chunks to Pyserini JSONL format
```powershell
python write_pyserini_corpus.py
```
**Output:** Creates `data/pyserini_corpus/211data/`, `macommunaute/`, and `combined/` directories with `.jsonl` files

### Step 3: Create Pyserini BM25 indexes
```powershell
.\index_corpus.ps1
```
**Output:** Creates `data/pyserini_indexes/` with BM25 indexes for all datasets  
**Time:** ~5-10 minutes depending on data size

### Step 4: Test search (optional)
```powershell
.\search.ps1
```
**Output:** Searches all indexes with example query and displays results

### Verify Your Work

After each step, you can verify:
```powershell
# Check corpus files created
ls ..\data\pyserini_corpus\combined\

# Check indexes created
ls ..\data\pyserini_indexes\combined\

# Count documents in corpus
(Get-Content ..\data\pyserini_corpus\combined\combined.jsonl).Count
```

### Alternative: Index Single Dataset

If you only want to index one dataset:
```powershell
python -m pyserini.index.lucene --collection JsonCollection --input data/pyserini_corpus/combined --index data/pyserini_indexes/combined --generator DefaultLuceneDocumentGenerator --storePositions --storeDocvectors --storeRaw --threads 4
```

---

## Files

- `write_pyserini_corpus.py` - Convert chunked JSON to Pyserini JSONL format
- `index_corpus.sh` / `index_corpus.ps1` - Create Pyserini indexes (Bash/PowerShell)
- `search.sh` / `search.ps1` - Search the indexes (Bash/PowerShell)

## Workflow

### 1. Convert Chunks to Pyserini Format

```bash
# From the project root directory
cd scripts
python write_pyserini_corpus.py
```

**Input:**
- `../data/chuncked/211data/chunks_211data.json`
- `../data/chuncked/macommunaute/*_chunks.json`

**Output:**
- `../data/pyserini_corpus/211data/211data.jsonl`
- `../data/pyserini_corpus/macommunaute/macommunaute.jsonl`
- `../data/pyserini_corpus/combined/combined.jsonl`

Each JSONL document has this format:
```json
{
  "id": "211data_1",
  "contents": "text content here...",
  "category_id": 1,
  "category_name": "Child, Youth and Family",
  "subcategory": "",
  "page_number": 1
}
```

### 2. Create Pyserini Indexes

**Linux/Mac:**
```bash
bash index_corpus.sh
```

**Windows:**
```powershell
.\index_corpus.ps1
```

This creates BM25 indexes in `../data/pyserini_indexes/` for:
- `211data`
- `macommunaute`
- `combined` (all data together)

**Manual indexing command:**
```bash
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/pyserini_corpus/combined \
  --index data/pyserini_indexes/combined \
  --generator DefaultLuceneDocumentGenerator \
  --storePositions --storeDocvectors --storeRaw \
  --threads 4
```

### 3. Search the Indexes

**Linux/Mac:**
```bash
bash search.sh
```

**Windows:**
```powershell
.\search.ps1
```

**Manual search command:**
```bash
python -m pyserini.search.lucene \
  --index data/pyserini_indexes/combined \
  --query "Where can I find housing support?" \
  --bm25 \
  --hits 10
```

## Parameters

### Indexing Parameters
- `--threads 4` - Adjust based on your CPU cores
- `--storePositions` - Store term positions (needed for phrase queries)
- `--storeDocvectors` - Store document vectors (needed for analysis)
- `--storeRaw` - Store raw document content (needed for retrieval)

### Search Parameters
- `--bm25` - Use BM25 scoring algorithm
- `--hits 10` - Number of results to return
- `--query` - Single query string
- `--topics` - Query file in TSV format (qid\tquery)
- `--output` - Save results to TREC format file

## What Was Adapted from Reference Code

### Original Reference Code
The reference code was designed for the BRIGHT dataset from Hugging Face with:
- Multiple predefined datasets (biology, earth_science, etc.)
- Data loaded from `datasets.load_dataset()`
- Loop over specific dataset names

### Adaptations for Dynamic RAGs
1. **Data Loading**: Changed from Hugging Face datasets to local JSON files
2. **Dataset Names**: Updated to `211data`, `macommunaute`, `combined`
3. **Document Structure**: Added metadata fields (category_id, category_name, subcategory, page_number)
4. **File Paths**: Updated to match the dynamic_rags project structure
5. **ID Generation**: Changed from dataset-specific IDs to sequential chunk IDs
6. **Multi-file Support**: Added logic to process multiple macommunaute chunk files
7. **Windows Support**: Added PowerShell scripts alongside bash scripts

### Core Pyserini Commands (Unchanged)
The actual Pyserini indexing and search commands remain the same - these are the standard Pyserini API calls that work for any document collection.

## Integration with RAG Pipeline

After indexing, you can use Pyserini search in your RAG pipeline:

```python
from pyserini.search.lucene import LuceneSearcher

# Initialize searcher
searcher = LuceneSearcher('data/pyserini_indexes/combined')

# Search
query = "Where can I find housing support?"
hits = searcher.search(query, k=10)

# Process results
for hit in hits:
    print(f"Score: {hit.score:.4f}")
    print(f"Document: {hit.docid}")
    print(f"Content: {hit.raw[:200]}...")
    print()
```

## Troubleshooting

### Index not found
- Make sure you ran `write_pyserini_corpus.py` first
- Check that corpus files exist in `data/pyserini_corpus/`
- Verify index directory was created in `data/pyserini_indexes/`

### Java errors
- Pyserini requires Java 11+
- Install with: `sudo apt install openjdk-11-jdk` (Linux) or download from Oracle (Windows/Mac)

### Memory errors
- Reduce `--threads` parameter
- Use smaller datasets (e.g., only `211data` instead of `combined`)

## Next Steps

1. Create query files for evaluation (see reference code `write_pyserini_queries.py`)
2. Create relevance judgments (qrels) for evaluation (see reference code `write_pyserini_qrels.py`)
3. Evaluate retrieval performance using standard IR metrics
4. Integrate BM25 retrieval with your RAG system
