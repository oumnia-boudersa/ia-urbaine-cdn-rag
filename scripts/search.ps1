# Adapted from reference code for dynamic_rags project
# Search indexes using Pyserini BM25 (Windows PowerShell)

Write-Host "Searching Pyserini indexes..." -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green

# Define datasets to search
$datasets = @("211data", "macommunaute", "combined")

# Example query (replace with your actual query)
$query = "Where can I find housing support?"

# Create output directory
New-Item -ItemType Directory -Force -Path "runs_bm25" | Out-Null

foreach ($dataset in $datasets) {
    Write-Host ""
    Write-Host "Searching $dataset..." -ForegroundColor Yellow
    
    # Check if index exists
    if (-not (Test-Path "data/pyserini_indexes/$dataset")) {
        Write-Host "⚠ Index not found for $dataset. Run index_corpus.ps1 first." -ForegroundColor Red
        continue
    }
    
    # Run search
    python -m pyserini.search.lucene `
        --index "data/pyserini_indexes/$dataset" `
        --query $query `
        --bm25 `
        --hits 10
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Search completed for $dataset" -ForegroundColor Green
    } else {
        Write-Host "✗ Search failed for $dataset" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "==============================" -ForegroundColor Green
Write-Host "✓ Search complete!" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green
Write-Host ""
Write-Host "To search with query files (TSV format):" -ForegroundColor Cyan
Write-Host "  python -m pyserini.search.lucene \"
Write-Host "    --index data/pyserini_indexes/combined \"
Write-Host "    --topics queries.tsv \"
Write-Host "    --output runs_bm25/bm25.combined.trec \"
Write-Host "    --bm25 \"
Write-Host "    --hits 100"
