# Adapted from reference code for dynamic_rags project
# Index corpus files using Pyserini (Windows PowerShell)

Write-Host "Creating Pyserini BM25 indexes..." -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green

# Define datasets
$datasets = @("211data", "macommunaute", "combined")

# Create indexes for each dataset
foreach ($dataset in $datasets) {
    Write-Host ""
    Write-Host "Indexing $dataset..." -ForegroundColor Yellow
    
    python -m pyserini.index.lucene `
        --collection JsonCollection `
        --input "data/pyserini_corpus/$dataset" `
        --index "data/pyserini_indexes/$dataset" `
        --generator DefaultLuceneDocumentGenerator `
        --storePositions --storeDocvectors --storeRaw `
        --threads 4
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Successfully indexed $dataset" -ForegroundColor Green
    } else {
        Write-Host "[FAILED] Failed to index $dataset" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "==================================" -ForegroundColor Green
Write-Host "[COMPLETE] Indexing complete!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host ""
Write-Host "Indexes created in: data/pyserini_indexes/" -ForegroundColor Cyan
Write-Host "Next step: Use search.ps1 to query the indexes" -ForegroundColor Cyan
