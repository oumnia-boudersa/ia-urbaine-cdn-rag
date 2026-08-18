"""
Adapted from reference code for dynamic_rags project.
Converts chunked JSON files to Pyserini JSONL format.
"""
import json
import os
import glob
from pathlib import Path

def load_chunks(file_path):
    """Load chunks from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_pyserini_corpus(chunks, output_path, dataset_name):
    """
    Write chunks to Pyserini JSONL format.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to output JSONL file
        dataset_name: Name prefix for document IDs
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    exist_docids = set()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks, 1):
            # Create unique document ID
            doc_id = f"{dataset_name}_{i}"
            doc_id = doc_id.replace(' ', '_')
            
            if doc_id in exist_docids:
                print(f'Duplicate document id found: {doc_id}')
                continue
            
            exist_docids.add(doc_id)
            
            # Pyserini document format
            doc = {
                'id': doc_id,
                'contents': chunk['text'],
                # Store metadata as additional fields
                'category_id': chunk.get('category_id'),
                'category_name': chunk.get('category_name', ''),
                'subcategory': chunk.get('subcategory', ''),
                'page_number': chunk.get('page_number')
            }
            
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f'✓ Wrote {len(exist_docids)} documents to {output_path}')
    return len(exist_docids)

def main():
    """Main processing function."""
    # Define paths
    CHUNKS_211_DATA = "../data/chuncked/211data/chunks_211data.json"
    CHUNKS_MACOMMUNAUTE_DIR = "../data/chuncked/macommunaute/"
    OUTPUT_DIR = "../data/pyserini_corpus/"
    
    print("="*60)
    print("Converting chunks to Pyserini format")
    print("="*60)
    
    # Process 211data
    print("\nProcessing 211data...")
    if os.path.exists(CHUNKS_211_DATA):
        chunks_211 = load_chunks(CHUNKS_211_DATA)
        output_211 = os.path.join(OUTPUT_DIR, "211data", "211data.jsonl")
        count_211 = write_pyserini_corpus(chunks_211, output_211, "211data")
        print(f"Total 211data documents: {count_211}\n")
    else:
        print(f"⚠ File not found: {CHUNKS_211_DATA}\n")
    
    # Process macommunaute files
    print("Processing macommunaute...")
    mac_files = glob.glob(os.path.join(CHUNKS_MACOMMUNAUTE_DIR, "*_chunks.json"))
    
    if mac_files:
        all_mac_chunks = []
        for file_path in mac_files:
            chunks = load_chunks(file_path)
            all_mac_chunks.extend(chunks)
            print(f"  Loaded {len(chunks)} chunks from {os.path.basename(file_path)}")
        
        output_mac = os.path.join(OUTPUT_DIR, "macommunaute", "macommunaute.jsonl")
        count_mac = write_pyserini_corpus(all_mac_chunks, output_mac, "macommunaute")
        print(f"Total macommunaute documents: {count_mac}\n")
    else:
        print(f"⚠ No macommunaute chunk files found in {CHUNKS_MACOMMUNAUTE_DIR}\n")
    
    # Create combined corpus
    print("Creating combined corpus...")
    all_chunks = []
    
    if os.path.exists(CHUNKS_211_DATA):
        all_chunks.extend(load_chunks(CHUNKS_211_DATA))
    
    if mac_files:
        for file_path in mac_files:
            all_chunks.extend(load_chunks(file_path))
    
    if all_chunks:
        output_combined = os.path.join(OUTPUT_DIR, "combined", "combined.jsonl")
        count_combined = write_pyserini_corpus(all_chunks, output_combined, "combined")
        print(f"Total combined documents: {count_combined}\n")
    
    print("="*60)
    print("✓ Corpus creation complete!")
    print("="*60)
    print(f"\nCorpus files created in: {OUTPUT_DIR}")
    print("\nNext step: Run index_corpus.sh or index_corpus.ps1 to create indexes")

if __name__ == "__main__":
    main()
