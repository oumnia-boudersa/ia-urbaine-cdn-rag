"""
Chunk loader module - loads pre-chunked JSON data
"""
import json
from typing import List, Dict
from pathlib import Path


class ChunkLoader:
    """Loader for pre-chunked JSON data"""
    
    def __init__(self):
        pass
    
    def load_json_file(self, file_path: str) -> List[Dict[str, any]]:
        """
        Load chunks from a JSON file
        
        Args:
            file_path: Path to JSON file containing chunks
            
        Returns:
            List of chunk dictionaries in standard format
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Convert to standard format
        standard_chunks = []
        for idx, chunk_data in enumerate(chunks_data):
            standard_chunk = {
                'chunk_id': f"{file_path.stem}_chunk_{chunk_data.get('chunk_id', idx)}",
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_type': '.json',
                'chunk_index': idx,
                'content': chunk_data.get('text', chunk_data.get('content', '')),
                'metadata': {
                    'chunk_id': chunk_data.get('chunk_id'),
                    'page_number': chunk_data.get('page_number'),
                    'category_id': chunk_data.get('category_id'),
                    'category_name': chunk_data.get('category_name'),
                    'subcategory': chunk_data.get('subcategory'),
                    'chunk_size': len(chunk_data.get('text', chunk_data.get('content', ''))),
                    'source_file': str(file_path)
                }
            }
            standard_chunks.append(standard_chunk)
        
        return standard_chunks
    
    def load_json_directory(self, directory: str, recursive: bool = True) -> List[Dict[str, any]]:
        """
        Load chunks from all JSON files in a directory
        
        Args:
            directory: Directory path containing JSON files
            recursive: Whether to recursively search subdirectories
            
        Returns:
            List of all chunks from all JSON files
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_chunks = []
        pattern = '**/*.json' if recursive else '*.json'
        
        for json_file in directory.glob(pattern):
            if json_file.is_file():
                try:
                    chunks = self.load_json_file(str(json_file))
                    all_chunks.extend(chunks)
                    print(f"Loaded {len(chunks)} chunks from {json_file.name}")
                except Exception as e:
                    print(f"Failed to load {json_file}: {e}")
        
        return all_chunks
    
    def load_single_json(self, file_path: str) -> List[Dict[str, any]]:
        """
        Load chunks from a single JSON file (alias for load_json_file)
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of chunk dictionaries
        """
        return self.load_json_file(file_path)
