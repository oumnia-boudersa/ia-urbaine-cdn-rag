"""
Document chunking module - splits documents into smaller chunks
"""
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
import tiktoken


class DocumentChunker:
    """Document chunker"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        method: str = 'recursive',
        split_by_organization: bool = False
    ):
        """
        Initialize chunker
        
        Args:
            chunk_size: Chunk size (in characters or tokens)
            chunk_overlap: Overlap size between chunks
            method: Chunking method ('recursive', 'token', or 'organization')
            split_by_organization: Whether to split by organization boundaries (for macommunaute format)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.method = method
        self.split_by_organization = split_by_organization
        
        if method == 'recursive':
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
            )
        elif method == 'token':
            self.splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif method == 'organization':
            # Split by organization, no size limit
            self.splitter = None
        else:
            raise ValueError(f"Unsupported chunking method: {method}")
    
    def chunk_document(self, document: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Chunk a single document
        
        Args:
            document: Document dictionary containing content field
            
        Returns:
            List of chunked documents
        """
        content = document.get('content', '')
        if not content.strip():
            return []
        
        # If chunking by organization
        if self.method == 'organization' or self.split_by_organization:
            return self._chunk_by_organization(document, content)
        
        # Standard chunking method
        chunks = self.splitter.split_text(content)
        
        chunked_docs = []
        for idx, chunk in enumerate(chunks):
            chunked_docs.append({
                'chunk_id': f"{document['file_name']}_chunk_{idx}",
                'file_name': document['file_name'],
                'file_path': document['file_path'],
                'file_type': document['file_type'],
                'chunk_index': idx,
                'content': chunk,
                'metadata': {
                    **document.get('metadata', {}),
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                }
            })
        
        return chunked_docs
    
    def _chunk_by_organization(self, document: Dict[str, any], content: str) -> List[Dict[str, any]]:
        """
        Chunk by organization boundaries (for macommunaute format)
        Format: ===== Organisation X: Name =====
        """
        import re
        
        # Match organization separator (supports French and English)
        org_pattern = r'=====\s*(?:Organisation|Organization)\s+\d+:\s*(.+?)\s*====='
        
        # Split document into organization blocks
        parts = re.split(org_pattern, content)
        
        # First element might be document header (no organization name)
        organizations = []
        org_names = []
        
        # Extract organization names and content
        matches = list(re.finditer(org_pattern, content))
        if not matches:
            # If no organization separators found, use standard chunking
            if self.method == 'recursive':
                temp_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunks = temp_splitter.split_text(content)
            else:
                chunks = [content]
            return self._create_chunks_from_list(document, chunks)
        
        # Extract content for each organization
        for i, match in enumerate(matches):
            org_name = match.group(1).strip()
            org_names.append(org_name)
            
            # Get content for this organization (between current match and next match)
            start_pos = match.end()
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)
            
            org_content = content[start_pos:end_pos].strip()
            
            # If organization content is too long, split further
            if len(org_content) > self.chunk_size * 2:
                # Create temporary splitter for splitting long organization content
                if self.method == 'recursive':
                    temp_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )
                    sub_chunks = temp_splitter.split_text(org_content)
                else:
                    # Simple splitting
                    sub_chunks = []
                    for j in range(0, len(org_content), self.chunk_size):
                        sub_chunks.append(org_content[j:j + self.chunk_size])
                
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    organizations.append({
                        'org_name': org_name,
                        'content': sub_chunk,
                        'is_subchunk': True,
                        'subchunk_index': sub_idx
                    })
            else:
                organizations.append({
                    'org_name': org_name,
                    'content': org_content,
                    'is_subchunk': False
                })
        
        # Create chunk documents
        chunked_docs = []
        chunk_idx = 0
        for org_data in organizations:
            chunk_id = f"{document['file_name']}_org_{chunk_idx}"
            if org_data['is_subchunk']:
                chunk_id += f"_sub_{org_data['subchunk_index']}"
            
            chunked_docs.append({
                'chunk_id': chunk_id,
                'file_name': document['file_name'],
                'file_path': document['file_path'],
                'file_type': document['file_type'],
                'chunk_index': chunk_idx,
                'content': org_data['content'],
                'metadata': {
                    **document.get('metadata', {}),
                    'organization_name': org_data['org_name'],
                    'chunk_size': len(org_data['content']),
                    'total_chunks': len(organizations),
                    'is_organization_chunk': True
                }
            })
            chunk_idx += 1
        
        return chunked_docs
    
    def _create_chunks_from_list(self, document: Dict[str, any], chunks: List[str]) -> List[Dict[str, any]]:
        """Create chunk documents from chunk list"""
        chunked_docs = []
        for idx, chunk in enumerate(chunks):
            chunked_docs.append({
                'chunk_id': f"{document['file_name']}_chunk_{idx}",
                'file_name': document['file_name'],
                'file_path': document['file_path'],
                'file_type': document['file_type'],
                'chunk_index': idx,
                'content': chunk,
                'metadata': {
                    **document.get('metadata', {}),
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                }
            })
        return chunked_docs
    
    def chunk_documents(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of documents
            
        Returns:
            List of all chunked documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, any]]) -> Dict[str, any]:
        """Get chunking statistics"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }
