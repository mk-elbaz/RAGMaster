"""
Chunking strategies for text processing.

This module demonstrates different chunking approaches:
1. **Semantic Chunking**: Split at natural boundaries (sentences, paragraphs)
2. **Hierarchical Chunking**: Split large sections first, then sub-chunk
3. **Overlap Strategy**: Maintain context across chunk boundaries

Why chunking matters:
- Too large: Retrieval becomes imprecise, LLM context gets diluted
- Too small: Lose context, break coherent thoughts
- No overlap: Important info at boundaries gets missed

Optimal sizes (rules of thumb):
- 500-1500 chars for sentence-transformers
- 150-300 chars overlap
- Adjust based on your domain (legal docs = larger, chat logs = smaller)
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    chunk_index: int
    source_id: str
    metadata: Dict[str, Any]
    
    def __repr__(self) -> str:
        return f"Chunk(idx={self.chunk_index}, source={self.source_id}, len={len(self.text)})"


class BaseChunker:
    """Base class for chunking strategies."""
    
    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Normalize whitespace while preserving paragraph breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Normalize line breaks (keep double newlines as paragraph markers)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough token estimation.
        
        Rule of thumb: 1 token ≈ 4 characters for English text.
        For accurate counts, use tiktoken library.
        """
        return len(text) // 4


class SemanticChunker(BaseChunker):
    """
    Semantic chunking with sentence-aware splitting.
    
    Strategy:
    1. Try to split at paragraph boundaries first
    2. If paragraph too large, split at sentence boundaries
    3. Apply overlap to maintain context
    
    This is better than simple character splitting because:
    - Preserves semantic units (complete thoughts)
    - Doesn't break mid-sentence
    - Maintains readability
    """
    
    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        """
        Args:
            chunk_size: Target size in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Sentence boundary regex (simple version)
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.sentence_pattern.split(text)
        # Re-add punctuation
        result = []
        for i, sent in enumerate(sentences[:-1]):
            result.append(sent + '.')
        if sentences[-1]:  # Last sentence might not have punctuation
            result.append(sentences[-1])
        return [s.strip() for s in result if s.strip()]
    
    def chunk(self, text: str, source_id: str, base_metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk text with sentence-aware splitting and overlap.
        
        Algorithm:
        1. Clean text
        2. Split into sentences
        3. Group sentences until reaching chunk_size
        4. Apply overlap by including last N characters of previous chunk
        """
        text = self.clean_whitespace(text)
        base_metadata = base_metadata or {}
        
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                metadata = {
                    **base_metadata,
                    "chunk_method": "semantic",
                    "sentence_count": len(current_chunk),
                    "char_count": len(chunk_text),
                    "token_estimate": self.estimate_tokens(chunk_text)
                }
                
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    source_id=source_id,
                    metadata=metadata
                ))
                
                chunk_index += 1
                
                # Apply overlap: keep sentences that fit in overlap window
                if self.overlap > 0:
                    overlap_text = chunk_text[-self.overlap:]
                    # Find which sentences fit in overlap
                    overlap_sentences = []
                    overlap_len = 0
                    for sent in reversed(current_chunk):
                        if overlap_len + len(sent) <= self.overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_len += len(sent)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_length = overlap_len
                else:
                    current_chunk = []
                    current_length = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            metadata = {
                **base_metadata,
                "chunk_method": "semantic",
                "sentence_count": len(current_chunk),
                "char_count": len(chunk_text),
                "token_estimate": self.estimate_tokens(chunk_text)
            }
            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                source_id=source_id,
                metadata=metadata
            ))
        
        return chunks


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunking: split large sections first, then sub-chunk.
    
    Strategy:
    1. Split document into major sections (by headings or double newlines)
    2. For each section, create chunks
    3. This preserves document structure better than flat chunking
    
    Use case:
    - Long documents with clear sections (technical docs, books, reports)
    - When you want to preserve heading context
    - When sections have different importance
    """
    
    def __init__(
        self,
        top_chunk_size: int = 4000,
        sub_chunk_size: int = 800,
        overlap: int = 200
    ):
        """
        Args:
            top_chunk_size: Maximum size for top-level sections
            sub_chunk_size: Target size for sub-chunks
            overlap: Overlap between sub-chunks
        """
        self.top_chunk_size = top_chunk_size
        self.sub_chunk_size = sub_chunk_size
        self.overlap = overlap
        
        # Pattern to detect section headers (lines with caps or ending with :)
        self.header_pattern = re.compile(
            r'^([A-Z][A-Z\s]{3,}|#{1,6}\s+.+|\d+\.\s+[A-Z].+)$',
            re.MULTILINE
        )
    
    def split_into_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into sections based on headers.
        
        Returns list of {header: str, content: str} dicts
        """
        sections = []
        
        # Find all headers
        matches = list(self.header_pattern.finditer(text))
        
        if not matches:
            # No headers found, treat as one section
            return [{"header": "", "content": text}]
        
        # Extract sections
        for i, match in enumerate(matches):
            header = match.group(0).strip()
            start = match.end()
            
            # Find end of section (next header or end of text)
            if i < len(matches) - 1:
                end = matches[i + 1].start()
            else:
                end = len(text)
            
            content = text[start:end].strip()
            
            if content:  # Only add non-empty sections
                sections.append({"header": header, "content": content})
        
        # Add any text before first header
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                sections.insert(0, {"header": "Preamble", "content": preamble})
        
        return sections
    
    def chunk(self, text: str, source_id: str, base_metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Hierarchical chunking: sections → sub-chunks.
        
        Algorithm:
        1. Split into sections by headers
        2. For small sections (<= top_chunk_size), sub-chunk them
        3. For large sections, sub-chunk with overlap
        4. Preserve header context in metadata
        """
        text = self.clean_whitespace(text)
        base_metadata = base_metadata or {}
        
        sections = self.split_into_sections(text)
        chunks = []
        chunk_index = 0
        
        semantic_chunker = SemanticChunker(
            chunk_size=self.sub_chunk_size,
            overlap=self.overlap
        )
        
        for section in sections:
            header = section["header"]
            content = section["content"]
            
            # Add header as context (if exists)
            if header:
                full_content = f"{header}\n\n{content}"
            else:
                full_content = content
            
            # Sub-chunk this section
            section_metadata = {
                **base_metadata,
                "section_header": header,
                "section_length": len(content)
            }
            
            sub_chunks = semantic_chunker.chunk(
                full_content,
                source_id,
                section_metadata
            )
            
            # Update chunk indices and add to result
            for chunk in sub_chunks:
                chunk.chunk_index = chunk_index
                chunk.metadata["chunk_method"] = "hierarchical"
                chunk.metadata["parent_section"] = header
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks


class SlidingWindowChunker(BaseChunker):
    """
    Simple sliding window chunker (character-based).
    
    This is the simplest approach - good for:
    - Quick prototyping
    - Uniform content without structure
    - Comparison baseline
    
    Not recommended for production (use SemanticChunker instead)
    because it can split mid-sentence or mid-word.
    """
    
    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, source_id: str, base_metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Sliding window over characters."""
        text = self.clean_whitespace(text)
        base_metadata = base_metadata or {}
        
        chunks = []
        i = 0
        chunk_index = 0
        n = len(text)
        
        while i < n:
            j = min(i + self.chunk_size, n)
            chunk_text = text[i:j]
            
            metadata = {
                **base_metadata,
                "chunk_method": "sliding_window",
                "start_pos": i,
                "end_pos": j,
                "char_count": len(chunk_text),
                "token_estimate": self.estimate_tokens(chunk_text)
            }
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                source_id=source_id,
                metadata=metadata
            ))
            
            if j == n:
                break
            
            i = j - self.overlap
            chunk_index += 1
        
        return chunks


class ChunkerFactory:
    """
    Factory to get the right chunker based on configuration.
    
    Usage:
        factory = ChunkerFactory()
        chunker = factory.get_chunker("hierarchical", chunk_size=1000)
        chunks = chunker.chunk(text, source_id)
    """
    
    @staticmethod
    def get_chunker(
        strategy: str = "semantic",
        chunk_size: int = 800,
        overlap: int = 200,
        **kwargs
    ) -> BaseChunker:
        """
        Get a chunker instance.
        
        Args:
            strategy: "semantic", "hierarchical", or "sliding_window"
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            **kwargs: Additional strategy-specific parameters
        """
        if strategy == "semantic":
            return SemanticChunker(chunk_size=chunk_size, overlap=overlap)
        
        elif strategy == "hierarchical":
            top_chunk_size = kwargs.get("top_chunk_size", 4000)
            return HierarchicalChunker(
                top_chunk_size=top_chunk_size,
                sub_chunk_size=chunk_size,
                overlap=overlap
            )
        
        elif strategy == "sliding_window":
            return SlidingWindowChunker(chunk_size=chunk_size, overlap=overlap)
        
        else:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Choose from: semantic, hierarchical, sliding_window"
            )


# Convenience function
def chunk_documents(
    documents: List[Any],
    strategy: str = "semantic",
    chunk_size: int = 800,
    overlap: int = 200
) -> List[Chunk]:
    """
    Chunk multiple documents using specified strategy.
    
    Args:
        documents: List of ParsedDocument objects
        strategy: Chunking strategy
        chunk_size: Target chunk size
        overlap: Overlap between chunks
    
    Returns:
        List of all chunks from all documents
    """
    chunker = ChunkerFactory.get_chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    all_chunks = []
    
    for doc in documents:
        chunks = chunker.chunk(
            text=doc.text,
            source_id=doc.source_id,
            base_metadata=doc.metadata
        )
        all_chunks.extend(chunks)
    
    return all_chunks
