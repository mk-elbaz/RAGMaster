"""
Hybrid RAG Index: Combines BM25 (keyword) + FAISS (semantic) retrieval.

This module demonstrates:
1. **BM25 Indexing**: Keyword-based retrieval (catches exact terms)
2. **FAISS Indexing**: Dense vector retrieval (understands meaning)
3. **Score Fusion**: Intelligently combine both approaches
4. **Persistence**: Save/load indexes for production use

Why Hybrid Search?
- BM25: Great for technical terms, names, specific phrases
- Embeddings: Great for semantic similarity, paraphrasing, concepts
- Hybrid: Best of both worlds - highest recall and precision

How Score Fusion Works:
1. Get top-K from BM25 and FAISS separately
2. Normalize scores to [0, 1] range
3. Combine: final_score = alpha * bm25_norm + (1-alpha) * faiss_norm
4. Re-rank by combined score
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from chunking import Chunk
from config import get_settings


class RetrievalResult:
    """Container for retrieval results with detailed metadata."""
    
    def __init__(
        self,
        chunk: Chunk,
        score: float,
        bm25_score: float = 0.0,
        faiss_score: float = 0.0,
        rank: int = 0
    ):
        self.chunk = chunk
        self.score = score  # Final fused score
        self.bm25_score = bm25_score
        self.faiss_score = faiss_score
        self.rank = rank
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.chunk.text,
            "source_id": self.chunk.source_id,
            "chunk_index": self.chunk.chunk_index,
            "metadata": self.chunk.metadata,
            "score": float(self.score),
            "bm25_score": float(self.bm25_score),
            "faiss_score": float(self.faiss_score),
            "rank": self.rank
        }
    
    def __repr__(self) -> str:
        return (
            f"RetrievalResult(rank={self.rank}, score={self.score:.3f}, "
            f"bm25={self.bm25_score:.3f}, faiss={self.faiss_score:.3f}, "
            f"source={self.chunk.source_id})"
        )


class HybridRAGIndex:
    """
    Hybrid retrieval system combining BM25 and dense embeddings.
    
    Architecture:
    - BM25: rank_bm25 library (Okapi BM25 variant)
    - Embeddings: sentence-transformers → FAISS
    - Fusion: Weighted combination of normalized scores
    """
    
    def __init__(self, embedding_model_name: Optional[str] = None):
        """
        Initialize the hybrid index.
        
        Args:
            embedding_model_name: Sentence transformer model name
                Popular choices:
                - "all-MiniLM-L6-v2": Fast, good balance (384 dim)
                - "all-mpnet-base-v2": Better quality (768 dim)
                - "multi-qa-mpnet-base-dot-v1": Optimized for Q&A
        """
        settings = get_settings()
        self.embedding_model_name = embedding_model_name or settings.embedding_model_name
        
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        
        # Storage
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        
        # Stats
        self.stats = {
            "total_chunks": 0,
            "total_documents": 0,
            "embedding_dim": self.embedder.get_sentence_embedding_dimension(),
            "model_name": self.embedding_model_name
        }
    
    def ingest_chunks(self, chunks: List[Chunk]):
        """
        Ingest pre-chunked documents.
        
        Args:
            chunks: List of Chunk objects
        """
        print(f"Ingesting {len(chunks)} chunks...")
        self.chunks.extend(chunks)
        
        # Update stats
        sources = set(chunk.source_id for chunk in chunks)
        self.stats["total_chunks"] = len(self.chunks)
        self.stats["total_documents"] = len(sources)
        
        print(f"Total chunks in index: {len(self.chunks)}")
    
    def build_embeddings(self, batch_size: Optional[int] = None):
        """
        Build FAISS index with embeddings.
        
        Process:
        1. Encode all chunk texts to vectors
        2. Normalize vectors for cosine similarity
        3. Build FAISS index (using Inner Product on normalized vectors)
        
        Args:
            batch_size: Encoding batch size (larger = faster but more memory)
        """
        settings = get_settings()
        batch_size = batch_size or settings.batch_size
        
        if not self.chunks:
            raise ValueError("No chunks to embed. Call ingest_chunks() first.")
        
        print(f"\nBuilding embeddings for {len(self.chunks)} chunks...")
        
        # Extract texts
        texts = [chunk.text for chunk in self.chunks]
        
        # Encode with progress bar
        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        self.embeddings = embeddings.astype('float32')
        
        # Build FAISS index
        # IndexFlatIP = brute force inner product (works as cosine with normalized vectors)
        # For larger datasets (>100k), consider IndexIVFFlat or IndexHNSWFlat
        d = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)  # Inner Product
        self.faiss_index.add(self.embeddings)
        
        self.stats["faiss_total_vectors"] = self.faiss_index.ntotal
        
        print(f"✓ FAISS index built: {self.faiss_index.ntotal} vectors, {d} dimensions")
    
    def build_bm25(self):
        """
        Build BM25 index.
        
        Process:
        1. Tokenize each chunk (simple whitespace splitting)
        2. Build BM25Okapi index
        
        Note: For better results, use proper tokenizer (spaCy, NLTK)
        or language-specific tokenizer.
        """
        if not self.chunks:
            raise ValueError("No chunks to index. Call ingest_chunks() first.")
        
        print(f"\nBuilding BM25 index...")
        
        # Simple whitespace tokenizer
        # TODO: Upgrade to spaCy or NLTK for production
        self.tokenized_corpus = [
            chunk.text.lower().split()
            for chunk in self.chunks
        ]
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"✓ BM25 index built: {len(self.tokenized_corpus)} documents")
    
    def build_all_indexes(self, batch_size: Optional[int] = None):
        """Convenience method to build both indexes."""
        self.build_embeddings(batch_size=batch_size)
        self.build_bm25()
        print("\n✓ All indexes built successfully!")
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search using BM25.
        
        Returns:
            List of (chunk_index, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_bm25() first.")
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def search_faiss(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search using FAISS (dense embeddings).
        
        Returns:
            List of (chunk_index, score) tuples
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_embeddings() first.")
        
        # Encode query
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search FAISS
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Filter out -1 indices (happens when index has fewer than top_k items)
        results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]
        
        return results
    
    @staticmethod
    def normalize_scores(score_pairs: List[Tuple[int, float]]) -> Dict[int, float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        This is crucial for fair score fusion between BM25 and FAISS,
        which have different score scales.
        
        Args:
            score_pairs: List of (index, score) tuples
        
        Returns:
            Dict mapping index to normalized score
        """
        if not score_pairs:
            return {}
        
        indices, scores = zip(*score_pairs)
        scores = np.array(scores, dtype=float)
        
        # Min-max normalization
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score < 1e-6:
            # All scores are the same
            return {idx: 1.0 for idx in indices}
        
        normalized = (scores - min_score) / (max_score - min_score)
        
        return {idx: float(norm) for idx, norm in zip(indices, normalized)}
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        bm25_k: Optional[int] = None,
        alpha: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval: combine BM25 and FAISS results.
        
        Algorithm:
        1. Retrieve top-K from BM25
        2. Retrieve top-K from FAISS
        3. Normalize scores separately
        4. Fuse: final_score = alpha * bm25_norm + (1-alpha) * faiss_norm
        5. Sort by fused score and return top-K
        
        Args:
            query_text: Search query
            top_k: Number of results to return
            bm25_k: Number of candidates from BM25 (default: top_k * 2)
            alpha: BM25 weight (0=pure embeddings, 1=pure BM25)
                   Default from settings
        
        Returns:
            List of RetrievalResult objects, sorted by score
        """
        settings = get_settings()
        alpha = alpha if alpha is not None else settings.hybrid_alpha
        bm25_k = bm25_k or (top_k * 2)
        
        # Retrieve from both indexes
        bm25_results = self.search_bm25(query_text, top_k=bm25_k)
        faiss_results = self.search_faiss(query_text, top_k=top_k)
        
        # Normalize scores
        bm25_normalized = self.normalize_scores(bm25_results)
        faiss_normalized = self.normalize_scores(faiss_results)
        
        # Fuse scores
        all_indices = set(bm25_normalized.keys()) | set(faiss_normalized.keys())
        fused_scores = {}
        
        for idx in all_indices:
            bm25_score = bm25_normalized.get(idx, 0.0)
            faiss_score = faiss_normalized.get(idx, 0.0)
            
            # Weighted combination
            fused = alpha * bm25_score + (1 - alpha) * faiss_score
            fused_scores[idx] = (fused, bm25_score, faiss_score)
        
        # Sort by fused score
        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        for rank, (idx, (fused_score, bm25_score, faiss_score)) in enumerate(ranked, start=1):
            result = RetrievalResult(
                chunk=self.chunks[idx],
                score=fused_score,
                bm25_score=bm25_score,
                faiss_score=faiss_score,
                rank=rank
            )
            results.append(result)
        
        return results
    
    def save(self, index_dir: str):
        """
        Persist index to disk.
        
        Saves:
        - Chunks (as JSON lines)
        - Embeddings (numpy array)
        - FAISS index (binary format)
        - BM25 tokenized corpus (JSON)
        - Metadata/stats (JSON)
        """
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving index to {index_dir}...")
        
        # Save chunks
        chunks_file = index_path / "chunks.jsonl"
        with open(chunks_file, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                chunk_data = {
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "source_id": chunk.source_id,
                    "metadata": chunk.metadata
                }
                f.write(json.dumps(chunk_data) + "\n")
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(index_path / "embeddings.npy", self.embeddings)
        
        # Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(index_path / "faiss.index"))
        
        # Save BM25 corpus
        if self.tokenized_corpus:
            with open(index_path / "bm25_corpus.json", "w", encoding="utf-8") as f:
                json.dump(self.tokenized_corpus, f)
        
        # Save metadata
        with open(index_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✓ Index saved successfully!")
    
    def load(self, index_dir: str):
        """
        Load index from disk.
        
        Args:
            index_dir: Directory containing saved index files
        """
        index_path = Path(index_dir)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")
        
        print(f"\nLoading index from {index_dir}...")
        
        # Load chunks
        chunks_file = index_path / "chunks.jsonl"
        self.chunks = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                chunk = Chunk(
                    text=data["text"],
                    chunk_index=data["chunk_index"],
                    source_id=data["source_id"],
                    metadata=data["metadata"]
                )
                self.chunks.append(chunk)
        
        # Load embeddings
        embeddings_file = index_path / "embeddings.npy"
        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)
        
        # Load FAISS index
        faiss_file = index_path / "faiss.index"
        if faiss_file.exists():
            self.faiss_index = faiss.read_index(str(faiss_file))
        
        # Load BM25 corpus
        bm25_file = index_path / "bm25_corpus.json"
        if bm25_file.exists():
            with open(bm25_file, "r", encoding="utf-8") as f:
                self.tokenized_corpus = json.load(f)
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Load metadata
        metadata_file = index_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                self.stats = json.load(f)
        
        print(f"✓ Index loaded: {len(self.chunks)} chunks")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            **self.stats,
            "has_embeddings": self.embeddings is not None,
            "has_faiss": self.faiss_index is not None,
            "has_bm25": self.bm25 is not None
        }
