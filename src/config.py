"""
Configuration management using Pydantic Settings.

This module demonstrates best practices for configuration:
- Type-safe settings with validation
- Environment variable loading
- Sensible defaults
- Easy testing with overrides
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    
    # Local LLM (alternative to OpenAI)
    use_local_llm: bool = Field(default=False, description="Use local LLM instead of OpenAI")
    local_llm_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local LLM (e.g., Ollama)"
    )
    local_llm_model: str = Field(default="llama2", description="Local LLM model name")
    
    # Embedding model for retrieval
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    # RAG Configuration
    chunk_size: int = Field(default=800, ge=100, le=4000, description="Characters per chunk")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")
    top_k_retrieval: int = Field(default=10, ge=1, le=50, description="Number of chunks to retrieve")
    hybrid_alpha: float = Field(
        default=0.5, # Reverted to original default to maintain syntactic correctness for float type
        le=1.0,
        description="BM25 vs embeddings weight (0=pure embeddings, 1=pure BM25)"
    )
    
    # Hierarchical chunking
    use_hierarchical_chunking: bool = Field(
        default=True,
        description="Use hierarchical chunking strategy"
    )
    top_chunk_size: int = Field(
        default=4000,
        description="Size of top-level chunks in hierarchical splitting"
    )
    
    # Paths
    index_dir: str = Field(default="./indexes", description="Directory for storing indexes")
    data_dir: str = Field(default="./data", description="Directory containing source documents")
    log_dir: str = Field(default="./logs", description="Directory for log files")
    
    # Observability
    prometheus_port: int = Field(default=8001, ge=1024, le=65535, description="Prometheus metrics port")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Performance
    batch_size: int = Field(default=64, ge=1, le=256, description="Batch size for embedding")
    max_workers: int = Field(default=4, ge=1, le=32, description="Max workers for parallel processing")
    
    # Answer generation
    max_tokens: int = Field(default=1000, ge=100, le=4000, description="Max tokens for LLM response")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    
    def validate_api_key(self) -> bool:
        """Check if API key is configured when using OpenAI."""
        if not self.use_local_llm and not self.openai_api_key:
            return False
        return True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
