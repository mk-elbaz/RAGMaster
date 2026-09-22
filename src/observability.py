"""
Observability: Logging and Metrics for RAG System.

This module implements:
1. **Structured Logging**: JSON logs with context
2. **Prometheus Metrics**: Query counts, latency, token usage
3. **Cost Tracking**: LLM token costs
4. **Performance Monitoring**: Retrieval and generation times

Best Practices:
- Log all queries with unique IDs for tracing
- Track both technical metrics (latency) and business metrics (cost)
- Use structured logging for easy parsing and analysis
"""

import os
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from functools import wraps

from prometheus_client import Counter, Histogram, Gauge, start_http_server
from pythonjsonlogger import jsonlogger

from config import get_settings


# ============================================================================
# Prometheus Metrics
# ============================================================================

# Counters
QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total number of RAG queries",
    ["status"]  # success or error
)

RETRIEVAL_TOTAL = Counter(
    "rag_retrieval_total",
    "Total number of retrieval operations",
    ["index_type"]  # bm25 or faiss
)

# Histograms (for timing)
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "RAG query end-to-end latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval latency",
    ["index_type"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "Answer generation latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
)

# Gauges (current values)
TOKENS_INPUT = Gauge(
    "rag_tokens_input",
    "Number of input tokens in last query"
)

TOKENS_OUTPUT = Gauge(
    "rag_tokens_output",
    "Number of output tokens in last query"
)

COST_DOLLARS = Gauge(
    "rag_cost_dollars",
    "Estimated cost of last query in dollars"
)


# ============================================================================
# Cost Calculator
# ============================================================================

class CostCalculator:
    """
    Calculate LLM costs based on token usage.
    
    Pricing (as of Nov 2025, update regularly):
    - GPT-4-turbo: $0.01/1K input, $0.03/1K output
    - GPT-3.5-turbo: $0.0005/1K input, $0.0015/1K output
    """
    
    PRICING = {
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    }
    
    @classmethod
    def calculate_cost(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost in dollars.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Cost in dollars
        """
        pricing = cls.PRICING.get(model)
        if not pricing:
            # Default to GPT-4 pricing (conservative estimate)
            pricing = cls.PRICING["gpt-4o"]
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost


# ============================================================================
# Structured Logger
# ============================================================================

class RAGLogger:
    """
    Structured logger for RAG operations.
    
    Logs are written in JSON format for easy parsing by log aggregators
    (ELK, Splunk, CloudWatch, etc.)
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files
        """
        settings = get_settings()
        self.log_dir = Path(log_dir or settings.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("rag_system")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # JSON file handler
        log_file = self.log_dir / f"rag_{datetime.now().strftime('%Y%m%d')}.jsonl"
        file_handler = logging.FileHandler(log_file)
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (optional, for debugging)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_query(
        self,
        query_id: str,
        query: str,
        num_results: int,
        retrieval_time: float,
        generation_time: float,
        total_time: float,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        cost: Optional[float] = None,
        status: str = "success",
        error: Optional[str] = None,
        **kwargs
    ):
        """Log a query event."""
        log_data = {
            "event_type": "query",
            "query_id": query_id,
            "query": query,
            "num_results": num_results,
            "retrieval_time_seconds": retrieval_time,
            "generation_time_seconds": generation_time,
            "total_time_seconds": total_time,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "cost_dollars": cost,
            "status": status,
            "error": error,
            **kwargs
        }
        
        if status == "success":
            self.logger.info("Query completed", extra=log_data)
        else:
            self.logger.error("Query failed", extra=log_data)
    
    def log_retrieval(
        self,
        query_id: str,
        index_type: str,
        num_results: int,
        latency: float,
        top_score: Optional[float] = None
    ):
        """Log a retrieval event."""
        log_data = {
            "event_type": "retrieval",
            "query_id": query_id,
            "index_type": index_type,
            "num_results": num_results,
            "latency_seconds": latency,
            "top_score": top_score
        }
        
        self.logger.info("Retrieval completed", extra=log_data)
    
    def log_index_build(
        self,
        num_documents: int,
        num_chunks: int,
        build_time: float,
        index_size_mb: Optional[float] = None
    ):
        """Log index building event."""
        log_data = {
            "event_type": "index_build",
            "num_documents": num_documents,
            "num_chunks": num_chunks,
            "build_time_seconds": build_time,
            "index_size_mb": index_size_mb
        }
        
        self.logger.info("Index built", extra=log_data)


# ============================================================================
# Monitoring Decorators
# ============================================================================

def monitor_query(logger: RAGLogger):
    """
    Decorator to monitor query execution.
    
    Usage:
        @monitor_query(logger)
        def my_query_function(query: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import uuid
            
            query_id = str(uuid.uuid4())
            start_time = time.time()
            
            try:
                # Execute function
                with QUERY_LATENCY.time():
                    result = func(*args, query_id=query_id, **kwargs)
                
                total_time = time.time() - start_time
                
                # Extract metrics from result
                if isinstance(result, dict):
                    debug_info = result.get("debug_info", {})
                    
                    # Update Prometheus metrics
                    QUERIES_TOTAL.labels(status="success").inc()
                    
                    if "tokens_used" in debug_info and debug_info["tokens_used"]:
                        tokens = debug_info["tokens_used"]
                        TOKENS_INPUT.set(tokens)  # Simplified
                    
                    # Log query
                    logger.log_query(
                        query_id=query_id,
                        query=kwargs.get("query", args[0] if args else ""),
                        num_results=debug_info.get("num_chunks_retrieved", 0),
                        retrieval_time=debug_info.get("retrieval_time", 0),
                        generation_time=debug_info.get("generation_time", 0),
                        total_time=total_time,
                        tokens_input=debug_info.get("tokens_used"),
                        status="success"
                    )
                
                return result
                
            except Exception as e:
                total_time = time.time() - start_time
                
                QUERIES_TOTAL.labels(status="error").inc()
                
                logger.log_query(
                    query_id=query_id,
                    query=kwargs.get("query", args[0] if args else ""),
                    num_results=0,
                    retrieval_time=0,
                    generation_time=0,
                    total_time=total_time,
                    status="error",
                    error=str(e)
                )
                
                raise
        
        return wrapper
    return decorator


# ============================================================================
# Initialize Monitoring
# ============================================================================

def start_metrics_server(port: Optional[int] = None):
    """
    Start Prometheus metrics server.
    
    Args:
        port: Port to listen on (default from settings)
    """
    settings = get_settings()
    port = port or settings.prometheus_port
    
    try:
        start_http_server(port)
        print(f"✓ Metrics server started on port {port}")
        print(f"  View metrics at: http://localhost:{port}/metrics")
    except OSError as e:
        print(f"Warning: Could not start metrics server on port {port}: {e}")


# ============================================================================
# Global Instances
# ============================================================================

# Create global logger instance
_logger = None

def get_logger() -> RAGLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = RAGLogger()
    return _logger
