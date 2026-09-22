"""
LLM Integration for answer generation with citations.

This module demonstrates:
1. **Prompt Engineering**: Building effective RAG prompts
2. **Citation Tracking**: Mapping answers back to sources
3. **Token Management**: Staying within context limits
4. **Structured Output**: JSON responses for easy parsing

Key Concepts:
- Grounded Generation: LLM must only use provided context
- Citation Format: Clear mapping to source chunks
- Token Budgeting: Context + Answer within model limits
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken

from openai import OpenAI

from hybrid_index import RetrievalResult
from config import get_settings


@dataclass
class Answer:
    """Structured answer with citations."""
    answer_text: str
    citations: List[int]  # Indices of cited chunks
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer_text,
            "citations": self.citations,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used
        }


class PromptBuilder:
    """
    Build effective RAG prompts.
    
    Best practices:
    1. Clear role definition
    2. Explicit constraints (use only provided context)
    3. Citation requirements
    4. Output format specification
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided document excerpts.

IMPORTANT RULES:
1. Only use information from the provided citations
2. If the answer isn't in the citations, say "I don't have enough information"
3. Include citation numbers [1], [2], etc. in your answer
4. Be accurate and comprehensive
5. When multiple citations support the same point, cite each source ONCE (e.g., [1][2][3])
6. Do NOT repeat the same citation number multiple times

FORMATTING GUIDELINES - USE ACTUAL MARKDOWN SYNTAX:
- Use ## for headings (e.g., "## Architecture Components")
- Use - or • for bullet lists (e.g., "- Item one" or "• Item one")
- Use 1. 2. 3. for numbered lists
- Use **text** for bold
- Use \\n\\n for line breaks between sections (escape the newlines in JSON)
- Example:
  ## Section Title\\n\\n• First bullet point [1]\\n• Second bullet point [2]

CRITICAL: Format your response as VALID JSON with escaped newlines:
{"answer": "## Architecture Overview\\n\\nThe system includes:\\n\\n• Component 1 [1]\\n• Component 2 [2]\\n\\n## Key Features\\n\\n• Feature A [1]\\n• Feature B [3]", "citations_used": [1, 2, 3], "confidence": 0.95}

DO NOT use literal newlines in the JSON - use \\n escape sequences!"""
    
    @staticmethod
    def build_query_prompt(
        query: str,
        retrieval_results: List[RetrievalResult],
        max_context_chunks: Optional[int] = None
    ) -> str:
        """
        Build the user prompt with query and context.
        
        Args:
            query: User's question
            retrieval_results: Retrieved chunks with scores
            max_context_chunks: Limit number of chunks (for token management)
        
        Returns:
            Formatted prompt string
        """
        # Limit chunks if specified
        if max_context_chunks:
            retrieval_results = retrieval_results[:max_context_chunks]
        
        # Build citations section
        citations = []
        for i, result in enumerate(retrieval_results, start=1):
            # Format citation with metadata
            source_info = f"{result.chunk.source_id}"
            if "file_name" in result.chunk.metadata:
                source_info = result.chunk.metadata["file_name"]
            
            citation = f"[{i}] Source: {source_info}\n{result.chunk.text}"
            citations.append(citation)
        
        citations_text = "\n\n".join(citations)
        
        prompt = f"""CONTEXT CITATIONS:
{citations_text}

QUESTION: {query}

Provide your answer using only the citations above. Include citation numbers in your answer."""
        
        return prompt
    
    @staticmethod
    def estimate_tokens(text: str, model: str = "gpt-4") -> int:
        """
        Estimate token count for a text.
        
        Uses tiktoken library for accurate counting.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4


class LLMClient:
    """
    Wrapper for LLM API calls with citation support.
    
    Supports:
    - OpenAI API
    - Local LLMs (Ollama, LM Studio, etc.)
    - Query rewriting for better retrieval
    """
    
    QUERY_REWRITE_PROMPT = """You are a query optimization assistant. Your job is to improve user queries for better document retrieval.

Given a user's question, rewrite it to be:
1. More specific and detailed
2. Include relevant synonyms and related terms
3. Expand abbreviations
4. Add context if the query is vague

Examples:
User: "how to use embeddings"
Rewritten: "how to use embeddings vectors semantic search machine learning natural language processing"

User: "RAG performance"
Rewritten: "RAG retrieval augmented generation performance optimization speed accuracy metrics evaluation"

User: "install deps"
Rewritten: "install dependencies packages requirements python pip npm"

Return ONLY the rewritten query, nothing else."""
    
    def __init__(self):
        settings = get_settings()
        self.settings = settings
        
        if settings.use_local_llm:
            # Local LLM
            self.client = OpenAI(
                base_url=settings.local_llm_base_url,
                api_key="not-needed"  # Local doesn't need key
            )
            self.model = settings.local_llm_model
            print(f"Using local LLM: {self.model}")
        else:
            # OpenAI
            if not settings.openai_api_key:
                raise ValueError(
                    "OpenAI API key not configured. "
                    "Set OPENAI_API_KEY in .env or use USE_LOCAL_LLM=true"
                )
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.model = settings.openai_model
            print(f"Using OpenAI model: {self.model}")
        
        self.prompt_builder = PromptBuilder()
    
    def rewrite_query(self, query: str) -> str:
        """
        Use LLM to rewrite/expand query for better retrieval.
        
        This technique (query rewriting/expansion) helps with:
        - Vague queries: "how to install" → "how to install dependencies packages pip npm"
        - Abbreviations: "RAG" → "RAG retrieval augmented generation"
        - Context: Adds related terms for better semantic search
        
        Args:
            query: Original user query
        
        Returns:
            Rewritten/expanded query
        """
        try:
            # Build API call parameters
            call_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.QUERY_REWRITE_PROMPT},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.3,  # Lower temperature for more focused rewrites
            }
            
            # Check if model needs max_completion_tokens (newer models)
            needs_completion_tokens = any(name in self.model.lower() for name in ["gpt-4o", "gpt-5", "o1"])
            if needs_completion_tokens:
                call_params["max_completion_tokens"] = 150
            else:
                call_params["max_tokens"] = 150
            
            response = self.client.chat.completions.create(**call_params)
            
            rewritten = response.choices[0].message.content.strip()
            print('===============================================')
            print(f"Query rewrite: '{query}' → '{rewritten}'")
            print('===============================================')
            return rewritten
            
        except Exception as e:
            print(f"Query rewrite failed: {e}, using original query")
            return query
    
    def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        max_context_chunks: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Answer:
        """
        Generate an answer with citations.
        
        Process:
        1. Build prompt with query + retrieved chunks
        2. Call LLM with system instructions
        3. Parse JSON response
        4. Extract citations and answer
        
        Args:
            query: User question
            retrieval_results: Retrieved chunks
            max_context_chunks: Limit context size
            temperature: LLM temperature (lower = more focused)
        
        Returns:
            Answer object with text and citations
        """
        temperature = temperature if temperature is not None else self.settings.temperature
        
        # Build prompt
        user_prompt = self.prompt_builder.build_query_prompt(
            query=query,
            retrieval_results=retrieval_results,
            max_context_chunks=max_context_chunks
        )
        
        # Estimate tokens (for monitoring)
        system_tokens = self.prompt_builder.estimate_tokens(
            self.prompt_builder.SYSTEM_PROMPT,
            self.model
        )
        user_tokens = self.prompt_builder.estimate_tokens(user_prompt, self.model)
        total_input_tokens = system_tokens + user_tokens
        
        print(f"\nGenerating answer...")
        print(f"Input tokens: ~{total_input_tokens}")
        
        try:
            # Call LLM
            # Use max_completion_tokens for newer models and max_tokens for older models
            # Newer models (2024+) that require max_completion_tokens: gpt-4o, gpt-4o-mini, gpt-5, etc.
            call_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.prompt_builder.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
            }
            
            # Check if model needs max_completion_tokens (newer models)
            needs_completion_tokens = any(name in self.model.lower() for name in ["gpt-4o", "gpt-5", "o1"])
            if needs_completion_tokens:
                call_params["max_completion_tokens"] = self.settings.max_tokens
            else:
                call_params["max_tokens"] = self.settings.max_tokens
            
            response = self.client.chat.completions.create(**call_params)
            
            # Extract response
            response_text = response.choices[0].message.content
            
            print("\n" + "="*60)
            print("🔍 RAW LLM RESPONSE:")
            print(response_text[:500])
            print("="*60)
            
            # Clean response - remove markdown code fences if present
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.startswith("```"):
                response_text = response_text.replace("```", "", 1)
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]
            response_text = response_text.strip()
            
            # Try to parse as JSON
            try:
                # Use strict=False to allow control characters
                response_json = json.loads(response_text, strict=False)
                answer_text = response_json.get("answer", response_text)
                citations = response_json.get("citations_used", [])
                confidence = response_json.get("confidence")
                print("✅ Successfully parsed JSON")
                print(f"   Answer length: {len(answer_text)} chars")
                print(f"   Citations: {citations}")
            except json.JSONDecodeError as e:
                # Try to extract just the answer field manually
                print(f"⚠️  JSON Parse Error: {e}")
                print("   Attempting manual extraction...")
                
                import re
                # Try to extract the answer field with regex
                answer_match = re.search(r'"answer"\s*:\s*"(.*?)"\s*,\s*"citations_used"', response_text, re.DOTALL)
                citations_match = re.search(r'"citations_used"\s*:\s*\[(.*?)\]', response_text)
                
                if answer_match:
                    answer_text = answer_match.group(1)
                    # Unescape the answer
                    answer_text = answer_text.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                    print("✅ Manual extraction successful")
                else:
                    # Last resort: treat whole response as answer
                    print("❌ Manual extraction failed, using full response")
                    answer_text = response_text
                
                if citations_match:
                    citations_str = citations_match.group(1)
                    citations = [int(x.strip()) for x in citations_str.split(',') if x.strip().isdigit()]
                else:
                    citations = self._extract_citation_numbers(response_text)
                
                confidence = None
            
            # Token usage
            tokens_used = None
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
            
            return Answer(
                answer_text=answer_text,
                citations=citations,
                confidence=confidence,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            # Return fallback answer
            return Answer(
                answer_text=f"Error generating answer: {str(e)}",
                citations=[],
                confidence=0.0
            )
    
    @staticmethod
    def _extract_citation_numbers(text: str) -> List[int]:
        """
        Extract citation numbers from text like [1], [2], etc.
        
        Regex pattern: \[(\d+)\]
        """
        import re
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        return [int(m) for m in matches]


class RAGPipeline:
    """
    Complete RAG pipeline: Retrieval → Generation → Citations.
    
    This is the high-level interface that ties everything together.
    """
    
    def __init__(self, index, llm_client: Optional[LLMClient] = None):
        """
        Args:
            index: HybridRAGIndex instance
            llm_client: LLMClient instance (creates new if None)
        """
        self.index = index
        self.llm_client = llm_client or LLMClient()
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        max_context_chunks: Optional[int] = None,
        use_query_rewriting: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve + generate answer.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            max_context_chunks: Max chunks to send to LLM (for token management)
            use_query_rewriting: Whether to use LLM to improve query before retrieval
        
        Returns dict with:
        - answer: Generated answer text
        - citations: List of citation numbers used
        - retrieval_results: Full retrieval results with scores
        - debug_info: Timing, tokens, etc.
        """
        import time
        
        start_time = time.time()
        
        # Optional: Query rewriting for better retrieval
        original_query = query
        if use_query_rewriting:
            rewrite_start = time.time()
            query = self.llm_client.rewrite_query(query)
            rewrite_time = time.time() - rewrite_start
        else:
            rewrite_time = 0.0
        
        # Step 1: Retrieval
        retrieval_start = time.time()
        retrieval_results = self.index.query(query, top_k=top_k)
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Generation (use original query for answer generation)
        generation_start = time.time()
        answer = self.llm_client.generate_answer(
            query=original_query,  # Use original for answering
            retrieval_results=retrieval_results,
            max_context_chunks=max_context_chunks
        )
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        # Build response
        response = {
            "query": original_query,
            "rewritten_query": query if use_query_rewriting else None,
            "answer": answer.answer_text,
            "citations": answer.citations,
            "confidence": answer.confidence,
            "retrieval_results": [r.to_dict() for r in retrieval_results],
            "debug_info": {
                "query_rewrite_time": rewrite_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "tokens_used": answer.tokens_used,
                "num_chunks_retrieved": len(retrieval_results),
                "num_citations_used": len(answer.citations)
            }
        }
        
        return response
    
    def format_answer_with_sources(self, response: Dict[str, Any]) -> str:
        """
        Format answer with source attribution for display.
        
        Returns nicely formatted string with answer and sources.
        """
        output = []
        
        # Answer
        output.append("ANSWER:")
        output.append(response["answer"])
        output.append("")
        
        # Citations
        if response["citations"]:
            output.append("SOURCES:")
            for cite_num in response["citations"]:
                if 0 < cite_num <= len(response["retrieval_results"]):
                    result = response["retrieval_results"][cite_num - 1]
                    source = result.get("source_id", "Unknown")
                    metadata = result.get("metadata", {})
                    
                    # Extract filename if available
                    if "file_name" in metadata:
                        source = metadata["file_name"]
                    
                    output.append(f"[{cite_num}] {source}")
        
        return "\n".join(output)
