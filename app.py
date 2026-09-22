"""
Chainlit RAG Application with Interactive Debug View.

This app demonstrates:
1. **Interactive Chat**: Ask questions about your documents
2. **Debug View**: See retrieval scores, BM25 vs FAISS breakdown
3. **Citations**: Clickable source references
4. **Real-time Feedback**: Streaming responses

Features:
- Hybrid search visualization
- Score fusion transparency
- Source document inspection
- Performance metrics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import chainlit as cl
from chainlit import Message, Text, make_async
import time
import json
from typing import Dict, Any

from hybrid_index import HybridRAGIndex
from llm_client import RAGPipeline, LLMClient
from config import get_settings


# Global index (loaded once)
_index = None
_pipeline = None


def load_index():
    """Load the pre-built index."""
    global _index, _pipeline
    
    settings = get_settings()
    index_dir = settings.index_dir
    
    if not Path(index_dir).exists():
        raise FileNotFoundError(
            f"Index not found at {index_dir}. "
            f"Please run: python src/build_index.py"
        )
    
    print("Loading index...")
    _index = HybridRAGIndex()
    _index.load(index_dir)
    
    print("Initializing LLM client...")
    llm_client = LLMClient()
    
    _pipeline = RAGPipeline(_index, llm_client)
    
    print("✓ System ready!")


def format_retrieval_debug(retrieval_results: list) -> str:
    """Format retrieval results for debug view."""
    lines = []
    
    for i, result in enumerate(retrieval_results[:5], 1):  # Show top 5
        lines.append(f"**Rank {i}** (Score: {result['score']:.3f})")
        lines.append(f"- BM25: {result['bm25_score']:.3f} | FAISS: {result['faiss_score']:.3f}")
        lines.append(f"- Source: `{result['source_id']}`")
        
        # Show snippet
        text = result['text'].strip()
        text = ' '.join(text.split())
        snippet = text[:200] + "..." if len(text) > 200 else text
        lines.append(f"- Text: {snippet}\n")
    
    return "\n".join(lines)


def parse_llm_response(answer_text: str) -> str:
    """
    Parse LLM response and decode escape sequences.
    The answer_text should already be extracted from JSON in llm_client.py,
    but if it's still JSON, parse it here as a fallback.
    """
    # Check if we still have JSON
    if answer_text and answer_text.strip().startswith('{'):
        try:
            # Still JSON - parse it
            parsed = json.loads(answer_text)
            answer_text = parsed.get("answer", answer_text)
            print("⚠️  Had to parse JSON in app.py - llm_client didn't extract it properly")
        except json.JSONDecodeError:
            print("⚠️  Answer looks like JSON but couldn't parse")
            pass
    
    # Replace escaped newlines with actual newlines
    if answer_text:
        # Decode escape sequences
        answer_text = answer_text.replace('\\n', '\n').replace('\\t', '\t')
    
    return answer_text


def format_sources(response: Dict[str, Any]) -> str:
    """Format sources with citations (deduplicated)."""
    if not response["citations"]:
        return ""
    
    # Deduplicate citations
    unique_citations = sorted(set(response["citations"]))
    
    sources = []
    for cite_num in unique_citations:
        if 0 < cite_num <= len(response["retrieval_results"]):
            result = response["retrieval_results"][cite_num - 1]
            source_id = result.get("source_id", "Unknown")
            metadata = result.get("metadata", {})
            
            # Get filename if available
            if "file_name" in metadata:
                filename = metadata["file_name"]
            else:
                filename = source_id
            
            sources.append(f"**[{cite_num}]** {filename}")
            
            # Show text snippet
            text = result["text"].strip()
            text = ' '.join(text.split())
            snippet = text[:300] + "..." if len(text) > 300 else text
            sources.append(f"> {snippet}\n")
    
    return "\n".join(sources)


def format_performance_metrics(debug_info: Dict[str, Any]) -> str:
    """Format performance metrics."""
    metrics = []
    metrics.append(f"- **Total Time**: {debug_info['total_time']:.2f}s")
    metrics.append(f"- **Retrieval Time**: {debug_info['retrieval_time']:.2f}s")
    metrics.append(f"- **Generation Time**: {debug_info['generation_time']:.2f}s")
    
    if 'query_rewrite_time' in debug_info and debug_info['query_rewrite_time'] > 0:
        metrics.append(f"- **Query Rewrite Time**: {debug_info['query_rewrite_time']:.2f}s")
    
    metrics.append(f"- **Tokens Used**: {debug_info.get('tokens_used', 'N/A')}")
    metrics.append(f"- **Chunks Retrieved**: {debug_info['num_chunks_retrieved']}")
    metrics.append(f"- **Citations Used**: {debug_info['num_citations_used']}")
    
    return "\n".join(metrics)


def get_unique_citation_count(citations: list) -> int:
    """Get count of unique citations."""
    return len(set(citations))


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    # Load index on first start
    global _index, _pipeline
    if _index is None:
        try:
            load_index()
        except Exception as e:
            await cl.Message(
                content=f"❌ **Error loading index:** {str(e)}\n\n"
                        f"Please build the index first:\n"
                        f"```bash\n"
                        f"python src/build_index.py --data-dir ./data --index-dir ./indexes\n"
                        f"```"
            ).send()
            return
    
    # Send welcome message
    settings = get_settings()
    stats = _index.get_stats()
    
    welcome = f"""# 🚀 Welcome to RAG Master!

I'm ready to answer questions about your documents using **hybrid search** (BM25 + semantic embeddings).

## 📊 Index Stats
- **Documents**: {stats['total_documents']}
- **Chunks**: {stats['total_chunks']}
- **Embedding Model**: {stats['model_name']}
- **LLM**: {settings.openai_model if not settings.use_local_llm else settings.local_llm_model}

## 💡 Try asking:
- "What is the main topic of the documents?"
- "Summarize the key points"
- "Find information about [specific topic]"

## 🔍 Features
- ✅ Hybrid retrieval (keyword + semantic)
- ✅ Grounded answers with citations
- ✅ Debug view showing retrieval scores
- ✅ Source document references

Ask me anything!
"""
    
    await cl.Message(content=welcome).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    query = message.content
    
    if not _pipeline:
        await cl.Message(content="❌ System not initialized. Please restart.").send()
        return
    
    # Show thinking message
    thinking_msg = cl.Message(content="🔍 Searching documents...")
    await thinking_msg.send()
    
    try:
        # Run RAG pipeline
        start_time = time.time()
        settings = get_settings()
        
        response = _pipeline.query(
            query=query,
            top_k=settings.top_k_retrieval,
            max_context_chunks=5,  # Reduced from 10 to 5 for faster generation
            use_query_rewriting=True  # Enable query rewriting
        )
        
        query_time = time.time() - start_time
        
        # Update thinking message to show retrieval complete
        thinking_msg.content = "✓ Found relevant chunks. Generating answer..."
        await thinking_msg.update()
        
        # Parse and format the answer
        answer_text = parse_llm_response(response["answer"])
        
        # Debug: Print what we got
        print("="*50)
        print("RAW ANSWER FROM RESPONSE:")
        print(response["answer"][:500])
        print("="*50)
        print("PARSED ANSWER:")
        print(answer_text[:500])
        print("="*50)
        
        # Remove thinking message
        await thinking_msg.remove()
        
        # Send main answer with proper markdown rendering
        answer_msg = cl.Message(content=answer_text)
        await answer_msg.send()
        
        # Send additional info in a compact format
        # Use a single message with expandable sections via markdown
        
        # Build the additional info message
        info_parts = []
        
        # 1. Sources
        sources_content = format_sources(response)
        if sources_content:
            unique_count = get_unique_citation_count(response["citations"])
            info_parts.append(f"### 📚 Sources \n\n{sources_content}")
        
        # 2. Performance Metrics
        metrics_content = format_performance_metrics(response["debug_info"])
        info_parts.append(f"### ⚡ Performance Metrics\n\n{metrics_content}")
        
        # 3. Retrieval Debug
        retrieval_content = format_retrieval_debug(response["retrieval_results"])
        info_parts.append(f"### 🔍 Retrieval Debug (Top 5)\n\n{retrieval_content}")
        
        # Send as one message for cleaner UI
        info_msg = cl.Message(content="\n\n---\n\n".join(info_parts))
        await info_msg.send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error:** {str(e)}\n\nPlease check your configuration and try again."
        ).send()
        
        # Remove thinking message if it still exists
        try:
            await thinking_msg.remove()
        except:
            pass


@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates."""
    print("Settings updated:", settings)


if __name__ == "__main__":
    # This allows running with: python app.py
    # But normally you'd use: chainlit run app.py
    pass
