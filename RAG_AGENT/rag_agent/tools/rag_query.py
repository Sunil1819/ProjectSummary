"""
Tool for querying Vertex AI RAG corpora and retrieving relevant information.
"""
import logging
from google.adk.tools.tool_context import ToolContext
from vertexai import rag
from ..config import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_TOP_K,
)
from .utils import check_corpus_exists, get_corpus_resource_name

logger = logging.getLogger(__name__)

def rag_query(
    corpus_name: str,
    query: str,
    tool_context: ToolContext,
) -> dict:
    """
    Query a Vertex AI RAG corpus with a user question and return relevant information.
    """
    try:
        logger.error(f"DEBUG RAG_QUERY: Starting query for corpus='{corpus_name}', query='{query}'")
        
        # TEMPORARY FIX: Hardcode the known corpus resource name
        if corpus_name == "trading":
            corpus_resource_name = "projects/fabled-badge-458903-t9/locations/us-central1/ragCorpora/3746994889972252672"
            logger.error(f"DEBUG RAG_QUERY: Using hardcoded resource name for 'trading': {corpus_resource_name}")
        elif corpus_name.startswith("projects/"):
            corpus_resource_name = corpus_name
            logger.error(f"DEBUG RAG_QUERY: Using provided resource name: {corpus_resource_name}")
        else:
            # Original logic for other corpora
            logger.error(f"DEBUG RAG_QUERY: Calling check_corpus_exists for '{corpus_name}'")
            corpus_exists = check_corpus_exists(corpus_name, tool_context)
            logger.error(f"DEBUG RAG_QUERY: check_corpus_exists returned: {corpus_exists}")
            
            if not corpus_exists:
                error_msg = f"Corpus '{corpus_name}' does not exist. Please create it first using the create_corpus tool."
                logger.error(f"DEBUG RAG_QUERY: Corpus does not exist, returning error: {error_msg}")
                return {
                    "status": "error",
                    "message": error_msg,
                    "query": query,
                    "corpus_name": corpus_name,
                }
            
            # Get the corpus resource name
            logger.error(f"DEBUG RAG_QUERY: Calling get_corpus_resource_name for '{corpus_name}'")
            corpus_resource_name = get_corpus_resource_name(corpus_name, tool_context)
            logger.error(f"DEBUG RAG_QUERY: get_corpus_resource_name returned: '{corpus_resource_name}'")
            
            if not corpus_resource_name:
                error_msg = f"Could not get resource name for corpus '{corpus_name}'"
                logger.error(f"DEBUG RAG_QUERY: {error_msg}")
                return {
                    "status": "error",
                    "message": error_msg,
                    "query": query,
                    "corpus_name": corpus_name,
                }
        
        # Configure retrieval parameters
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=DEFAULT_TOP_K,
            filter=rag.Filter(vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD),
        )
        
        # Perform the query
        logger.error(f"DEBUG RAG_QUERY: Performing retrieval query on corpus (resource: {corpus_resource_name}) for: '{query}'")
        print(f"\n🔎 Performing retrieval query on corpus for: '{query}'")
        response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=corpus_resource_name,
                )
            ],
            text=query,
            rag_retrieval_config=rag_retrieval_config,
        )
        
        # Rest of the function remains the same...
        results = []
        if hasattr(response, "contexts") and response.contexts:
            for ctx_group in response.contexts.contexts:
                result = {
                    "source_uri": (
                        ctx_group.source_uri if hasattr(ctx_group, "source_uri") else ""
                    ),
                    "source_name": (
                        ctx_group.source_display_name
                        if hasattr(ctx_group, "source_display_name")
                        else ""
                    ),
                    "text": ctx_group.text if hasattr(ctx_group, "text") else "",
                    "score": ctx_group.score if hasattr(ctx_group, "score") else 0.0,
                }
                results.append(result)
        
        print("\n✅ Retrieval complete. Found the following context:")
        if results:
            for i, res in enumerate(results):
                print("-" * 20)
                print(f"  Context Chunk {i+1}:")
                print(f"  Source: {res.get('source_name', 'N/A')}")
                print(f"  Relevance Score: {res.get('score', 0.0):.4f}")
                print(f"  Text: \"{res.get('text', '')}\"")
            print("-" * 20)
        else:
            print("  No relevant context was found in the documents.")
        
        if not results:
            return {
                "status": "warning",
                "message": f"No results found in corpus for query: '{query}'. This might be because the files are 0 bytes (not properly uploaded).",
                "query": query,
                "corpus_name": corpus_name,
                "results": [],
                "results_count": 0,
            }
        
        return {
            "status": "success",
            "message": f"Successfully queried corpus",
            "query": query,
            "corpus_name": corpus_name,
            "results": results,
            "results_count": len(results),
        }
        
    except Exception as e:
        error_msg = f"Error querying corpus: {str(e)}"
        logger.error(f"DEBUG RAG_QUERY: Exception occurred: {error_msg}")
        logging.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "query": query,
            "corpus_name": corpus_name,
        }