"""
Enhanced tool for listing all available Vertex AI RAG corpora with better error handling.
"""
import os
from typing import Dict, List, Union, Optional
from google.cloud import aiplatform
from vertexai import rag
import vertexai

def list_corpora(
    project_id: Optional[str] = None,
    location: str = "us-central1"
) -> dict:
    """
    List all available Vertex AI RAG corpora with enhanced error handling.
    
    Args:
        project_id: Google Cloud project ID (if None, uses default)
        location: Google Cloud location/region
        
    Returns:
        dict: A list of available corpora and status, with each corpus containing:
            - resource_name: The full resource name to use with other tools
            - display_name: The human-readable name of the corpus
            - create_time: When the corpus was created
            - update_time: When the corpus was last updated
    """
    try:
        # Initialize Vertex AI if not already done
        if project_id is None:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                # Try to get from gcloud config
                import subprocess
                try:
                    result = subprocess.run(
                        ["gcloud", "config", "get-value", "project"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    project_id = result.stdout.strip()
                except:
                    pass
        
        if not project_id:
            return {
                "status": "error",
                "message": "Project ID not found. Please set GOOGLE_CLOUD_PROJECT environment variable or pass project_id parameter.",
                "corpora": [],
            }
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=project_id, location=location)
        except Exception as init_error:
            return {
                "status": "error",
                "message": f"Failed to initialize Vertex AI: {str(init_error)}",
                "corpora": [],
            }
        
        # Alternative method 1: Try using aiplatform directly
        try:
            # Initialize aiplatform client
            aiplatform.init(project=project_id, location=location)
            
            # Use the RAG API client directly
            from google.cloud.aiplatform import rag as rag_client
            
            # Create parent path
            parent = f"projects/{project_id}/locations/{location}"
            
            # List corpora using the client
            client = rag_client.RagCorpusServiceClient()
            request = rag_client.ListRagCorporaRequest(parent=parent)
            page_result = client.list_rag_corpora(request=request)
            
            corpus_info: List[Dict[str, Union[str, int]]] = []
            for corpus in page_result:
                corpus_data: Dict[str, Union[str, int]] = {
                    "resource_name": corpus.name,
                    "display_name": corpus.display_name,
                    "create_time": str(corpus.create_time) if corpus.create_time else "",
                    "update_time": str(corpus.update_time) if corpus.update_time else "",
                }
                corpus_info.append(corpus_data)
            
            return {
                "status": "success",
                "message": f"Found {len(corpus_info)} available corpora",
                "corpora": corpus_info,
                "method": "aiplatform_client"
            }
            
        except Exception as client_error:
            # Alternative method 2: Try the original rag.list_corpora()
            try:
                corpora = rag.list_corpora()
                
                # Handle different response types
                if hasattr(corpora, '__iter__'):
                    corpus_list = list(corpora)
                else:
                    corpus_list = [corpora] if corpora else []
                
                corpus_info: List[Dict[str, Union[str, int]]] = []
                for corpus in corpus_list:
                    corpus_data: Dict[str, Union[str, int]] = {
                        "resource_name": getattr(corpus, 'name', str(corpus)),
                        "display_name": getattr(corpus, 'display_name', 'Unknown'),
                        "create_time": str(getattr(corpus, 'create_time', '')),
                        "update_time": str(getattr(corpus, 'update_time', '')),
                    }
                    corpus_info.append(corpus_data)
                
                return {
                    "status": "success",
                    "message": f"Found {len(corpus_info)} available corpora",
                    "corpora": corpus_info,
                    "method": "rag_list_corpora"
                }
                
            except Exception as rag_error:
                return {
                    "status": "error",
                    "message": f"Both methods failed. Client error: {str(client_error)}, RAG error: {str(rag_error)}",
                    "corpora": [],
                    "debug_info": {
                        "project_id": project_id,
                        "location": location,
                        "client_error": str(client_error),
                        "rag_error": str(rag_error)
                    }
                }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "corpora": [],
            "debug_info": {
                "project_id": project_id if 'project_id' in locals() else None,
                "location": location,
                "error_type": type(e).__name__
            }
        }

