

from google.adk.tools.tool_context import ToolContext
from vertexai.preview import rag
import logging

logger = logging.getLogger(__name__)

def DUMP_AVAILABLE_CORPORA(tool_context: ToolContext) -> dict:
    """
    FOR DEBUGGING ONLY. Lists all available RAG corpora that the agent can see from the server.
    This helps diagnose why a corpus might not be found. It takes no arguments.
    """
    project_id = tool_context.app_config.get("project_id")
    location = tool_context.app_config.get("location")

    if not project_id or not location:
        return {"status": "CRITICAL_ERROR", "message": "Project ID or Location not found in agent's environment."}

    try:
        # This is the raw API call from the server
        corpora = rag.list_corpora(project=project_id, location=location)
        
        # Create a simple list of the display names the agent sees
        seen_names = [c.display_name for c in corpora]
        
        return {
            "status": "success",
            "corpora_found_count": len(seen_names),
            "corpora_display_names": seen_names,
        }
    except Exception as e:
        # Return the actual error message for debugging
        return {"status": "API_ERROR", "message": str(e)}