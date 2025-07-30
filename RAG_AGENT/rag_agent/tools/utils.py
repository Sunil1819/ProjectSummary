import logging
import re
from google.adk.tools.tool_context import ToolContext
from vertexai.preview import rag

logger = logging.getLogger(__name__)

def get_corpus_resource_name(corpus_name: str, tool_context: ToolContext) -> str:
    """
    Finds the full resource name for a corpus by its display name using flexible matching.
    
    Args:
        corpus_name (str): The display name of the corpus from user input.
        tool_context (ToolContext): The context object containing app_config.
    Returns:
        str: The full resource name of the corpus, or an empty string if not found.
    """
    logger.info(f"Looking for corpus with name: '{corpus_name}'")

    # If already a full resource name, return as-is
    if re.match(r"^projects/[^/]+/locations/[^/]+/ragCorpora/[^/]+$", corpus_name):
        logger.info(f"Input is already a full resource name: {corpus_name}")
        return corpus_name

    project_id = tool_context.app_config.get("project_id")
    location = tool_context.app_config.get("location")
    
    logger.info(f"Using project_id: {project_id}, location: {location}")

    if not project_id or not location:
        logger.error("Project ID or Location not found in app_config")
        return ""

    try:
        logger.info("Fetching corpora list from Vertex AI RAG...")
        corpora = rag.list_corpora(project=project_id, location=location)
        corpora_list = list(corpora)
        
        logger.info(f"Found {len(corpora_list)} corpora total")
        
        if not corpora_list:
            logger.warning(f"No corpora found in project {project_id}")
            return ""
        
        user_input_clean = corpus_name.lower().strip()
        
        # Multiple matching strategies
        matches = []
        
        for corpus in corpora_list:
            display_name = getattr(corpus, 'display_name', '')
            resource_name = corpus.name
            
            # Extract corpus ID from resource name for matching
            corpus_id_match = re.search(r'/ragCorpora/([^/]+)$', resource_name)
            corpus_id = corpus_id_match.group(1) if corpus_id_match else ''
            
            logger.debug(f"Checking corpus - Display: '{display_name}', ID: '{corpus_id}', Resource: {resource_name}")
            
            # Strategy 1: Exact display name match
            if display_name and display_name.lower().strip() == user_input_clean:
                logger.info(f"✓ Exact display name match: '{display_name}'")
                return resource_name
            
            # Strategy 2: Partial display name match
            if display_name and user_input_clean in display_name.lower():
                matches.append(('partial_display', display_name, resource_name))
                
            # Strategy 3: Corpus ID match
            if corpus_id and corpus_id.lower() == user_input_clean:
                logger.info(f"✓ Corpus ID match: '{corpus_id}'")
                return resource_name
                
            # Strategy 4: Partial corpus ID match
            if corpus_id and user_input_clean in corpus_id.lower():
                matches.append(('partial_id', corpus_id, resource_name))
        
        # If we have matches, prefer display name matches over ID matches
        if matches:
            # Sort by match type priority: partial_display > partial_id
            matches.sort(key=lambda x: 0 if x[0] == 'partial_display' else 1)
            match_type, match_name, resource_name = matches[0]
            logger.info(f"✓ Found {match_type} match: '{match_name}' -> {resource_name}")
            return resource_name
            
        # No matches found - log available options
        logger.warning(f"✗ No corpus found matching: '{corpus_name}'")
        logger.info("Available corpora:")
        for corpus in corpora_list:
            display_name = getattr(corpus, 'display_name', 'N/A')
            corpus_id_match = re.search(r'/ragCorpora/([^/]+)$', corpus.name)
            corpus_id = corpus_id_match.group(1) if corpus_id_match else 'N/A'
            logger.info(f"  - Display: '{display_name}', ID: '{corpus_id}'")
        
        return ""
        
    except Exception as e:
        logger.error(f"Error fetching corpora: {str(e)}")
        raise


def find_corpus_by_any_identifier(identifier: str, tool_context: ToolContext) -> str:
    """
    Find a corpus by any identifier - display name, corpus ID, or full resource name.
    
    Args:
        identifier (str): Any identifier for the corpus
        tool_context (ToolContext): The tool context
    Returns:
        str: The full resource name if found, empty string otherwise
    """
    logger.info(f"Finding corpus by identifier: '{identifier}'")
    
    # First try the main function
    result = get_corpus_resource_name(identifier, tool_context)
    if result:
        return result
    
    # If that fails, try some common variations
    variations = [
        identifier.lower(),
        identifier.upper(), 
        identifier.title(),
        identifier.replace('_', '-'),
        identifier.replace('-', '_')
    ]
    
    for variation in variations:
        if variation != identifier:
            logger.info(f"Trying variation: '{variation}'")
            result = get_corpus_resource_name(variation, tool_context)
            if result:
                logger.info(f"✓ Found match with variation: '{variation}'")
                return result
    
    return ""


def check_corpus_exists(corpus_name: str, tool_context: ToolContext) -> bool:
    """
    Check if a corpus exists using flexible matching.
    
    Args:
        corpus_name (str): The identifier of the corpus to check.
        tool_context (ToolContext): The tool context for state and app_config.
    Returns:
        bool: True if the corpus exists, False otherwise.
    """
    logger.info(f"Checking if corpus exists: '{corpus_name}'")
    
    # Check cache first
    state_key = f"corpus_exists_{corpus_name}"
    if tool_context.state.get(state_key):
        logger.info(f"Corpus '{corpus_name}' found in cache")
        return True
        
    try:
        # Try flexible matching
        corpus_resource_name = find_corpus_by_any_identifier(corpus_name, tool_context)
        
        if corpus_resource_name:
            logger.info(f"✓ Corpus '{corpus_name}' exists: {corpus_resource_name}")
            # Cache the result
            tool_context.state[state_key] = True
            tool_context.state[f"corpus_resource_{corpus_name}"] = corpus_resource_name
            
            if not tool_context.state.get("current_corpus"):
                tool_context.state["current_corpus"] = corpus_name
                logger.info(f"Set '{corpus_name}' as current corpus")
            return True
        else:
            logger.info(f"✗ Corpus '{corpus_name}' does not exist")
            return False
            
    except Exception as e:
        logger.error(f"Error checking if corpus '{corpus_name}' exists: {str(e)}")
        return False


def get_corpus_resource_name_from_cache(corpus_name: str, tool_context: ToolContext) -> str:
    """
    Get the cached resource name for a corpus.
    
    Args:
        corpus_name (str): The corpus identifier
        tool_context (ToolContext): The tool context
    Returns:
        str: The cached resource name or empty string
    """
    return tool_context.state.get(f"corpus_resource_{corpus_name}", "")


def set_current_corpus(corpus_name: str, tool_context: ToolContext) -> bool:
    """
    Set the current corpus in the tool context state.
    
    Args:
        corpus_name (str): The name of the corpus to set as current.
        tool_context (ToolContext): The tool context for state management.
    Returns:
        bool: True if the corpus exists and was set as current, False otherwise.
    """
    logger.info(f"Setting current corpus to: '{corpus_name}'")
    
    if check_corpus_exists(corpus_name, tool_context):
        tool_context.state["current_corpus"] = corpus_name
        logger.info(f"✓ Current corpus set to: '{corpus_name}'")
        return True
    else:
        logger.warning(f"✗ Cannot set current corpus - '{corpus_name}' not found")
        return False


def get_current_corpus_resource_name(tool_context: ToolContext) -> str:
    """
    Get the resource name of the current corpus.
    
    Args:
        tool_context (ToolContext): The tool context
    Returns:
        str: The resource name of the current corpus
    """
    current_corpus = tool_context.state.get("current_corpus")
    if not current_corpus:
        logger.warning("No current corpus set")
        return ""
    
    # Try to get from cache first
    resource_name = get_corpus_resource_name_from_cache(current_corpus, tool_context)
    if resource_name:
        return resource_name
    
    # If not in cache, look it up
    return find_corpus_by_any_identifier(current_corpus, tool_context)


def debug_corpus_info(corpus_identifier: str, tool_context: ToolContext) -> None:
    """
    Debug function to show detailed information about corpus lookup.
    
    Args:
        corpus_identifier (str): The corpus identifier to debug
        tool_context (ToolContext): The tool context
    """
    print(f"\n=== DEBUG: Corpus Info for '{corpus_identifier}' ===")
    
    try:
        project_id = tool_context.app_config.get("project_id")
        location = tool_context.app_config.get("location")
        print(f"Project: {project_id}")
        print(f"Location: {location}")
        
        corpora = rag.list_corpora(project=project_id, location=location)
        corpora_list = list(corpora)
        
        print(f"\nFound {len(corpora_list)} total corpora:")
        
        target_lower = corpus_identifier.lower().strip()
        
        for i, corpus in enumerate(corpora_list, 1):
            display_name = getattr(corpus, 'display_name', 'N/A')
            resource_name = corpus.name
            
            # Extract corpus ID
            corpus_id_match = re.search(r'/ragCorpora/([^/]+)$', resource_name)
            corpus_id = corpus_id_match.group(1) if corpus_id_match else 'N/A'
            
            print(f"\n{i}. Corpus Details:")
            print(f"   Display Name: '{display_name}'")
            print(f"   Corpus ID: '{corpus_id}'")
            print(f"   Resource Name: {resource_name}")
            
            # Check matches
            matches = []
            if display_name and display_name.lower().strip() == target_lower:
                matches.append("EXACT display name")
            if display_name and target_lower in display_name.lower():
                matches.append("PARTIAL display name")
            if corpus_id and corpus_id.lower() == target_lower:
                matches.append("EXACT corpus ID")
            if corpus_id and target_lower in corpus_id.lower():
                matches.append("PARTIAL corpus ID")
                
            if matches:
                print(f"   🎯 MATCHES: {', '.join(matches)}")
        
        # Try the actual lookup
        print(f"\n=== Lookup Results ===")
        result = find_corpus_by_any_identifier(corpus_identifier, tool_context)
        if result:
            print(f"✓ Found: {result}")
        else:
            print(f"✗ Not found")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")


# Usage example:
"""
# To debug what's happening with "trading":
debug_corpus_info("trading", tool_context)

# To check if it exists:
exists = check_corpus_exists("trading", tool_context)
print(f"Trading corpus exists: {exists}")

# To get the resource name:
resource_name = find_corpus_by_any_identifier("trading", tool_context)
print(f"Resource name: {resource_name}")
"""