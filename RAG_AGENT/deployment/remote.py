# C:\Users\Sunil\handson\RAG_AGENT\deployment\remote.py

import os
import sys
from google.cloud import storage
from vertexai.preview import rag
import vertexai
from absl import app, flags
from dotenv import load_dotenv
from vertexai import agent_engines
from vertexai.preview import reasoning_engines
from typing import Dict, Any, Optional

# This assumes your agent.py defines a 'root_agent' variable.
# If you switched to the create_root_agent factory, this import would change.
from rag_agent.agent import root_agent

FLAGS = flags.FLAGS
flags.DEFINE_string("description", None, "Description for the RAG corpus.")
flags.DEFINE_string("project_id", None, "GCP project ID.")
flags.DEFINE_string("location", None, "GCP location.")
flags.DEFINE_string("bucket", None, "GCP bucket.")
flags.DEFINE_string("resource_id", None, "ReasoningEngine resource ID.")
flags.DEFINE_string("user_id", "test_user", "User ID for session operations.")
flags.DEFINE_string("session_id", None, "Session ID for operations.")
flags.DEFINE_bool("create", False, "Creates a new deployment.")
flags.DEFINE_bool("delete", False, "Deletes an existing deployment.")
flags.DEFINE_bool("list", False, "Lists all deployments.")
flags.DEFINE_bool("create_session", False, "Creates a new session.")
flags.DEFINE_bool("send", False, "Sends a message to the deployed agent.")
flags.DEFINE_bool("list_corpora", False, "Lists all RAG corpora.")
flags.DEFINE_bool("get_corpus", False, "Gets details of a specific RAG corpus.")
flags.DEFINE_string("corpus_id", None, "RAG corpus ID/name for operations.")
# flags.DEFINE_string("gcs_uri", None, "GCS URI for importing documents to corpus.")
flags.DEFINE_bool("list_files", False, "Lists files in a RAG corpus.")
flags.DEFINE_string(
    "message",
    "Using the corpus named trading, what is Artificial Intelligence?",
    # "hiii",
    "Message to send to the agent.",
)

# Flags for creating a corpus
flags.DEFINE_bool("create_corpus", False, "Creates a new RAG corpus.")
flags.DEFINE_string("corpus_name", None, "Display name for the new RAG corpus.")
flags.DEFINE_string("file_path", None, "Local path OR GCS URI to upload to the corpus.")

flags.DEFINE_bool("reimport", False, "Re-import a file to corpus.")
flags.DEFINE_string("gcs_uri", None, "GCS URI for reimport.")
flags.DEFINE_bool("check_operations", False, "Check import operation status.")

flags.mark_bool_flags_as_mutual_exclusive(
    [
        "create",
        "delete",
        "list",
        "create_session",
        "send",
        "create_corpus",
        "list_corpora",
        "get_corpus", 
        "list_files",
        "check_operations",
        "reimport",
    ]
)

def create(project_id: str, location: str) -> None:
    """Creates a new deployment using your root_agent."""
    print(f"Initializing Vertex AI with project={project_id}, location={location}")
    app_obj = reasoning_engines.AdkApp(
        agent=root_agent,
        enable_tracing=True,
        env_vars={
            "project_id": project_id,
            "location": location,
        },
    )
    print("Vertex AI initialization successful")
    remote_app = agent_engines.create(
        display_name="Custom-Tools-Agent-Final",
        agent_engine=app_obj,
        requirements=[
            "google-cloud-aiplatform[adk,agent_engines,rag]",
            "google-cloud-storage",
        ],
        extra_packages=["./rag_agent"],
    )
    print(f"Created remote app: {remote_app.resource_name}")


def create_rag_corpus(
    display_name: str,
    description: Optional[str] = None,
    embedding_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a new RAG corpus in Vertex AI.
   
    Args:
        display_name: A human-readable name for the corpus
        description: Optional description for the corpus
        embedding_model: The embedding model to use (default: text-embedding-004)
   
    Returns:
        A dictionary containing the created corpus details including:
        - status: "success" or "error"
        - corpus_name: The full resource name of the created corpus
        - corpus_id: The ID portion of the corpus name
        - display_name: The human-readable name provided
        - error_message: Present only if an error occurred
    """
    if embedding_model is None:
        embedding_model = "text-embedding-004"  # Default embedding model
    try:
        # Configure embedding model
        embedding_model_config = rag.EmbeddingModelConfig(
            publisher_model=f"publishers/google/models/{embedding_model}"
        )
       
        # Create the corpus
        corpus = rag.create_corpus(
            display_name=display_name,
            description=description or f"RAG corpus: {display_name}",
            embedding_model_config=embedding_model_config,
        )
       
        # Extract corpus ID from the full name
        corpus_id = corpus.name.split('/')[-1]
       
        return {
            "status": "success",
            "corpus_name": corpus.name,
            "corpus_id": corpus_id,
            "display_name": corpus.display_name,
            "message": f"Successfully created RAG corpus '{display_name}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "message": f"Failed to create RAG corpus: {str(e)}"
        }



def delete(resource_id: str) -> None:
    remote_app = agent_engines.get(resource_id)
    remote_app.delete(force=True)
    print(f"Deleted remote app: {resource_id}")


def list_deployments() -> None:
    deployments = agent_engines.list()
    if not deployments:
        print("No deployments found.")
        return
    print("Deployments:")
    for deployment in deployments:
        print(f"- {deployment.resource_name}")


def create_session(resource_id: str, user_id: str) -> None:
    remote_app = agent_engines.get(resource_id)
    remote_session = remote_app.create_session(user_id=user_id)
    print("Created session:")
    print(f"  Session ID: {remote_session['id']}")
    print(f"  User ID: {remote_session['userId']}")
    print(f"  App name: {remote_session['appName']}")
    print("\nUse this session ID with --session_id when sending messages.")


def send_message(resource_id: str, user_id: str, session_id: str, message: str) -> None:
    remote_app = agent_engines.get(resource_id)
    print(f"Sending message to session {session_id}:")
    print(f"Message: {message}")
    print("\nResponse:")
    for event in remote_app.stream_query(
        user_id=user_id,
        session_id=session_id,
        message=message,
    ):
        print(event)


def list_corpora() -> None:
    """Lists all RAG corpora in the project."""
    print("Retrieving RAG corpora...")
    try:
        corpora_pager = rag.list_corpora()
        corpora_list = list(corpora_pager)  # Convert pager to list
        
        if not corpora_list:
            print("No RAG corpora found.")
            return
        
        print(f"\nFound {len(corpora_list)} RAG corpora:")
        print("-" * 80)
        for corpus in corpora_list:
            print(f"Display Name: {corpus.display_name}")
            print(f"Resource Name: {corpus.name}")
            print(f"Create Time: {corpus.create_time}")
            print("-" * 80)
    except Exception as e:
        print(f"Error listing corpora: {e}")


def get_corpus_details(corpus_name: str) -> None:
    """Gets detailed information about a specific RAG corpus."""
    print(f"Retrieving details for corpus: {corpus_name}")
    try:
        corpus = rag.get_corpus(name=corpus_name)
        # print(corpus)
        print(f"\nCorpus Details:")
        print(f"Display Name: {corpus.display_name}")
        print(f"Resource Name: {corpus.name}")
        
        # Additional details if available
        if hasattr(corpus, 'description') and corpus.description:
            print(f"Description: {corpus.description}")
            
    except Exception as e:
        print(f"Error getting corpus details: {e}")


def list_corpus_files(corpus_name: str) -> None:
    """Lists all files in a specific RAG corpus."""
    print(f"Retrieving files for corpus: {corpus_name}")
    try:
        files_pager = rag.list_files(corpus_name=corpus_name)
        files_list = list(files_pager)  # Convert pager to list
        
        if not files_list:
            print(f"No files found in corpus '{corpus_name}'.")
            return
            
        print(f"\nFound {len(files_list)} files in corpus '{corpus_name}':")
        print("-" * 80)
        for file in files_list:
            print(f"Display Name: {file.display_name}")
            print(f"Resource Name: {file.name}")
            print(f"Create Time: {file.create_time}")
            if hasattr(file, 'size_bytes'):
                print(f"Size: {file.size_bytes} bytes")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error listing files: {e}")


def create_rag_corpus(
    display_name: str,
    description: Optional[str] = None,
    embedding_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a new RAG corpus in Vertex AI.
   
    Args:
        display_name: A human-readable name for the corpus
        description: Optional description for the corpus
        embedding_model: The embedding model to use (default: text-embedding-004)
   
    Returns:
        A dictionary containing the created corpus details including:
        - status: "success" or "error"
        - corpus_name: The full resource name of the created corpus
        - corpus_id: The ID portion of the corpus name
        - display_name: The human-readable name provided
        - error_message: Present only if an error occurred
    """
    if embedding_model is None:
        embedding_model = "text-embedding-004"  # Default embedding model
    try:
        # Configure embedding model
        embedding_model_config = rag.EmbeddingModelConfig(
            publisher_model=f"publishers/google/models/{embedding_model}"
        )
       
        # Create the corpus
        corpus = rag.create_corpus(
            display_name=display_name,
            description=description or f"RAG corpus: {display_name}",
            embedding_model_config=embedding_model_config,
        )
       
        # Extract corpus ID from the full name
        corpus_id = corpus.name.split('/')[-1]
       
        return {
            "status": "success",
            "corpus_name": corpus.name,
            "corpus_id": corpus_id,
            "display_name": corpus.display_name,
            "message": f"Successfully created RAG corpus '{display_name}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "message": f"Failed to create RAG corpus: {str(e)}"
        }
 
def import_document_to_corpus(
    corpus_id: str,
    gcs_uri: str
) -> Dict[str, Any]:
    """
    Imports a document from Google Cloud Storage into a RAG corpus.
    Uses the minimal required parameters to avoid any compatibility issues.
   
    Args:
        corpus_id: The ID of the corpus to import the document into
        gcs_uri: GCS path of the document to import (gs://bucket-name/file-name)
   
    Returns:
        A dictionary containing:
        - status: "success" or "error"
        - corpus_id: The ID of the corpus
        - message: Status message
    """
    try:
        # Construct full corpus name
        PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
        LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION","us-central1")
        corpus_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}"
       
        # Import document with minimal configuration
        # Use the most basic form of the API call to avoid parameter issues
        result = rag.import_files(
            corpus_name,
            [gcs_uri]  # Single path in a list
        )
       
        # Return success result
        return {
            "status": "success",
            "corpus_id": corpus_id,
            "message": f"Successfully imported document {gcs_uri} to corpus '{corpus_id}'"
        }
    except Exception as e:
        return {
            "status": "error",
            "corpus_id": corpus_id,
            "error_message": str(e),
            "message": f"Failed to import document: {str(e)}"
        }
 


def check_import_operations() -> None:
    """Check the status of import operations."""
    print("Checking import operations...")
    try:
        # List recent operations
        from google.cloud import aiplatform
        
        client = aiplatform.gapic.JobServiceClient()
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        parent = f"projects/{project_id}/locations/{location}"
        
        operations = client.list_operations(name=parent)
        
        print("Recent operations:")
        for op in operations:
            print(f"Name: {op.name}")
            print(f"Done: {op.done}")
            if hasattr(op, 'metadata'):
                print(f"Metadata: {op.metadata}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error checking operations: {e}")


def search_corpus(corpus_name: str, query: str, top_k: int = 5) -> None:
    """Search for relevant chunks in a RAG corpus."""
    print(f"Searching corpus '{corpus_name}' for: '{query}'")
    try:
        response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=corpus_name,
                )
            ],
            text=query,
            similarity_top_k=top_k,
        )
        
        if not response.contexts:
            print("No relevant documents found.")
            return
            
        print(f"\nFound {len(response.contexts)} relevant chunks:")
        print("-" * 80)
        for i, context in enumerate(response.contexts, 1):
            print(f"Result {i}:")
            print(f"Distance: {context.distance}")
            print(f"Source: {context.source_uri}")
            print(f"Content: {context.text[:200]}...")  # First 200 chars
            print("-" * 80)
            
    except Exception as e:
        print(f"Error searching corpus: {e}")


def main(argv=None):
    if argv is None:
        argv = flags.FLAGS(sys.argv)
    else:
        argv = flags.FLAGS(argv)

    load_dotenv()
    project_id = (
        FLAGS.project_id if FLAGS.project_id else os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    description=FLAGS.description if FLAGS.description else "RAG Corpus"
    location = FLAGS.location if FLAGS.location else os.getenv("GOOGLE_CLOUD_LOCATION")
    bucket = FLAGS.bucket if FLAGS.bucket else os.getenv("GOOGLE_CLOUD_STAGING_BUCKET")
    user_id = FLAGS.user_id

    if not all([project_id, location, bucket]):
        print("Missing required GCP config.")
        return

    print(f"Initializing Vertex AI with project={project_id}, location={location}")
    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=bucket,
    )
    print("Vertex AI initialization successful")

    if FLAGS.create:
        create(project_id, location)
    elif FLAGS.create_corpus:
        if not FLAGS.corpus_name:
            print("Error: --corpus_name and --file_path are required.")
            return
        a=create_rag_corpus(
           FLAGS.corpus_name,FLAGS.description
        )
        print(a)
    elif FLAGS.delete:
        if not FLAGS.resource_id: 
            print("resource_id is required for delete")
            return
        delete(FLAGS.resource_id)
    elif FLAGS.list:
        list_deployments()
    elif FLAGS.create_session:
        if not FLAGS.resource_id: 
            print("resource_id is required for create_session")
            return
        create_session(FLAGS.resource_id, user_id)
    elif FLAGS.send:
        if not FLAGS.resource_id or not FLAGS.session_id: 
            print("resource_id and session_id are required for send")
            return
        send_message(FLAGS.resource_id, user_id, FLAGS.session_id, FLAGS.message)
    elif FLAGS.list_corpora:
        list_corpora()
    elif FLAGS.get_corpus:
        if not FLAGS.corpus_id:
            print("Error: --corpus_id is required for get_corpus")
            return
        get_corpus_details(FLAGS.corpus_id)
    elif FLAGS.list_files:
        if not FLAGS.corpus_id:
            print("Error: --corpus_id is required for list_files")
            return
        list_corpus_files(FLAGS.corpus_id)
    elif FLAGS.reimport:
        if not FLAGS.corpus_id or not FLAGS.gcs_uri:
            print("Error: --corpus_id and --gcs_uri are required for reimport")
            return
        a=import_document_to_corpus(
            FLAGS.corpus_id,
            FLAGS.gcs_uri
        )
        print(a)
    else:
        print(
            "Please specify one of: --create, --delete, --list, --create_corpus, "
            "--list_corpora, --get_corpus, --list_files, --create_session, or --send"
        )


if __name__ == "__main__":
    app.run(main)