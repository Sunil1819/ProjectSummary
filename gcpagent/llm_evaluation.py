import json
from google.oauth2 import service_account
import vertexai
from vertexai.preview.language_models import TextGenerationModel, EvaluationTextClassificationSpec
CONFIG = {
    "service_account_json": "C:\\Users\\Sunil\\Downloads\\GCP_creds.json",
    "project_id": "gcp-agents",
    "location": "us-central1",
    "model_name": "text-bison@001",
    "gcs_jsonl_path": "gs://testpipelinetemplate/tests.jsonl",
}

def load_credentials():
    """Load credentials from JSON service account file"""
    return service_account.Credentials.from_service_account_file(
        CONFIG["service_account_json"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

def run_evaluation():
    """Run the LLM text classification evaluation"""
    try:
        # 1. Authenticate
        credentials = load_credentials()
        
        # 2. Initialize Vertex AI
        vertexai.init(
            project=CONFIG["project_id"],
            location=CONFIG["location"],
            credentials=credentials
        )
        
        # 3. Load model
        model = TextGenerationModel.from_pretrained(CONFIG["model_name"])
        
        # 4. Set up evaluation
        eval_spec = EvaluationTextClassificationSpec(
            ground_truth_data=[CONFIG["gcs_jsonl_path"]],
            class_names=["toxic", "not_toxic"],
            target_column_name="ground_truth"
        )
        
        # 5. Run evaluation
        results = model.evaluate(task_spec=eval_spec)
        
        # 6. Print results
        print("\n=== Evaluation Results ===")
        print(f"AUPRC: {results.au_Prc:.4f}")
        print(f"AUROC: {results.au_Roc:.4f}")
        print(f"Log Loss: {results.log_Loss:.4f}")
        print("\nConfusion Matrix:")
        print(json.dumps(results.confusion_matrix, indent=2))

        return results
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    run_evaluation()