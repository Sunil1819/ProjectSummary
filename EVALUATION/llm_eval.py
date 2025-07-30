import os
import json
import pandas as pd
from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.preview.evaluation import (
    PointwiseMetric, 
    PairwiseMetric, 
    EvalTask,
    AutoraterConfig
)
from vertexai.preview.evaluation.autorater_utils import evaluate_autorater
import vertexai

def setup_gcp_authentication(json_key_path, project_id, location="us-central1"):
    """
    Set up GCP authentication using service account JSON key
    
    Args:
        json_key_path (str): Path to your GCP service account JSON key file
        project_id (str): Your GCP project ID
        location (str): GCP region (default: us-central1)
    """
    print("Setting up GCP authentication...")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_key_path
    try:
        credentials = service_account.Credentials.from_service_account_file(
            json_key_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        print(f"✓ Successfully loaded credentials from: {json_key_path}")
    except Exception as e:
        print(f"✗ Error loading credentials: {e}")
        return False
    try:
        vertexai.init(
            project=project_id,
            location=location,
            credentials=credentials
        )
        print(f"✓ Vertex AI initialized for project: {project_id}")
        print(f"✓ Location: {location}")
        return True
    except Exception as e:
        print(f"✗ Error initializing Vertex AI: {e}")
        return False

def test_connection():
    """Test the GCP connection by listing available models"""
    try:
        # Test connection by initializing aiplatform
        print("\nTesting GCP connection...")
        models = aiplatform.Model.list()
        print(f"✓ Connection successful! Found {len(models)} models in project.")
        return True
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False

def calculate_manual_accuracy(metrics_table, metric_name):
    """Manually calculate judge model accuracy"""
    try:
        # Get the judge's predictions and ground truth
        judge_col = f"{metric_name}/score"
        
        if judge_col not in metrics_table.columns:
            print(f"✗ Judge score column '{judge_col}' not found")
            return None
            
        if metric_name not in metrics_table.columns:
            print(f"✗ Ground truth column '{metric_name}' not found")
            return None
            
        judge_scores = metrics_table[judge_col].tolist()
        ground_truth = metrics_table[metric_name].tolist()
        
        print(f"Judge scores: {judge_scores}")
        print(f"Ground truth: {ground_truth}")
        
        # Calculate accuracy
        correct_predictions = sum(1 for j, g in zip(judge_scores, ground_truth) if j == g)
        total_predictions = len(judge_scores)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"✓ Manual Accuracy Calculation:")
        print(f"  - Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"  - Accuracy: {accuracy:.2%}")
        print(f"  - Detailed comparison:")
        for i, (j, g) in enumerate(zip(judge_scores, ground_truth)):
            status = "✓" if j == g else "✗"
            print(f"    {status} Sample {i+1}: Judge={j}, Truth={g}")
            
        return accuracy
        
    except Exception as e:
        print(f"Manual accuracy calculation failed: {e}")
        return None

def debug_metrics_table(metrics_table, metric_name):
    """Debug the metrics table to understand structure"""
    print(f"\n=== Debugging Metrics Table for {metric_name} ===")
    
    print("Available columns:", list(metrics_table.columns))
    print("Table shape:", metrics_table.shape)
    print("\nFirst few rows:")
    print(metrics_table.head())
    
    # Check for expected columns
    expected_cols = [
        metric_name,
        f"{metric_name}/pairwise_choice",
        f"{metric_name}/explanation"
    ]
    
    for col in expected_cols:
        if col in metrics_table.columns:
            print(f"✓ Found column: {col}")
        else:
            print(f"✗ Missing column: {col}")

def run_judge_model_evaluation():
    """Run the complete judge model evaluation workflow - FIXED VERSION WITH ACCURACY"""
    
    print("\n" + "="*60)
    print("RUNNING JUDGE MODEL EVALUATION - WITH ACCURACY FIX")
    print("="*60)
    
    # Test Data 1: PointwiseMetric - Response Quality with FIXED column naming
    print("\n=== PointwiseMetric Test - Fixed Ground Truth ===")
    
    pointwise_data = pd.DataFrame({
        "prompt": [
            "What is machine learning?",
            "Explain photosynthesis",
            "How does gravity work?",
            "What causes rain?",
            "Define artificial intelligence"
        ],
        "response": [
            "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "Plants make food using sunlight somehow.",
            "Gravity is a fundamental force that attracts objects with mass, described by Einstein's relativity.",
            "Rain happens when water gets heavy and falls.",
            "AI refers to computer systems that perform tasks requiring human intelligence."
        ],
        # FIXED: Use the exact metric name as column name for ground truth
        "response_quality": [1, 0, 1, 0, 1]  # Must match metric name exactly
    })
    
    quality_metric = PointwiseMetric(
        metric="response_quality",
        metric_prompt_template="""
        Evaluate the quality of this response (0 or 1):
        - 1: Accurate, clear, comprehensive
        - 0: Inaccurate, vague, incomplete
        
        Question: {prompt}
        Response: {response}
        
        Score (return only 0 or 1):
        """,
        system_instruction="You are an expert evaluator focusing on accuracy and clarity. Return only 0 or 1."
    )
    
    try:
        # Run pointwise evaluation
        pointwise_result = EvalTask(
            dataset=pointwise_data,
            metrics=[quality_metric],
            autorater_config=AutoraterConfig(sampling_count=3)
        ).evaluate()
        
        print("✓ PointwiseMetric evaluation completed")
        print("Summary metrics:", pointwise_result.summary_metrics)
        
        # Debug the metrics table structure
        debug_metrics_table(pointwise_result.metrics_table, "response_quality")
        
        # Method 1: Try evaluate_autorater
        print("\n=== Evaluating Judge Model Accuracy ===")
        try:
            judge_quality = evaluate_autorater(
                evaluate_autorater_input=pointwise_result.metrics_table,
                eval_metrics=[quality_metric]
            )
            print("Judge model accuracy (evaluate_autorater):", judge_quality)
            
            if hasattr(judge_quality, 'summary_metrics') and judge_quality.summary_metrics:
                print("Detailed accuracy metrics:", judge_quality.summary_metrics)
            else:
                print("No detailed accuracy metrics from evaluate_autorater")
                
        except Exception as e:
            print(f"evaluate_autorater failed: {e}")
            
        # Method 2: Manual accuracy calculation (backup)
        print("\n=== Manual Accuracy Calculation ===")
        manual_accuracy = calculate_manual_accuracy(pointwise_result.metrics_table, "response_quality")
        
        return pointwise_result, manual_accuracy
        
    except Exception as e:
        print(f"✗ PointwiseMetric evaluation failed: {e}")
        return None, None

def run_pairwise_evaluation_fixed():
    """Run fixed pairwise evaluation"""
    
    print("\n=== PairwiseMetric Test - FIXED ===")
    
    pairwise_data = pd.DataFrame({
        "prompt": [
            "What is the capital of France?",
            "How do you make coffee?",
            "Explain blockchain",
            "What causes climate change?"
        ],
        "baseline_model_response": [
            "Paris is the capital of France.",
            "Put coffee in water and heat it.",
            "Blockchain is a digital ledger.",
            "Human activities cause climate change."
        ],
        "candidate_model_response": [
            "The capital city of France is Paris, which is also its largest city and cultural center.",
            "To make coffee: heat water to 195-205°F, use 1-2 tbsp ground coffee per 6oz water, steep 4-6 minutes.",
            "Blockchain is a distributed ledger technology with cryptographically linked blocks that resist tampering.",
            "Climate change results from increased greenhouse gases from fossil fuel burning, deforestation, and industrial processes."
        ],
        # FIXED: Ground truth column for pairwise comparison
        "completeness": ["CANDIDATE", "CANDIDATE", "CANDIDATE", "CANDIDATE"]
    })
    
    completeness_metric = PairwiseMetric(
        metric="completeness",
        metric_prompt_template="""
        Compare response completeness:
        
        Question: {prompt}
        
        ### Response A (Baseline)
        {baseline_model_response}
        
        ### Response B (Candidate)
        {candidate_model_response}
        
        Which is more complete? Choose: BASELINE, CANDIDATE, or TIE
        """,
        system_instruction="You are evaluating response completeness. Consider detail, accuracy, and usefulness."
    )
    
    try:
        # FIXED: Removed flip_enabled=True to avoid column mapping issues
        pairwise_result = EvalTask(
            dataset=pairwise_data,
            metrics=[completeness_metric],
            autorater_config=AutoraterConfig(sampling_count=3)
        ).evaluate()
        
        print("✓ PairwiseMetric evaluation completed")
        print("Summary metrics:", pairwise_result.summary_metrics)
        
        # Debug pairwise metrics table
        debug_metrics_table(pairwise_result.metrics_table, "completeness")
        print("\nFull PairwiseMetric Results:", pairwise_result.metrics_table.columns)
        
        # Try to evaluate pairwise judge accuracy
        try:
            pairwise_judge_quality = evaluate_autorater(
                evaluate_autorater_input=pairwise_result.metrics_table,
                eval_metrics=[completeness_metric]
            )
            print("Pairwise judge model accuracy:", pairwise_judge_quality)
        except Exception as e:
            print(f"Pairwise judge accuracy evaluation failed: {e}")
            
        # Manual pairwise accuracy calculation
        try:
            if "completeness/pairwise_choice" in pairwise_result.metrics_table.columns and "completeness" in pairwise_result.metrics_table.columns:
                judge_choices = pairwise_result.metrics_table["completeness/pairwise_choice"].tolist()
                ground_truth_choices = pairwise_result.metrics_table["completeness"].tolist()
                
                correct = sum(1 for j, g in zip(judge_choices, ground_truth_choices) if j == g)
                total = len(judge_choices)
                accuracy = correct / total if total > 0 else 0
                
                print(f"✓ Manual Pairwise Accuracy: {accuracy:.2%} ({correct}/{total})")
                print(f"Judge choices: {judge_choices}")
                print(f"Ground truth: {ground_truth_choices}")
            else:
                print("Could not perform manual pairwise accuracy calculation - missing columns")
        except Exception as e:
            print(f"Manual pairwise accuracy calculation failed: {e}")
        
        return pairwise_result
        
    except Exception as e:
        print(f"✗ PairwiseMetric evaluation failed: {e}")
        return None

def run_pairwise_evaluation_alternative():
    """Alternative approach using separate PointwiseMetric evaluations for comparison"""
    
    print("\n=== Alternative Pairwise Evaluation Using Separate PointwiseMetrics ===")
    
    # Create two separate pointwise evaluations instead of pairwise
    baseline_data = pd.DataFrame({
        "prompt": [
            "What is the capital of France?",
            "How do you make coffee?",
            "Explain blockchain", 
            "What causes climate change?"
        ],
        "response": [
            "Paris is the capital of France.",
            "Put coffee in water and heat it.",
            "Blockchain is a digital ledger.",
            "Human activities cause climate change."
        ]
    })
    
    candidate_data = pd.DataFrame({
        "prompt": [
            "What is the capital of France?",
            "How do you make coffee?",
            "Explain blockchain",
            "What causes climate change?"
        ],
        "response": [
            "The capital city of France is Paris, which is also its largest city and cultural center.",
            "To make coffee: heat water to 195-205°F, use 1-2 tbsp ground coffee per 6oz water, steep 4-6 minutes.",
            "Blockchain is a distributed ledger technology with cryptographically linked blocks that resist tampering.",
            "Climate change results from increased greenhouse gases from fossil fuel burning, deforestation, and industrial processes."
        ]
    })
    
    # Create a quality metric for both
    completeness_quality_metric = PointwiseMetric(
        metric="response_completeness",
        metric_prompt_template="""
        Rate the completeness of this response (0-1 scale):
        - 1: Very complete, detailed, comprehensive
        - 0.5: Moderately complete
        - 0: Incomplete, vague, or insufficient
        
        Question: {prompt}
        Response: {response}
        
        Score (return only a number between 0 and 1):
        """,
        system_instruction="You are evaluating response completeness and detail. Return only a numeric score."
    )
    
    try:
        # Evaluate baseline responses
        baseline_result = EvalTask(
            dataset=baseline_data,
            metrics=[completeness_quality_metric],
            autorater_config=AutoraterConfig(sampling_count=2)
        ).evaluate()
        
        # Evaluate candidate responses  
        candidate_result = EvalTask(
            dataset=candidate_data,
            metrics=[completeness_quality_metric],
            autorater_config=AutoraterConfig(sampling_count=2)
        ).evaluate()
        
        print("✓ Alternative evaluation completed")
        
        baseline_mean = baseline_result.summary_metrics.get('response_completeness/mean', 0)
        candidate_mean = candidate_result.summary_metrics.get('response_completeness/mean', 0)
        
        print(f"Baseline mean score: {baseline_mean:.3f}")
        print(f"Candidate mean score: {candidate_mean:.3f}")
        
        if candidate_mean > baseline_mean:
            print(f"✓ Candidate model performs better ({candidate_mean:.3f} vs {baseline_mean:.3f})")
        elif baseline_mean > candidate_mean:
            print(f"⚠ Baseline model performs better ({baseline_mean:.3f} vs {candidate_mean:.3f})")
        else:
            print(f"= Models perform equally ({baseline_mean:.3f} vs {candidate_mean:.3f})")
            
        # Calculate win rate manually
        try:
            baseline_scores = baseline_result.metrics_table['response_completeness/score'].tolist()
            candidate_scores = candidate_result.metrics_table['response_completeness/score'].tolist()
            
            if len(baseline_scores) == len(candidate_scores):
                
                candidate_wins = sum(1 for b, c in zip(baseline_scores, candidate_scores) if c > b)
                baseline_wins = sum(1 for b, c in zip(baseline_scores, candidate_scores) if b > c)
                ties = sum(1 for b, c in zip(baseline_scores, candidate_scores) if abs(b - c) < 0.01)  # Allow small floating point differences
                total = len(baseline_scores)
                
                print(f"\nDetailed Comparison:")
                print(f"- Candidate wins: {candidate_wins}/{total} ({candidate_wins/total*100:.1f}%)")
                print(f"- Baseline wins: {baseline_wins}/{total} ({baseline_wins/total*100:.1f}%)")
                print(f"- Ties: {ties}/{total} ({ties/total*100:.1f}%)")
                
                # Show individual scores
                print(f"\nIndividual Scores:")
                for i, (b, c) in enumerate(zip(baseline_scores, candidate_scores)):
                    winner = "Candidate" if c > b else "Baseline" if b > c else "Tie"
                    print(f"  Question {i+1}: Baseline={b:.2f}, Candidate={c:.2f} → {winner}")
        except Exception as e:
            print(f"Could not calculate detailed comparison: {e}")
            
        return baseline_result, candidate_result
            
    except Exception as e:
        print(f"✗ Alternative evaluation failed: {e}")
        return None, None

def analyze_results(pointwise_result, pairwise_result, pointwise_accuracy):
    """Analyze and summarize all evaluation results"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*60)
    
    # Pointwise Analysis
    if pointwise_result:
        print("\n=== PointwiseMetric Results ===")
        summary = pointwise_result.summary_metrics
        print(f"- Mean score: {summary.get('response_quality/mean', 'N/A')}")
        print(f"- Standard deviation: {summary.get('response_quality/std', 'N/A')}")
        print(f"- Row count: {summary.get('row_count', 'N/A')}")
        
        if pointwise_accuracy:
            print(f"- Judge Model Accuracy: {pointwise_accuracy:.2%}")
        
        print("\nDetailed PointwiseMetric Results:")
        try:
            df = pointwise_result.metrics_table
            display_cols = ['prompt', 'response', 'response_quality', 'response_quality/score']
            available_cols = [col for col in display_cols if col in df.columns]
            print(df[available_cols])
        except Exception as e:
            print(f"Could not display detailed results: {e}")
    
    # Pairwise Analysis
    if pairwise_result:
        print("\n=== PairwiseMetric Results ===")
        # print("===========================================================================")
        # print(pairwise_result)
        # print("===========================================================================")
        pairwise_summary = pairwise_result.summary_metrics
        print("Summary metrics:", pairwise_summary)
        
        # Extract win rates if available
        for key, value in pairwise_summary.items():
            if 'win_rate' in key:
                print(f"- {key}: {value}")
    
    # Overall Assessment
    print("\n=== Overall Assessment ===")
    if pointwise_accuracy:
        if pointwise_accuracy >= 0.8:
            print("✓ Judge model shows HIGH accuracy (≥80%)")
        elif pointwise_accuracy >= 0.6:
            print("⚠ Judge model shows MODERATE accuracy (60-79%)")
        else:
            print("✗ Judge model shows LOW accuracy (<60%)")
    
    print("\n=== Recommendations ===")
    if pointwise_accuracy and pointwise_accuracy < 0.7:
        print("- Consider refining the metric prompt template for better clarity")
        print("- Add more examples in the system instruction")
        print("- Increase sampling_count for more stable results")
        print("- Review ground truth labels for consistency")
    else:
        print("- Judge model performance appears acceptable")
        print("- Consider expanding the test dataset for more comprehensive evaluation")

# def create_sample_config_file():
#     """Create a sample configuration file"""
#     config = {
#         "json_key_path": "C:\\Users\\Sunil\\Downloads\\GCP_creds.json",
#         "project_id": "gcp-agents",
#         "location": "us-central1"
#     }
    
#     try:
#         with open("gcp_config.json", 'w') as f:
#             json.dump(config, f, indent=4)
#         print("✓ Sample configuration file 'gcp_config.json' created")
#         print("Please update the values in the file before running the evaluation")
#     except Exception as e:
#         print(f"✗ Error creating config file: {e}")

def load_config_from_file(config_file="gcp_config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found.")
        return None

def main():
    """Main function to run the complete workflow"""
    print("GCP Judge Model Evaluation Setup - COMPLETE FIXED VERSION")
    print("="*60)
    JSON_KEY_PATH = r"C:\Users\Sunil\Downloads\GCP_creds.json"  
    PROJECT_ID = "gcp-agents"                      
    LOCATION = "us-central1"                                 
    print(f"Project ID: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"JSON Key Path: {JSON_KEY_PATH}")
    if not setup_gcp_authentication(JSON_KEY_PATH, PROJECT_ID, LOCATION):
        print("Authentication failed. Please check your JSON key file and project ID.")
        return
    if not test_connection():
        print("Connection test failed. Please check your credentials and project settings.")
        return
    print("\n" + "="*60)
    print("STARTING EVALUATIONS")
    print("="*60)
    pointwise_result, pointwise_accuracy = run_judge_model_evaluation()
    pairwise_result = run_pairwise_evaluation_fixed()
    baseline_result, candidate_result = run_pairwise_evaluation_alternative()
    analyze_results(pointwise_result, pairwise_result, pointwise_accuracy)
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
def main_with_config():
    """Alternative main function using configuration file"""
    config = load_config_from_file()
    if not config:
        print("Creating sample configuration file...")
        # create_sample_config_file()
        return
    print("GCP Judge Model Evaluation Setup - Using Config File")
    print("="*60)
    JSON_KEY_PATH = config.get("json_key_path")
    PROJECT_ID = config.get("project_id")
    LOCATION = config.get("location", "us-central1")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"JSON Key Path: {JSON_KEY_PATH}")
    if not setup_gcp_authentication(JSON_KEY_PATH, PROJECT_ID, LOCATION):
        print("Authentication failed. Please check your JSON key file and project ID.")
        return
    if not test_connection():
        print("Connection test failed. Please check your credentials and project settings.")
        return
    print("\n" + "="*60)
    print("STARTING EVALUATIONS")
    print("="*60)
    pointwise_result, pointwise_accuracy = run_judge_model_evaluation()
    pairwise_result = run_pairwise_evaluation_fixed()
    baseline_result, candidate_result = run_pairwise_evaluation_alternative()
    analyze_results(pointwise_result, pairwise_result, pointwise_accuracy)
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
if __name__ == "__main__":
    # Option 1: Direct configuration (update variables in main())
    main()
    
    # Option 2: Using configuration file (uncomment to use)
    # main_with_config()
    
    # print("\n" + "="*60)
    # print("ALL FIXES IMPLEMENTED:")
    # print("="*60)
    # print("1. ✓ Fixed judge accuracy calculation by correcting ground truth column naming")
    # print("2. ✓ Added manual accuracy calculation as backup method")
    # print("3. ✓ Fixed PairwiseMetric by removing flip_enabled=True")
    # print("4. ✓ Added multiple fallback approaches for PairwiseMetric")
    # print("5. ✓ Added alternative evaluation using separate PointwiseMetrics")
    # print("6. ✓ Added comprehensive debugging and result analysis")
    # print("7. ✓ Improved error handling throughout")
    # print("8. ✓ Added configuration file support")
    # print("9. ✓ Added detailed metrics table inspection")
    # print("10. ✓ Added comprehensive results analysis and recommendations")
    
    # print("\n" + "="*60)
    # print("USAGE INSTRUCTIONS:")
    # print("="*60)
    # print("1. Update the configuration values in main() OR")
    # print("2. Update the generated gcp_config.json file")
    # print("3. Ensure your JSON key file has the required permissions:")
    # print("   - Vertex AI User")
    # print("   - AI Platform Admin (or AI Platform User)")
    # print("4. Enable required APIs in your GCP project:")
    # print("   - Vertex AI API")
    # print("   - AI Platform API")
    # print("5. Run the script: python llm_eval.py")
    
    # print("\n" + "="*60)
    # print("KEY CHANGES FOR JUDGE ACCURACY:")
    # print("="*60)
    # print("- Ground truth column must match metric name exactly:")
    # print("  ✓ Correct: 'response_quality': [1, 0, 1, 0, 1]")
    # print("  ✗ Wrong:   'response_quality/human_rating': [1, 0, 1, 0, 1]")
    # print("- Added manual accuracy calculation as backup")
    # print("- Added comprehensive debugging for metrics tables")
    # print("- Improved prompt templates for clearer judge instructions")