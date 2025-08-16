import os
import json
import time
import sys
from pathlib import Path
from azure.identity import AzureCliCredential, InteractiveBrowserCredential

# Add the parent directory to the path to import content_understanding_client
sys.path.append(os.path.join(os.path.dirname(__file__)))
from python.content_understanding_client import AzureContentUnderstandingClient

# --- 1. CONFIGURATION ---
# These values should be set by the calling notebook or script
# DO NOT hardcode personal information or file paths here

def get_azure_ai_endpoint():
    """Get Azure AI endpoint from environment or return None for user to set"""
    return os.getenv("AZURE_AI_ENDPOINT", None)

# Default API version - can be overridden
DEFAULT_API_VERSION = "2024-12-01-preview"

# --- 2. HELPER FUNCTIONS ---

def get_access_token():
    """
    Acquires an access token for Azure AI Services.
    It first tries to use credentials from a logged-in Azure CLI.
    If that fails, it falls back to an interactive browser login.
    """
    try:
        # This is the preferred method for local development. It uses the credentials
        # from the 'az login' command you already ran.
        print("Attempting to authenticate using Azure CLI credentials...")
        credential = AzureCliCredential()
        token_response = credential.get_token("https://cognitiveservices.azure.com/.default")
        print("Successfully authenticated using Azure CLI.")
        return token_response.token
    except Exception:
        # This fallback is for when the script can't find the Azure CLI path.
        # It will open a browser window for you to log in.
        print("\nCould not find Azure CLI credentials. Falling back to interactive browser login.")
        print("A browser window may open for you to sign in.")
        try:
            credential = InteractiveBrowserCredential()
            token_response = credential.get_token("https://cognitiveservices.azure.com/.default")
            print("Successfully authenticated using interactive browser.")
            return token_response.token
        except Exception as e:
            print("ERROR: Both Azure CLI and interactive browser authentication failed.")
            print(f"Details: {e}")
            return None

def create_content_understanding_client(azure_ai_endpoint, access_token, api_version=None):
    """
    Creates an instance of AzureContentUnderstandingClient.
    First tries token authentication, then falls back to subscription key if available.
    """
    if not api_version:
        api_version = DEFAULT_API_VERSION
    
    # Try with token provider first
    def token_provider():
        return access_token
    
    try:
        client = AzureContentUnderstandingClient(
            endpoint=azure_ai_endpoint,
            api_version=api_version,
            token_provider=token_provider
        )
        return client
    except Exception as e:
        # If token authentication fails, try with subscription key from environment
        subscription_key = os.getenv("AZURE_AI_SERVICE_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        if subscription_key:
            print(f"Token authentication failed, trying with subscription key...")
            try:
                client = AzureContentUnderstandingClient(
                    endpoint=azure_ai_endpoint,
                    api_version=api_version,
                    subscription_key=subscription_key
                )
                return client
            except Exception as e2:
                print(f"Both token and subscription key authentication failed: {e2}")
                raise e2
        else:
            print(f"Token authentication failed and no subscription key found: {e}")
            raise e

# --- 3. MAIN EXECUTION ---
def run_analysis(video_path, schema_path, azure_ai_endpoint=None, output_dir=None, analyzer_id=None, api_version=None):
    """
    Run the full video analysis workflow using AzureContentUnderstandingClient.
    
    Args:
        video_path: Path to the video file to analyze
        schema_path: Path to the schema file
        azure_ai_endpoint: Azure AI endpoint URL (if None, will try to get from env)
        output_dir: Directory to save output (defaults to same dir as video)
        analyzer_id: Optional custom analyzer ID
        api_version: API version to use
    
    Returns:
        Tuple of (success, output_path, error_message)
    """
    # Validate inputs
    if not azure_ai_endpoint:
        azure_ai_endpoint = get_azure_ai_endpoint()
        if not azure_ai_endpoint:
            return False, None, "Azure AI endpoint not provided. Please set AZURE_AI_ENDPOINT environment variable or pass as parameter."
    
    # Set output directory if not provided
    if not output_dir:
        output_dir = os.path.dirname(os.path.abspath(video_path))
        
    # Set analyzer ID if not provided
    if not analyzer_id:
        analyzer_id = f"auto-highlight-analyzer-{int(time.time())}"
        
    # Set API version if not provided
    if not api_version:
        api_version = DEFAULT_API_VERSION
        
    print(f"--- Starting Video Analysis: {os.path.basename(video_path)} ---")
    print(f"Using analyzer ID: {analyzer_id}")

    # Get the access token
    access_token = get_access_token()
    if not access_token:
        return False, None, "Authentication failure"

    try:
        # Create the Content Understanding client
        client = create_content_understanding_client(azure_ai_endpoint, access_token, api_version)
        
        # Create/update the analyzer with schema
        print(f"\nAttempting to create/update analyzer '{analyzer_id}'...")
        try:
            response = client.begin_create_analyzer(
                analyzer_id=analyzer_id,
                analyzer_template_path=schema_path
            )
            print(f"Successfully created or updated analyzer '{analyzer_id}'.")
        except Exception as e:
            if "409" in str(e) or "Conflict" in str(e):
                print(f"Analyzer '{analyzer_id}' already exists. Continuing with existing analyzer.")
            else:
                raise e
        
        # Start analysis job
        print(f"\nStarting analysis for video file: '{video_path}'...")
        if not os.path.exists(video_path):
            return False, None, f"Video file not found at '{video_path}'."
            
        analyze_response = client.begin_analyze(analyzer_id, video_path)
        print("Analysis job started. Polling for results...")
        
        # Wait for results with polling
        final_result = client.poll_result(
            response=analyze_response,
            timeout_seconds=3600,  # 1 hour timeout
            polling_interval_seconds=30
        )
        
        if not final_result:
            return False, None, "Analysis failed or timed out"
            
        # Create output file path
        base_name = os.path.basename(video_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_filename = f"analysis_result_{file_name_without_ext}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=4)
        
        print(f"The final structured output has been saved to: {output_path}")
        print("Video analysis complete. Results saved to:", output_path)
        
        return True, output_path, None
        
    except Exception as e:
        error_message = f"Analysis failed with error: {str(e)}"
        print(f"ERROR: {error_message}")
        return False, None, error_message

def main():
    """
    Main function for command-line usage.
    This is NOT used when called from the notebook.
    """
    print("ERROR: This script should be called from the highlight generation notebook.")
    print("Please run highlights_notebook.ipynb instead.")
    print("\nIf you want to use this script directly, you need to:")
    print("1. Set AZURE_AI_ENDPOINT environment variable")
    print("2. Provide video_path and schema_path as arguments")
    print("3. Ensure you're authenticated with Azure CLI (az login)")

if __name__ == "__main__":
    main()
