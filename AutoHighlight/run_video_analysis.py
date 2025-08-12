import os
import json
import time
import requests
from azure.identity import AzureCliCredential, InteractiveBrowserCredential

# --- 1. CONFIGURATION ---  
# This section is now pre-filled with the details you provided.
AZURE_AI_ENDPOINT = "https://ai-aistudiotestcuwestus203841201294.openai.azure.com/" 

# Define a unique name for our custom analyzer.
ANALYZER_ID = "soccer-highlights-analyzer531416788"+ str(int(time.time())) 

# Using raw strings for Windows paths to avoid issues with backslashes.
SCHEMA_FILE_PATH = r"C:\Users\t-kjindel\OneDrive - Microsoft\Desktop\Highlights Generation\video_analysis_schema.json"
VIDEO_FILE_PATH = r"C:\Users\t-kjindel\Downloads\videoplayback (3).mp4"

# The API version for Content Understanding. This should match the documentation.
API_VERSION = "2025-05-01-preview"

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

def create_or_update_analyzer(schema_path: str, access_token: str):
    """
    Creates or updates a custom analyzer in Azure AI Content Understanding.
    This function reads our JSON schema and sends it to Azure to define
    what information we want to extract.
    """
    print(f"\nAttempting to create/update analyzer '{ANALYZER_ID}'...")

    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Schema file not found at '{schema_path}'.")
        return False
    except json.JSONDecodeError:
        print(f"ERROR: Schema file at '{schema_path}' is not valid JSON.")
        return False

    analyzer_url = f"{AZURE_AI_ENDPOINT}/contentunderstanding/analyzers/{ANALYZER_ID}?api-version={API_VERSION}"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.put(analyzer_url, headers=headers, json=schema_data)
        
        # A 201 means it was newly created. A 200 means it was updated.
        if response.status_code in [200, 201]:
            print(f"Successfully created or updated analyzer '{ANALYZER_ID}'.")
            return True
        # A 409 means the analyzer with this name already exists. This is OK for us.
        elif response.status_code == 409:
            print(f"Analyzer '{ANALYZER_ID}' already exists. Continuing.")
            return True
        else:
            print(f"ERROR: Failed to create/update analyzer. Status Code: {response.status_code}")
            print(f"Response Body: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"ERROR: A network error occurred: {e}")
        return False

def analyze_local_video(video_path: str, access_token: str):
    """
    Uploads a local video file and starts the analysis job.
    Returns the URL to poll for results.
    """
    print(f"\nStarting analysis for local video file: '{video_path}'...")
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at '{video_path}'.")
        return None

    analyze_url = f"{AZURE_AI_ENDPOINT}/contentunderstanding/analyzers/{ANALYZER_ID}:analyze?api-version={API_VERSION}"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/octet-stream"
    }

    try:
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()
        response = requests.post(analyze_url, headers=headers, data=video_data)
        if response.status_code == 202:
            result_url = response.headers.get("Operation-Location")
            request_id = response.headers.get("x-ms-request-id")
            print(f"Analysis job started. You can check the results at: {result_url}")
            print(f"DEBUG: Request ID for backend tracing: {request_id}")
            return result_url
        else:
            print(f"ERROR: Failed to start analysis. Status Code: {response.status_code}")
            print(f"Response Body: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"ERROR: A network error occurred: {e}")
        return None

def poll_for_results(result_url: str, access_token: str):
    """
    Continuously checks the result URL until the analysis job is completed.
    Uses a single access token for the duration of the polling.
    """
    print("\nPolling for analysis results. This may take minutes...")
    
    headers = {"Authorization": f"Bearer {access_token}"}
        
    while True:
        try:
            response = requests.get(result_url, headers=headers)
            if response.status_code == 200:
                result_json = response.json()
                status = result_json.get("status")
                print(f"Current analysis status: {status}")
                if status == "Succeeded":
                    print("Analysis completed successfully!")
                    return result_json
                elif status == "Failed":
                    print("ERROR: Analysis failed.")
                    print(f"Details: {result_json}")
                    return None
            else:
                print(f"ERROR: Failed to get status. Code: {response.status_code}")
                return None
            
            # Wait for 30 seconds before polling again.
            time.sleep(30)
            
        except requests.exceptions.RequestException as e:
            print(f"ERROR: A network error occurred while polling: {e}")
            return None

# --- 3. MAIN EXECUTION ---
def run_analysis(video_path, schema_path, output_dir=None, analyzer_id=None):
    """
    Run the full video analysis workflow with specified parameters.
    
    Args:
        video_path: Path to the video file to analyze
        schema_path: Path to the schema file
        output_dir: Directory to save output (defaults to same dir as video)
        analyzer_id: Optional custom analyzer ID
    
    Returns:
        Tuple of (success, output_path, error_message)
    """
    # Set output directory if not provided
    if not output_dir:
        output_dir = os.path.dirname(os.path.abspath(video_path))
        
    # Set analyzer ID if not provided
    if not analyzer_id:
        analyzer_id = f"auto-highlight-analyzer-{int(time.time())}"
        
    print(f"--- Starting Video Analysis: {os.path.basename(video_path)} ---")
    print(f"Using analyzer ID: {analyzer_id}")

    # Get the access token
    access_token = get_access_token()
    if not access_token:
        return False, None, "Authentication failure"

    # Create/update the analyzer with schema
    if not create_or_update_analyzer(schema_path, access_token):
        return False, None, "Analyzer creation failure"
        
    # Start analysis job
    result_url = analyze_local_video(video_path, access_token)
    if not result_url:
        return False, None, "Analysis job start failure"
    
    # Wait for results
    final_result = poll_for_results(result_url, access_token)
    
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
    
    print("\n--- Analysis Complete ---")
    print(f"The final structured output has been saved to: {output_path}")
    
    return True, output_path, None

def main():
    """Main function to orchestrate the video analysis workflow."""
    print("--- Starting Soccer Highlight Generation Demo (Entra ID Auth) ---")
    
    success, output_path, error = run_analysis(VIDEO_FILE_PATH, SCHEMA_FILE_PATH)
    
    if not success:
        print(f"Analysis failed: {error}")

if __name__ == "__main__":
    main()
