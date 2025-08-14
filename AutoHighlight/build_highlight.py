import os, sys, json, math, time, pathlib, datetime, textwrap, re
from typing import List, Dict, Any
from openai import AzureOpenAI, APIError, RateLimitError
from jsonschema import validate, ValidationError
from schema_manager import (
    DEFAULT_VIDEO_TYPE,
    DEFAULT_CLIP_DENSITY,
    DEFAULT_TARGET_DURATION_S,
    DEFAULT_PERSONALIZATION
)

PROMPT_PATH = "reasoning_prompt.txt"
SEGMENTS_PATH = "prefiltered_segments_NBA_USA_France.json"
VIDEO_TYPE = DEFAULT_VIDEO_TYPE
CLIP_DENSITY = DEFAULT_CLIP_DENSITY
RUNTIME_S = DEFAULT_TARGET_DURATION_S
PERSONALIZATION = DEFAULT_PERSONALIZATION
OUT_DIR = "."
EVENT_FIELD = "EventType"

# These will be set by the notebook before calling main()
CONFIG_VIDEO_TYPE = None
CONFIG_CLIP_DENSITY = None
CONFIG_TARGET_DURATION_S = None
CONFIG_PERSONALIZATION = None

# Get credentials from environment variables (set by the notebook)
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = "o1"
API_VERSION = "2024-12-01-preview"
TEMPERATURE = 0.4

# Validate that credentials are available
if not ENDPOINT:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set. Please run the API configuration cell in the notebook first.")
if not API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable not set. Please run the API configuration cell in the notebook first.")

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "SelectedClips": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["SegmentId", "Act", "NarrativeRole", "WhyChosen"],
                "properties": {
                    "SegmentId": {"type": "string"},
                    "Act": {"enum": ["Introduction", "Climax", "Resolution"]},
                    "NarrativeRole": {"type": "string"},
                    "WhyChosen": {"type": "string"}
                }
            }
        },
        "ActSummaries": {
            "type": "object",
            "required": ["Introduction", "Climax", "Resolution"]
        },
        "DidReturnFewerThanRequested": {"type": "boolean"}
    },
    "required": ["SelectedClips", "ActSummaries", "DidReturnFewerThanRequested"]
}

def safe_chat(client: AzureOpenAI, messages, deployment, temperature=None):
    back = 2
    for _ in range(4):
        try:
            # Create parameters for the API call
            params = {
                "model": deployment,
                "messages": messages,
                "max_completion_tokens": 32000,
                "response_format": {"type": "json_object"}  # Request JSON format explicitly
            }
            
            # Only add temperature if it's provided (avoid for o1 model)
            if temperature is not None and deployment != "o1":
                params["temperature"] = temperature
                
            rsp = client.chat.completions.create(**params)
            content = rsp.choices[0].message.content.strip()
            
            # Try to clean up any non-JSON content
            if content.startswith("```json"):
                content = re.sub(r'^```json\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
            elif content.startswith("```"):
                content = re.sub(r'^```\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                
            return content
        except (APIError, RateLimitError) as e:
            print(f"OpenAI API error: {e}")
            time.sleep(back)
            back *= 2
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(back)
            back *= 2
    raise RuntimeError("LLM call failed after retries")

def build_prompt(tpl: str, video_type: str, n: int, segs: list) -> str:
    return (tpl
            .replace("{{video_type}}", video_type)
            .replace("{{N}}", str(n))
            .replace("{{segments_json}}", json.dumps(segs, indent=2)))

def validate_json(txt: str) -> dict:
    # Clean up the text to extract JSON
    json_str = txt.strip()
    
    # Try to find JSON content between triple backticks if it exists
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', json_str)
    if json_match:
        json_str = json_match.group(1).strip()
    
    # If we still don't have valid JSON, try to find content between curly braces
    if not json_str.startswith('{'):
        curly_match = re.search(r'(\{[\s\S]*\})', json_str)
        if curly_match:
            json_str = curly_match.group(1).strip()
    
    if not json_str:
        raise ValueError("Could not find valid JSON content in the model response")
        
    try:
        obj = json.loads(json_str)
        validate(obj, FINAL_SCHEMA)
        return obj
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Attempted to parse: {json_str[:100]}...")  # Print first 100 chars for debugging
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        print(f"Validation error: {e}")
        raise

def main() -> None:
    # Use config values if they were set, otherwise fall back to defaults
    video_type = CONFIG_VIDEO_TYPE if CONFIG_VIDEO_TYPE is not None else VIDEO_TYPE
    clip_density = CONFIG_CLIP_DENSITY if CONFIG_CLIP_DENSITY is not None else CLIP_DENSITY
    target_duration_s = CONFIG_TARGET_DURATION_S if CONFIG_TARGET_DURATION_S is not None else RUNTIME_S
    personalization = CONFIG_PERSONALIZATION if CONFIG_PERSONALIZATION is not None else PERSONALIZATION
    
    print(f"Using configuration:")
    print(f"  Video Type: {video_type}")
    print(f"  Clip Density: {clip_density}")
    print(f"  Target Duration: {target_duration_s}s")
    print(f"  Personalization: {personalization}")
    
    prompt_tpl = pathlib.Path(PROMPT_PATH).read_text(encoding="utf-8")
    segments   = json.loads(pathlib.Path(SEGMENTS_PATH).read_text(encoding="utf-8"))
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
    )
    # System message: full instructions and schema
    filled_prompt = (
        prompt_tpl
        .replace("{{video_type}}", video_type)
        .replace("{{clip_density}}", str(clip_density))
        .replace("{{target_duration_s}}", str(target_duration_s))
        .replace("{{personalization}}", str(personalization))
    )
    # Add an explicit JSON format instruction to the system message
    filled_prompt += "\n\nRESPONSE FORMAT: Return ONLY a valid JSON object. Do not include markdown formatting, code blocks, or any text outside the JSON object."
    system = {"role": "system", "content": filled_prompt}

    # User message: just the input JSON
    input_json = {
        "video_type": video_type,
        "clip_density": clip_density,
        "target_duration_s": target_duration_s,
        "personalization": personalization,
        "Segments": segments
    }
    user_content = json.dumps(input_json, indent=2)

    print("=== USER MESSAGE SENT TO MODEL ===")
    print(user_content)
    debug_prompt_path = pathlib.Path(OUT_DIR) / "debug_full_prompt.txt"
    debug_prompt_path.write_text(user_content, encoding="utf-8")

    # Debug: print and save the system prompt sent to the model
    print("=== SYSTEM MESSAGE SENT TO MODEL ===")
    print(filled_prompt)
    debug_system_path = pathlib.Path(OUT_DIR) / "debug_system_prompt.txt"
    debug_system_path.write_text(filled_prompt, encoding="utf-8")

    user = {"role": "user", "content": user_content}
    draft = safe_chat(client, [system, user], DEPLOYMENT)
    checklist = textwrap.dedent(f"""
    Checklist:
    1. JSON valid & schema compliant?
    2. Chronological order?
    3. Unique SegmentIds?
    4. Each act present?
    5. ≥2 distinct {EVENT_FIELD} values unless fewer exist.
    6. Runtime ≤ {target_duration_s}s if provided.
    Reply PASS or JSON {{"Issues":[],"FixTips":""}}
    Draft:
    ```json
    {draft}
    ```""")
    critique = safe_chat(client, [system, {"role":"user","content":checklist}],
                         DEPLOYMENT)
    if critique.strip() != "PASS":
        revision_prompt = textwrap.dedent(f"""
        Fix every issue below and return corrected JSON only.
        Draft:
        ```json
        {draft}
        ```
        Issues:
        ```json
        {critique}
        ```""")
        final_txt = safe_chat(client, [system, {"role":"user","content":revision_prompt}],
                              DEPLOYMENT)
    else:
        final_txt = draft
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    # Debug: print and save raw model output
    print("RAW MODEL OUTPUT:")
    print(final_txt)
    debug_path = pathlib.Path(OUT_DIR) / f"raw_model_output_{ts}.txt"
    debug_path.write_text(final_txt, encoding="utf-8")
    if not final_txt.strip():
        print("Model returned empty response. Exiting.")
        sys.exit(1)
        
    try:
        final_obj = validate_json(final_txt)
        # Save final output in the same folder as the script
        out_path = pathlib.Path(__file__).parent / "final_highlight_result.json"
    except Exception as e:
        print(f"ERROR: Failed to process model output: {str(e)}")
        print("Attempting fallback JSON extraction...")
        
        # Fallback: Try to create a minimal valid structure
        try:
            # Extract any segment data we can find
            segment_match = re.search(r'"SelectedClips"\s*:\s*(\[[\s\S]*?\])', final_txt)
            if segment_match:
                segments_json = segment_match.group(1)
                try:
                    segments = json.loads(segments_json)
                    # Create a minimal valid structure
                    final_obj = {
                        "SelectedClips": segments,
                        "ActSummaries": {
                            "Introduction": "Auto-generated introduction",
                            "Climax": "Auto-generated climax",
                            "Resolution": "Auto-generated resolution"
                        },
                        "DidReturnFewerThanRequested": False
                    }
                    print("Successfully created fallback JSON structure")
                    out_path = pathlib.Path(__file__).parent / "final_highlight_result.json"
                except Exception as json_err:
                    print(f"Fallback extraction failed: {str(json_err)}")
                    sys.exit(1)
            else:
                print("Could not extract any valid segment data")
                sys.exit(1)
        except Exception as fallback_err:
            print(f"All recovery attempts failed: {str(fallback_err)}")
            sys.exit(1)
    out_path.write_text(json.dumps(final_obj, indent=2), encoding="utf-8")
    print(f"✓ Final highlight plan saved to {out_path}")

if __name__ == "__main__":
    main()
