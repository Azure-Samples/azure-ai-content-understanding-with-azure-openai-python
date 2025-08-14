import os
import sys
import json
import shutil
import time
from pathlib import Path
from openai import AzureOpenAI, APIError, RateLimitError


# Configuration
BASE_DIR      = Path(__file__).parent.parent  # Go up one level from Helper folder
CONFIG_PATH   = BASE_DIR / "SchemaConfig.json"
ACTIVE_SCHEMA = BASE_DIR / "VideoAnalysisSchema.json"
SCHEMAS_DIR   = BASE_DIR / "schemas"

# Azure OpenAI settings (for schema generation)
# These will be loaded from environment variables set by the notebook
ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = "2024-12-01-preview"
DEPLOYMENT  = "o1"

# Placeholders
DEFAULT_VIDEO_TYPE = "table tennis"
DEFAULT_CLIP_DENSITY = 1.0  # clips per minute
DEFAULT_PERSONALIZATION = "none"  
DEFAULT_TARGET_DURATION_S = 300  # seconds




client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version=API_VERSION
)

def safe_chat(messages, max_tokens=4096):
    backoff = 2
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=messages,
                max_completion_tokens=max_tokens  # Azure OpenAI expects this
            )
            return resp.choices[0].message.content.strip()
        except (APIError, RateLimitError) as e:
            print(f"[safe_chat] Attempt {attempt+1} failed: {e}")
            time.sleep(backoff)
            backoff *= 2
        except Exception as e:
            print(f"[safe_chat] Unexpected error on attempt {attempt+1}: {e}")
            time.sleep(backoff)
            backoff *= 2
    sys.exit("LLM call failed after retries")

def load_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps({}), encoding="utf-8")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def generate_schema_llm(video_type: str, example_path: Path, clip_density: float = None, target_duration_s: int = None, personalization: str = None) -> str:
    example = example_path.read_text(encoding="utf-8")
    clip_density_val = clip_density if clip_density is not None else DEFAULT_CLIP_DENSITY
    target_duration_val = target_duration_s if target_duration_s is not None else DEFAULT_TARGET_DURATION_S
    personalization_val = personalization if personalization is not None else DEFAULT_PERSONALIZATION
    extra = (
        f"\n- Target clip density: {clip_density_val} clips per minute."
        f"\n- Target total duration: {target_duration_val} seconds."
        f"\n- Personalization: {personalization_val}"
    )
    prompt = (
        "You are an expert in Azure Content Understanding custom analyzer schemas.\n"
        "Here is an example schema for soccer:\n\n"
        f"{example}\n\n"
        f"Generate a complete, valid analyzer schema JSON for '{video_type}' videos, "
        "matching the same format and level of detail."
        f"{extra}"
    )
    system_msg = {"role": "system", "content": "Generate Azure Content Understanding custom analyzer schemas."}
    user_msg   = {"role": "user",   "content": prompt}
    return safe_chat([system_msg, user_msg])

def rule_based_schema_validation(schema_obj):
    """
    Fast, deterministic check for required schema structure.
    """
    required_top = ["description", "baseAnalyzerId", "config", "fieldSchema"]
    for key in required_top:
        if key not in schema_obj:
            print(f"Missing top-level key: {key}")
            return False
    segments = schema_obj.get("fieldSchema", {}).get("fields", {}).get("Segments", {})
    items = segments.get("items", {})
    properties = items.get("properties", {})
    if not isinstance(properties, dict) or not properties:
        print("No properties found in Segments.items.properties")
        return False
    for prop, val in properties.items():
        if "type" not in val or "description" not in val:
            print(f"Property {prop} missing type or description")
            return False
    return True



def activate_schema(video_type: str, clip_density: float = None, target_duration_s: int = None, 
                  personalization: str = None, auto_mode: bool = False, human_in_the_loop_review: bool = True):
    """Main workflow to generate, review, and activate a schema.
    
    Args:
        video_type: Type of video (e.g., "soccer", "basketball")
        clip_density: Clips per minute (optional)
        target_duration_s: Target highlight duration in seconds (optional)
        personalization: Any specific personalization instructions (optional)
        auto_mode: If True, use automatic review without user interaction (for GUI apps)
        human_in_the_loop_review: If True, enable human review (maps to auto_mode inversely)
    """
    # Map human_in_the_loop_review to auto_mode (they are opposites)
    if not auto_mode:  # Only override auto_mode if it wasn't explicitly set
        auto_mode = not human_in_the_loop_review
    cfg = load_config()
    schema_path = SCHEMAS_DIR / f"{video_type}.json"

    # Load defaults from config if not provided
    video_cfg = cfg.get(video_type, {}) if isinstance(cfg.get(video_type, {}), dict) else {}
    if clip_density is None:
        clip_density = video_cfg.get("clip_density")
    if target_duration_s is None:
        target_duration_s = video_cfg.get("target_duration_s")
    if personalization is None:
        personalization = video_cfg.get("personalization", DEFAULT_PERSONALIZATION)

    if schema_path.exists():
        src = schema_path
    elif video_type in cfg and isinstance(cfg[video_type], str) and Path(cfg[video_type]).exists():
        src = Path(cfg[video_type])
    else:
        SCHEMAS_DIR.mkdir(exist_ok=True)
        # Try to use example schema for requested video_type, fallback to soccer
        example_path = BASE_DIR / "schemas" / f"{video_type}.json"
        if not example_path.exists():
            print(f"Example schema for '{video_type}' not found, falling back to soccer.json.")
            example_path = BASE_DIR / "schemas" / "soccer.json"
        if not example_path.exists():
            sys.exit("No example schema found for LLM prompt.")
        print(f"Using example schema: {example_path}")

        # 1. Generate the initial schema text
        initial_schema_text = generate_schema_llm(video_type, example_path, clip_density, target_duration_s, personalization)

        # 2. Rule-based validation instead of LLM evaluation
        try:
            schema_obj = json.loads(initial_schema_text)
        except Exception as e:
            print(f"Schema is not valid JSON: {e}")
            sys.exit(1)

        # Warn if description does not mention requested video_type
        desc = schema_obj.get("description", "").lower()
        if video_type.lower() not in desc:
            print(f"Warning: Generated schema description does not mention '{video_type}'. Description: {desc}")

        if rule_based_schema_validation(schema_obj):
            print("Rule-based validation passed.")
            text_for_review = initial_schema_text
        else:
            print("Rule-based validation failed. Proceeding to human review anyway.")
            text_for_review = initial_schema_text

        # 3. Call review function with the correct schema text
        temp_path = BASE_DIR / "temp_review_schema.json"
        review_successful = human_review(text_for_review, temp_path, auto_mode=auto_mode)

        if not review_successful:
            if auto_mode:
                # For GUI apps, raise exception instead of exiting
                raise ValueError("Schema review failed. No schema was activated.")
            else:
                sys.exit("Human review was aborted. No schema was activated.")

        # 4. Use the human-approved schema from the temp file
        src = temp_path

        # 5. Save the final version to the permanent location
        dest = SCHEMAS_DIR / f"{video_type}.json"
        shutil.copy(src, dest)

        # Store config as a dict for this video_type
        cfg[video_type] = cfg.get(video_type, {}) if isinstance(cfg.get(video_type, {}), dict) else {}
        if clip_density is not None:
            cfg[video_type]["clip_density"] = clip_density
        if target_duration_s is not None:
            cfg[video_type]["target_duration_s"] = target_duration_s
        if personalization is not None:
            cfg[video_type]["personalization"] = personalization
        cfg[video_type]["schema_path"] = str(dest)
        save_config(cfg)

        # Set src to dest for final copy, then clean up temp file
        src = dest
        if temp_path.exists():
            temp_path.unlink()

    shutil.copy(src, ACTIVE_SCHEMA)
    # Always overwrite ACTIVE_SCHEMA with the latest schema for the requested type
    try:
        shutil.copy(src, ACTIVE_SCHEMA)
        print(f"\nActivated schema for '{video_type}' -> {ACTIVE_SCHEMA}")
        # Log the first few lines of the schema for confirmation
        with open(ACTIVE_SCHEMA, 'r', encoding='utf-8') as f:
            preview = ''.join(f.readlines()[:10])
        print(f"Schema preview (first 10 lines):\n{preview}")
        return str(ACTIVE_SCHEMA)  # Return the path to the activated schema
    except Exception as e:
        print(f"Error updating active schema: {e}")
        sys.exit(1)


def add_property_llm(current_schema: str, property_description: str) -> str:
    """
    Call the LLM to add a new property to the schema based on user description.
    """
    prompt = (
        "You are an expert in JSON schema design. Given the current schema and a user description, "
        "add a new property to the Segments.items.properties section. Return only the updated JSON schema.\n\n"
        f"Current schema:\n{current_schema}\n\n"
        f"User request: {property_description}"
    )
    return safe_chat([
        {"role": "system", "content": "Add a property to the schema."},
        {"role": "user", "content": prompt}
    ])


def auto_review(schema_text: str, temp_path: Path) -> bool:
    """Non-interactive schema review for use in GUI applications"""
    print("\n[auto_review] Processing LLM output...")
    if not schema_text or not schema_text.strip().startswith('{'):
        print("[auto_review] LLM did not return valid JSON. Aborting review.")
        return False
    try:
        obj = json.loads(schema_text)
        # Just save the schema as-is without interactive review
        temp_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print(f"\n[auto_review] Schema written to {temp_path}")
        return True
    except json.JSONDecodeError as e:
        print(f"[auto_review] JSONDecodeError: {e}")
        return False

def human_review(schema_text: str, temp_path: Path, auto_mode: bool = False) -> bool:
    """Interactive schema review. If auto_mode is True, uses auto_review instead."""
    if auto_mode:
        return auto_review(schema_text, temp_path)
    
    print("\n[human_review] Raw LLM output:")
    print(schema_text)
    if not schema_text or not schema_text.strip().startswith('{'):
        print("[human_review] LLM did not return valid JSON. Aborting review.")
        return False
    try:
        obj = json.loads(schema_text)
    except json.JSONDecodeError as e:
        print(f"[human_review] JSONDecodeError: {e}")
        return False

    while True:
        props = obj.get("fieldSchema", {}).get("fields", {}).get("Segments", {})
        items = props.get("items", {}).get("properties", {})
        approved = {}
        print("\nReview each schema property for 'Segments':")
        for key, val in items.items():
            print(f"\nProperty: {key}\nType: {val.get('type','')}\nDescription: {val.get('description','')}")
            ans = input("Include this property? (yes/no): ").strip().lower()
            if ans == 'yes':
                approved[key] = val
        obj['fieldSchema']['fields']['Segments']['items']['properties'] = approved
        # Offer to add a new property
        add_more = input("\nWould you like to add a new property to this schema? (yes/no): ").strip().lower()
        if add_more == 'yes':
            user_desc = input("Describe the property you want to add (e.g., name, type, description): ").strip()
            obj_str = json.dumps(obj, indent=2)
            updated_schema = add_property_llm(obj_str, user_desc)
            try:
                obj = json.loads(updated_schema)
                print("[human_review] Property added! Restarting review...")
                continue
            except json.JSONDecodeError as e:
                print(f"[human_review] Failed to parse updated schema from LLM: {e}")
                return False
        else:
            # Write final reviewed schema to file
            temp_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
            print(f"\nReviewed schema written to {temp_path}")
            return True

if __name__ == "__main__":
    # This part is for command-line execution and can be adapted or removed
    # For notebook usage, the parameters are passed directly to activate_schema
    video_type = "soccer" # Example value
    clip_density = 1.0
    target_duration_s = 120
    personalization = "exciting moments"
    print(f"Using video_type: '{video_type}'")
    print(f"Clip density: {clip_density}")
    print(f"Target duration (s): {target_duration_s}")
    print(f"Personalization: {personalization}")
    activate_schema(video_type, clip_density=clip_density, target_duration_s=target_duration_s, personalization=personalization)

