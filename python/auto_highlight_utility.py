import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .utility import generate_schema_llm, add_property_llm, OpenAIAssistant


EXAMPLE_SCHEMA_TYPE = "soccer"  # Default example schema type if none found

OUTPUT_DIR = Path(__file__).parents[3] / "output"


# --- analyzer generation ---
def human_review(schema_obj: dict, openai_assistant: Optional[OpenAIAssistant]) -> bool:
    """Interactive schema review. If auto_mode is True, uses auto_review instead."""  
    print("\n[human_review] Raw LLM output:")
    print(json.dumps(schema_obj, indent=2))

    if not schema_obj or not isinstance(schema_obj, dict):
        print("[human_review] Input is not a valid dict. Aborting review.")
        return False

    while True:
        props = schema_obj.get("fieldSchema", {}).get("fields", {}).get("Segments", {})
        items = props.get("items", {}).get("properties", {})
        approved = {}
        print("\nReview each schema property for 'Segments':")
        for key, val in items.items():
            print(f"\nProperty: {key}\nType: {val.get('type','')}\nDescription: {val.get('description','')}")
            ans = input("Include this property? (yes/no): ").strip().lower()
            if ans == 'yes':
                approved[key] = val

        schema_obj['fieldSchema']['fields']['Segments']['items']['properties'] = approved

        if openai_assistant:
            # Offer to add a new property
            add_more = input("\nWould you like to add a new property to this schema? (yes/no): ").strip().lower()
            if add_more == 'yes':
                user_desc = input("Describe the property you want to add (e.g., name, type, description): ").strip()
                schema_obj_str = json.dumps(schema_obj, indent=2)
                updated_schema = add_property_llm(openai_assistant, schema_obj_str, user_desc)
                
                if not isinstance(updated_schema, dict):
                    print("[human_review] LLM did not return a valid dict. Aborting.")
                    return schema_obj
                schema_obj = updated_schema
                print("[human_review] Property added! Restarting review...")
                continue
            else:
                return schema_obj
        else:
            print("\nNo OpenAIAssistant provided, skipping property addition.")
            return schema_obj


def activate_schema(
        video_type: str,
        analyzer_dir: Union[str, Path],
        openai_assistant: Optional[OpenAIAssistant] = None,
        output_dir: Union[str, Path] = OUTPUT_DIR,
        clip_density: float = 1.0,
        target_duration_s: int = 100, 
        personalization: str = "none",
        human_in_the_loop_review: bool = True):
    analyzer_dir = Path(analyzer_dir)
    output_dir = Path(output_dir)

    schema_path = analyzer_dir / f"{video_type}.json"

    if schema_path.exists():
        print(f"Schema for '{video_type}' already exists at '{schema_path}'. Using existing schema.")
    elif not analyzer_dir.exists():
        raise ValueError(f"Analyzer folder '{analyzer_dir}' does not exist.")
    elif openai_assistant is None:
        raise ValueError("OpenAIAssistant instance must be provided to generate new schemas.")
    else:
        example_path = analyzer_dir / f"{EXAMPLE_SCHEMA_TYPE}.json"
        if not example_path.exists():
            raise ValueError(f"No schema found for '{video_type}' and default example schema '{example_path}' does not exist.")
        print(f"Using example schema: {example_path} and LLM to generate schema for '{video_type}'.")

        # Generate the initial schema text
        generated_schema = generate_schema_llm(openai_assistant, video_type, str(example_path), clip_density, target_duration_s, personalization)

        # Warn if description does not mention requested video_type
        desc = generated_schema.get("description", "").lower()
        if video_type.lower() not in desc:
            print(f"Warning: Generated schema description does not mention '{video_type}'. Description: {desc}")

        if human_in_the_loop_review:
            generated_schema = human_review(generated_schema, openai_assistant)

        # Save the final schema
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(generated_schema, f, indent=2)
        print(f"Schema for '{video_type}' saved to '{schema_path}'.")
    
    return schema_path


# --- result json parsing ---
def _extract_value(field_obj: Dict[str, Any]) -> Any:
    """
    Extracts the value from a field object, checking for known value types.

    Args:
        field_obj (dict): The field object containing possible value types.

    Returns:
        Any: The extracted value, or None if not found.
    """
    for key in ("valueString","valueInteger","valueNumber","valueBoolean"):
        if key in field_obj:
            return field_obj[key]
    return None

def extract_segments(
    analysis: Dict[str, Any],
    segment_field: str = "Segments"
) -> List[Dict[str, Any]]:
    """
    Extracts segment dictionaries from the analysis JSON.

    Args:
        analysis (dict): The analysis JSON object.
        segment_field (str, optional): The field name for segments. Defaults to "Segments".

    Returns:
        list: List of segment dictionaries.
    """
    out = []
    for content in analysis["result"]["contents"]:
        fields = content.get("fields", {})
        if segment_field not in fields:
            continue
        for seg_wrapper in fields[segment_field]["valueArray"]:
            flat = {k: _extract_value(v) for k, v in seg_wrapper["valueObject"].items()}
            out.append(flat)
    out.sort(key=lambda s: s.get("StartTimeMs", 0))
    return out

def smart_highlight_filter(
    segments: List[Dict[str, Any]],
    k_start: int = 4,
    k_end: int = 4,
    event_field: str = "PlayEvent",
    score_field: str = "HighlightScore",
    rare_event_weight: float = 1.5,
    min_per_event: int = 1,
    n_quantile: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Filters segments to select highlights based on scores, event rarity, and coverage.

    Args:
        segments (list): List of segment dictionaries.
        k_start (int, optional): Number of segments to always keep from the start. Defaults to 4.
        k_end (int, optional): Number of segments to always keep from the end. Defaults to 4.
        event_field (str, optional): Field name for event type. Defaults to "PlayEvent".
        score_field (str, optional): Field name for highlight score. Defaults to "HighlightScore".
        rare_event_weight (float, optional): Weight multiplier for rare events. Defaults to 1.5.
        min_per_event (int, optional): Minimum per event type. Defaults to 1.
        n_quantile (float, optional): Fraction of core candidates to keep. Defaults to 0.4.

    Returns:
        list: Filtered list of segment dictionaries.
    """
    n = len(segments)
    if n == 0:
        return []
    first_k = segments[:k_start]
    last_k = segments[-k_end:] if k_end else []
    core_candidates = segments[k_start:n-k_end] if n > (k_start + k_end) else []
    # Count event frequencies
    event_counts = defaultdict(int)
    for s in core_candidates:
        event_counts[s.get(event_field)] += 1
    # Score adjustment for rare events
    scored = []
    for s in core_candidates:
        score = s.get(score_field, 0.0)
        event = s.get(event_field)
        if event_counts[event] == 1:
            score *= rare_event_weight
        scored.append((score, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    core_n = max(1, int(len(core_candidates) * n_quantile))
    core = [s for _, s in scored[:core_n]]
    # Ensure all event types are present
    present = {s.get(event_field) for s in core}
    all_events = {s.get(event_field) for s in core_candidates}
    missing = all_events - present
    for ev in missing:
        fallback = max((s for s in core_candidates if s.get(event_field) == ev),
                       key=lambda s: s.get(score_field, 0.0), default=None)
        if fallback and fallback not in core:
            core.append(fallback)
    # Ensure at least 40% of all segments are included
    min_total = max(1, int(n * 0.4))
    combined = first_k + core + last_k
    seen = set()
    unique = []
    for s in combined:
        sid = s.get("SegmentId")
        if sid not in seen:
            unique.append(s)
            seen.add(sid)
    if len(unique) < min_total:
        # Add more from the remaining segments by score
        remaining = [s for s in segments if s.get("SegmentId") not in seen]
        remaining.sort(key=lambda s: s.get(score_field, 0.0), reverse=True)
        unique += remaining[:min_total - len(unique)]
    unique.sort(key=lambda s: s.get("StartTimeMs", 0))
    return unique

def get_filtered_segments(
    input_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Loads segments from a JSON file, filters them, and writes the filtered segments to a new file.

    Args:
        input_path (str): Path to the input JSON file.
        output_path (str, optional): Path to write the filtered segments. If None, a default is used.

    Returns:
        str: Path to the output file containing filtered segments.
    """
    if output_path is None:
        # Generate default output path based on input filename
        input_dir = os.path.dirname(input_path)
        input_filename = os.path.basename(input_path)
        name_without_ext = os.path.splitext(input_filename)[0]
        output_path = os.path.join(input_dir, f"prefiltered_segments_{name_without_ext}.json")
    
    with open(input_path, "r", encoding="utf-8") as f:
        analysis_json = json.load(f)
    
    segments = extract_segments(analysis_json)
    clean_segments = smart_highlight_filter(segments)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_segments, f, indent=2)
    
    print(f"Parsed {len(segments)} segments -> kept {len(clean_segments)}")
    if len(segments) > 0:
        percent = 100 * len(clean_segments) / len(segments)
        print(f"Kept {percent:.1f}% of segments")
    else:
        print("No segments found.")
    
    return output_path
