import json, math, random, os
from collections import defaultdict

def _extract_value(field_obj):
    for key in ("valueString","valueInteger","valueNumber","valueBoolean"):
        if key in field_obj:
            return field_obj[key]
    return None

def extract_segments(analysis, segment_field="Segments"):
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

def smart_highlight_filter(segments, k_start=4, k_end=4,
                          event_field="PlayEvent", score_field="HighlightScore",
                          rare_event_weight=1.5, min_per_event=1, n_quantile=0.4):
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

def process_json(input_path, output_path=None):
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
    
    return output_path, segments, clean_segments

# Run the parser if executed directly
if __name__ == "__main__":
    input_path = r"C:\Users\t-kjindel\OneDrive - Microsoft\Desktop\Highlights Generation\analysis_result_videoplayback (3).json"
    output_path = r"C:\Users\t-kjindel\OneDrive - Microsoft\Desktop\Highlights Generation\prefiltered_segments_NBA_USA_France.json"
    process_json(input_path, output_path)
