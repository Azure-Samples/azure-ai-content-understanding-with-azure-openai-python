# 🎥 AI-Powered Highlights Generator

Generate cinematic, story-driven highlight reels from any video using Azure AI Content Understanding and LLM-based narrative reasoning.

This project transforms long-form sports or event footage into compelling summaries, automatically identifying key segments, reasoning about narrative flow, and stitching the final highlight reel — all with minimal user input.

---

## 🚀 What It Does

This pipeline enables:

- 🧠 **Automatic Event Detection** using Azure AI’s Content Understanding service
- ✂️ **Smart Segment Filtering** that prioritizes rare and high-scoring moments
- 📖 **Narrative-Aware Highlight Selection** using a structured LLM planning process
- 🎬 **Video Stitching Engine** with captions, transitions, and optional audio
- ✅ **Evaluation & Feedback Loop** to assess diversity, coherence, and coverage

The result? A single MP4 highlight reel you can share, stream, or customize — generated entirely from raw footage.

---

## 🛠️ How It Works

The highlight generation pipeline consists of **7 modular layers**, each performing a crucial transformation step:

| Layer | Description |
|-------|-------------|
| **1. Schema Upload** | Configures Azure AI to understand domain-specific events via a custom JSON schema. |
| **2. Analyzer Execution** | Authenticates, uploads the video, and triggers analysis via Content Understanding API. |
| **3. Segment Filtering** | Filters segments based on highlight scores, diversity, and coverage heuristics. |
| **4. LLM Reasoning** | Uses GPT-based logic to structure highlights into acts: *Introduction → Climax → Resolution*. |
| **5. Video Stitching** | Concatenates selected clips with effects, transitions, and optional speed ramping. |
| **6. Evaluation Layer** | Evaluates visual/audio quality, engagement, and narrative consistency. |
| **7. Output** | Delivers the final highlight video + logs + optional summary report. |

### 🧩 Full System Architecture

<img width="1897" height="923" alt="image" src="https://github.com/user-attachments/assets/a8382ba9-0f06-43de-9695-460f8760c53a" />

---

## 📂 Inputs & Outputs

This section outlines the key input/output files at each stage of the pipeline.

| 🔢 Layer | 📥 Input(s) | 📤 Output(s) |
|---------|-------------|--------------|
| **Layer 1**<br>Schema Upload & Submission | - `video_analysis_schema.json`<br>- Raw `.mp4` file | - Video submitted to Azure CU |
| **Layer 2**<br>Analyzer Execution | - Analyzer ID<br>- Video reference | - `analysis_result_<video>.json` |
| **Layer 3**<br>Segment Filtering | - `analysis_result_*.json` | - `prefiltered_segments_*.json` |
| **Layer 4**<br>LLM-Based Reasoning | - `prefiltered_segments_*.json`<br>- `reasoning_prompt.txt` | - `final_highlight_result.json` |
| **Layer 5**<br>Video Stitching | - `final_highlight_result.json`<br>- `.mp4` video file | - `highlight.mp4` |
| **Layer 6**<br>Evaluation (Optional) | - `highlight.mp4` | - `summary_report.json`<br>- Logs & diagnostics |

