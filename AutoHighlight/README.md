# AutoHighlight: AI-Powered Video Highlights Generation

Welcome! AutoHighlight is an innovative AI-powered pipeline that automatically generates compelling video highlights from any video content. By combining **Azure Content Understanding** with **OpenAI's advanced reasoning capabilities**, it intelligently identifies the most engaging moments and creates professional highlight reels tailored to your specific needs.

## What AutoHighlight Does

Transform hours of video content into captivating highlights in minutes:

- **Smart Content Analysis**: Uses Azure Content Understanding to analyze every frame and identify key moments
- **AI-Powered Selection**: Leverages OpenAI o1's reasoning to select the most compelling clips based on your preferences  
- **Automated Editing**: Creates seamless highlight videos with professional transitions and timing
- **Personalized Results**: Tailors highlights to specific content types (sports, presentations, events) and your custom preferences
- **Production Ready**: Robust pipeline suitable for content creators, marketers, educators, and media professionals

Perfect for creating highlight reels from sports events, keynote presentations, product demos, educational content, entertainment shows, and more!

## Key Features

- **Multi-Domain Support**: Pre-configured schemas for sports (soccer, basketball, football), presentations, events, and custom content types
- **Intelligent Timestamp Selection**: AI reasoning identifies the most impactful moments based on content analysis
- **Flexible Output Control**: Customize highlight duration, clip density, transitions, and personalization preferences
- **Professional Video Processing**: FFmpeg-based editing with support for various resolutions and effects
- **User-Friendly Interface**: Jupyter notebook with step-by-step guidance and clear documentation

The pipeline uses the latest Azure AI Content Understanding API (2024-12-01-preview) combined with OpenAI's most advanced reasoning models.

## Repository Structure

| File | Description |
|------|-------------|
| `HighlightsNotebook.ipynb` | **Main pipeline** - Complete end-to-end highlight generation workflow |
| `Helper/SchemaManager.py` | Manages video analysis schemas and configurations |
| `Helper/BuildHighlight.py` | OpenAI-powered highlight planning and timestamp selection |
| `Helper/JsonParser.py` | Processes and filters Azure Content Understanding output |
| `Helper/VideoStitchingFfmpeg.py` | FFmpeg-based video editing and assembly |
| `Helper/` | Python helper modules folder |
| `schemas/` | Pre-built analysis schemas for different content types |
| `Requirements.txt` | Python package dependencies |

## Getting Started

### Prerequisites

Make sure you have the following:

- **Python 3.8+**
- **FFmpeg** (for video processing)
- **Azure AI Services** resource with Content Understanding enabled
- **Azure OpenAI** resource with GPT-4o or newer model deployed


## Azure Resource Configuration

### Option 1: Automated Setup (Recommended)

If you have Azure Developer CLI installed:

```bash
# Login to Azure
az login

# Create resources automatically (requires subscription admin permissions)
azd init -t AutoHighlight
azd up
```

### Option 2: Manual Setup

1. **Create Azure AI Services resource**:
   - Go to Azure Portal → Create Resource → AI Services
   - Enable Content Understanding capabilities
   - Grant yourself **Cognitive Services User** role

2. **Create Azure OpenAI resource**:
   - Deploy a GPT-o1
   - Grant yourself **Cognitive Services OpenAI User** role

3. **Configure credentials**:
   - Open `highlights_notebook.ipynb`
   - Update the API Configuration cell with your endpoints and API keys

## Usage

1. **Open the main notebook**:
   ```bash
   jupyter notebook highlights_notebook.ipynb
   ```

2. **Configure your video**:
   - Set `SOURCE_VIDEO_PATH` to your video file
   - Choose appropriate `VIDEO_TYPE` (soccer, keynote, etc.)
   - Customize highlight preferences (duration, density, personalization)

3. **Run the pipeline**:
   - Execute each cell step-by-step
   - The notebook guides you through:
     - Schema generation and activation
     - Video analysis with Azure Content Understanding
     - AI-powered highlight selection with OpenAI
     - Final video assembly

4. **Get your highlights**:
   - Professional highlight video saved as `highlight.mp4`
   - Detailed analysis and planning data for review

## Supported Content Types

| Schema Type | Best For | Key Features Detected |
|-------------|----------|----------------------|
| **soccer** | Football/Soccer matches | Goals, saves, tackles, celebrations |
| **basketball** | Basketball games | Shots, dunks, steals, crowd reactions |
| **keynote** | Presentations, talks | Key quotes, audience reactions, slides |
| **product demo** | Software demos, launches | Feature highlights, user interactions |
| **education** | Lectures, tutorials | Important concepts, demonstrations |
| **custom** | Any content type | Configurable based on your needs |

## Customization

### Adding New Content Types

1. Create a new schema file in `schemas/your_content_type.json`
2. Define what events and moments to detect
3. Set scoring criteria for highlight-worthiness
4. Use `VIDEO_TYPE = "your_content_type"` in the notebook

### Advanced Configuration

- **Personalization**: Add specific keywords or themes you want emphasized
- **Clip Density**: Control how many clips per minute (low/medium/high)
- **Transitions**: Choose between cuts, fades, or custom effects
- **Resolution**: Output in 720p, 1080p, or custom resolutions

## Pipeline Architecture

```
1. Schema Generation → 2. Video Analysis → 3. Content Filtering → 4. AI Selection → 5. Video Assembly
     (Custom AI)         (Azure CU)        (Rule-based)       (OpenAI o1)    (FFmpeg)
```

1. **Schema Generation**: Creates custom Azure analyzer using OpenAI reasoning
2. **Video Analysis**: Azure Content Understanding analyzes every frame
3. **Content Filtering**: Rules-based filtering removes low-quality segments  
4. **AI Selection**: OpenAI o1 intelligently selects and arranges final clips
5. **Video Assembly**: FFmpeg stitches clips into professional highlight reel

## Sample Results

**Input**: 45-minute soccer match  
**Output**: 2-minute highlight reel with 8 key moments  
**Processing Time**: ~12 minutes  
**Quality**: Professional-grade with smooth transitions  

## Technical Details

The highlight generation pipeline consists of **5 modular stages**:

| Stage | Description | Input | Output |
|-------|-------------|-------|--------|
| **1. Schema Generation** | Creates custom Azure AI analyzer using OpenAI reasoning | Video type, preferences | Custom analyzer schema |
| **2. Video Analysis** | Submits video to Azure Content Understanding | Source video, schema | Frame analysis results |
| **3. Content Filtering** | Rule-based filtering removes low-quality segments | Analysis results | Filtered segments |
| **4. AI Reasoning** | OpenAI o1 selects optimal timestamps and narrative flow | Filtered segments | Highlight plan |
| **5. Video Assembly** | FFmpeg stitches clips with transitions and effects | Highlight plan, source video | Final highlight MP4 |

## Troubleshooting

**Common Issues**:
- **FFmpeg not found**: Install FFmpeg and ensure it's in your PATH
- **API authentication errors**: Verify your Azure credentials and resource permissions
- **Video format issues**: Ensure your video is in a standard format (MP4, AVI, MOV)
- **Memory errors**: For very large videos, consider splitting into smaller segments

**Package Installation Issues**:
```bash
# If you encounter package conflicts, try:
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Legal Notices

**Trademarks** - This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos is subject to those third-party's policies.

**Data Collection** - The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

