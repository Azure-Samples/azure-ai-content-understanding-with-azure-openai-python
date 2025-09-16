import json
import os
import re
from string import Template
from typing import Any, List, Literal, Optional, Tuple, Union

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential
import tiktoken

SCENE_GENERATION_PROMPT = """
    You are given the segment index, descriptions and the transcripts of clip segments from a video with timestamps in miliseconds. Combine the segments into scenes based on the 2 main steps:
    Step 1: Check if the video segment can be a single scene or combine with other segments or broken down into multiple scenes based on the visual description and the transcript. A scene is a segment of the video where a continous block for storytelling unfolds within a specific time, place, and set of characters. The big long or single scenes should be broken into smaller sub-scenes to structure the videos coherently. The generated scenes will be used in the next step to generate chapters which are higher level of distinct content of the video, such as a topic change. The transcript or the description can be empty.
    Step 2: Output the scene result in the structured format with start and end time of the scene in miliseconds and the description of the scene. Include the segment indexes that belong to the scene in the output.
    
    Here are the segment index, detailed descriptions and transcripts of the video segments:

    ${descriptions}
    """

CHAPTER_GENERATION_PROMPT = """
    You are given the descriptions and the transcripts of scenes. Combine the scenes into chapters based on the 2 main steps:
    Step 1: Check if the scene can be a single chapter or combine with other scenes or broken down to create chapters based on the visual description and the transcript. A chapter is a collection of scenes or content that share a common theme, setting, or narrative purpose. Chapters are higher level of distinct content of the video, such as a topic change. The transcript or the description can be empty. Don't generate too short chapters as they are higher level of content.
    Step 2: Output the chapter result in the structured format with start and end time of the chapter in miliseconds and the title of the chapter. Keep the chapter title concise and descriptive. Include the scene indexes that belong to the chapter.

    Here are the detailed descriptions and transcripts of the video scenes:

    ${descriptions}
    """

DEDUP_PROMPT = """
    Given an input list of tags, remove duplicate tags which are semantically similar.

    Here is the input tag list:

    ${tag_list}
    """

BASE_DIR = os.path.dirname(__file__)

HIGHLIGHT_PLAN_PROMPT_PATH = os.path.join(BASE_DIR, "reasoning_prompt.txt")


class VideoTagResponse(BaseModel):
    """The video tag response analyzer
    Attributes:
        tags (list[str]): The list of tags
    """

    tags: list[str] = Field(..., description="The list of tags in the video")


class SegmentID(BaseModel):
    """The video segment id analyzer
    Attributes:
        id (int): The value string
    """

    id: int = Field(..., description="The index of video segment.")


class VideoScene(BaseModel):
    """The video scene analyzer
    Attributes:
        startTimeMs (int): The start time stamp of the scene in miliseconds
        endTimeMs (int): The end time stamp of the scene in miliseconds
        description (str): The detail description
    """

    startTimeMs: int = Field(
        ..., description="The start time stamp of the scene in miliseconds."
    )
    endTimeMs: int = Field(
        ..., description="The end time stamp of the scene in miliseconds."
    )
    description: str = Field(..., description="The detail description of the scene.")


class VideoSceneWithID(VideoScene):
    """The video scene analyzer with segment ID
    Attributes:
        segmentIDs (list[SegmentID]): The list of segment IDs
    """

    segmentIDs: list[SegmentID] = Field(
        ..., description="The list of segment indexes that in the scene."
    )


class VideoSceneWithTranscript(VideoSceneWithID):
    """The video scene analyzer with transcript
    Attributes:
        transcript (str): The transcript of the scene
    """

    transcript: str = Field(..., description="The transcript of the scene.")


class VideoSceneResponse(BaseModel):
    """The video scene response analyzer
    Attributes:
        scenes (list[VideoSceneWithID]): The list of scenes in the video
    """

    scenes: list[VideoSceneWithID] = Field(
        ..., description="The list of scenes in the video."
    )


class VideoSceneResponseWithTranscript(BaseModel):
    """The video scene response analyzer with transcript
    Attributes:
        scenes (list[VideoSceneWithTranscript]): The list of scenes in the video
    """

    scenes: list[VideoSceneWithTranscript] = Field(
        ..., description="The list of scenes in the video."
    )


class VideoChapter(BaseModel):
    """The video chapter analyzer
    Attributes:
        startTimeMs (int): The start time stamp of the chapter in miliseconds
        endTimeMs (int): The end time stamp of the chapter in miliseconds
        title (str): The title of the chapter
        scene_ids (list[int]): The list of indexes in the chapter
    """

    startTimeMs: int = Field(
        ..., description="The start time stamp of the chapter in miliseconds."
    )
    endTimeMs: int = Field(
        ..., description="The end time stamp of the chapter in miliseconds."
    )
    title: str = Field(..., description="The title of the chapter.")
    scene_ids: list[int] = Field(..., description="The list of scene indexes.")


class VideoChapterResponse(BaseModel):
    """The video chapter response analyzer
    Attributes:
        chapters (list[VideoChapter]): The list of chapters in the video
    """

    chapters: list[VideoChapter] = Field(
        ..., description="The list of chapters in the video."
    )


class MMIField(BaseModel):
    type: Literal["string", "date", "time", "number", "integer"]
    method: Literal["classify", "generate"]
    description: str
    enum: Optional[List[str]] = None


class KeyValueField(BaseModel):
    key: str
    value: MMIField

class ObjectField(BaseModel):
    type: str
    properties: List[KeyValueField]  # OpenAI structured output doesn't support Dict

class SegmentField(BaseModel):
    type: str
    method: str
    items: ObjectField


class SegmentFields(BaseModel):
    Segments: SegmentField


class SegmentFieldSchema(BaseModel):
    fields: SegmentFields


class SegementConfig(BaseModel):
    returnDetails: bool
    segmentationMode: str
    segmentationDefinition: Optional[str] = None


class SegmentAnalyzer(BaseModel):
    description: str
    baseAnalyzerId: str
    config: SegementConfig
    fieldSchema: SegmentFieldSchema


class ActSummary(BaseModel):
    """Summary for each act in the highlight plan.
    Attributes:
        ActName (str): The name of the act (e.g., Introduction, Climax, Resolution)
        Summary (str): The summary text for the act
    """
    ActName: str
    Summary: str


class HighlightClip(BaseModel):
    """A single highlight clip in the highlight plan.
    Attributes:
        SegmentId (str): The segment identifier
        StartTimeMs (int): The start time in milliseconds
        EndTimeMs (int): The end time in milliseconds
        Act (str): The act this clip belongs to
        NarrativeRole (str): The narrative role of the clip
        WhyChosen (str): Reason for selection
    """
    SegmentId: str
    StartTimeMs: int
    EndTimeMs: int
    Act: str
    NarrativeRole: str
    WhyChosen: str


class HighlightPlan(BaseModel):
    """The highlight plan output structure.
    Attributes:
        SelectedClips (List[HighlightClip]): List of selected highlight clips
        ActSummaries (List[ActSummary]): List of act summaries
        DidReturnFewerThanRequested (bool): Whether fewer clips than requested were returned
    """
    SelectedClips: List[HighlightClip]
    ActSummaries: List[ActSummary]
    DidReturnFewerThanRequested: bool


class OpenAIAssistant:
    """Azure OpenAI Assistant client"""

    def __init__(
        self,
        aoai_end_point: str,
        aoai_api_version: str,
        deployment_name: str,
        aoai_api_key: str,
    ):
        if aoai_api_key is None or aoai_api_key == "":
            print("Using Entra ID/AAD to authenticate")
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )

            self.client = AzureOpenAI(
                api_version=aoai_api_version,
                azure_endpoint=aoai_end_point,
                azure_ad_token_provider=token_provider,
            )
        else:
            print("Using API key to authenticate")
            self.client = AzureOpenAI(
                api_version=aoai_api_version,
                azure_endpoint=aoai_end_point,
                api_key=aoai_api_key,
            )

        self.model = deployment_name

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3)
    )
    def _chat_completion_request(self, messages, tools=None, tool_choice=None):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                seed=0,
                temperature=0.0,
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

    def get_answer(
        self,
        system_message: str,
        prompt: Union[str, Any],
        input_schema=None,
        output_schema=None,
    ):
        """Get an answer from the assistant."""

        def schema_to_tool(schema: Any):
            assert schema.__doc__, f"{schema.__name__} is missing a docstring."
            return [
                {
                    "type": "function",
                    "function": {
                        "name": schema.__name__,
                        "description": schema.__doc__,
                        "parameters": schema.schema(),
                    },
                }
            ], {"type": "function", "function": {"name": schema.__name__}}

        tools = None
        tool_choice = None
        if output_schema:
            tools, tool_choice = schema_to_tool(output_schema)

        if input_schema:
            user_message = f"Schema: ```{input_schema.model_json_schema()}```\nData: ```{input_schema.parse_obj(prompt).model_dump_json()}```"
        else:
            user_message = prompt

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        response = self._chat_completion_request(
            messages, tools=tools, tool_choice=tool_choice
        )
        assistant_message = response.choices[0].message
        if assistant_message.content:
            return assistant_message.content
        else:
            try:
                return json.loads(
                    assistant_message.tool_calls[0].function.arguments, strict=False
                )
            except:
                return assistant_message.tool_calls[0].function.arguments

    def get_structured_output_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: BaseModel,
        seed: int = 0,
        temperature: float = 0.0,
        is_model_after_2024: bool = False,
    ):
        """
        Get a structured output answer from the assistant.

        Args:
            system_prompt (str): The system prompt.
            user_prompt (str): The user prompt.
            response_format (BaseModel): The expected response format as a Pydantic model.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.

        Returns:
            BaseModel: The parsed response as an instance of the response_format model.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
            if is_model_after_2024:
                completion = self.client.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    max_completion_tokens=50000,
                    seed=seed,
                )
            else:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    max_tokens=4096,
                    seed=seed,
                    temperature=temperature,
                )
            response = completion.choices[0].message.parsed
            return response
        except Exception as ex:
            print(f"Unable to generate ChatCompletion response. Exception: {ex}")
            return None


def get_token_count(text: str, model_name: str = "gpt-4o") -> int:
    """
    Get the token count of a text for a given model.

    Args:
        text (str): The text to tokenize.
        model_name (str): The model name for encoding.

    Returns:
        int: The token count.
    """
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    return len(tokens)


def _get_next_processing_segments(
    contents: list, start_idx: int, token_size_threshold: int = 32000
) -> Tuple[int, str]:
    """
    Get the next set of processing segments that fit within the token size threshold.

    Args:
        contents (list): The list of segments.
        start_idx (int): The start index.
        token_size_threshold (int, optional): The token size threshold. Defaults to 32000.

    Returns:
        Tuple[int, str]: The end index and the segment contents as a string.
    """
    end_idx = start_idx
    segment_contents = ""
    numb_tokens = 0
    while numb_tokens < token_size_threshold and end_idx < len(contents):
        start_time = contents[end_idx]["startTimeMs"]
        end_time = contents[end_idx]["endTimeMs"]
        value_str = contents[end_idx]["fields"]["segmentDescription"]["valueString"]
        descriptions = (
            f"Segment {end_idx}: From {start_time}ms to {end_time}ms: {value_str}"
        )
        description_tokens = get_token_count(descriptions)
        numb_tokens += description_tokens
        transcripts = "---- Transcript: \n"
        for item in contents[end_idx]["transcriptPhrases"]:
            transcripts += (
                str(item["startTimeMs"])
                + "ms --> "
                + str(item["endTimeMs"])
                + "ms : "
                + item["text"]
            )
        numb_tokens += get_token_count(transcripts)
        segment_contents += descriptions + transcripts
        end_idx += 1
    return end_idx, segment_contents


def generate_scenes(
    video_segment_result: dict, openai_assistant: OpenAIAssistant
) -> VideoSceneResponseWithTranscript:
    """
    Generate scenes from the video segment result using the OpenAI assistant.

    Args:
        video_segment_result (dict): The video segment result.
        openai_assistant (OpenAIAssistant): The AI assistant client.

    Returns:
        VideoSceneResponseWithTranscript: The list of scenes with transcripts.
    """
    contents = video_segment_result["result"]["contents"]

    start_idx = 0
    end_idx = 0
    final_scene_list = []
    while end_idx < len(contents):
        # Generate the scenes from the pre-processed list
        end_idx, next_segment_content = _get_next_processing_segments(
            contents, start_idx
        )
        scene_generation_prompt = Template(SCENE_GENERATION_PROMPT).substitute(
            descriptions=next_segment_content
        )
        scence_response = VideoSceneResponse(scenes=[])
        scence_response = openai_assistant.get_structured_output_answer(
            "", scene_generation_prompt, VideoSceneResponse
        )
        print(scence_response)

        scenes_with_transcript = _extract_transcripts_for_scenes(
            video_segment_result, scence_response
        )

        if end_idx < len(contents):
            final_scene_list.extend(scenes_with_transcript.scenes[:-1])
        else:
            final_scene_list.extend(scenes_with_transcript.scenes)
        last_scene = scence_response.scenes[-1]
        start_idx = last_scene.segmentIDs[0].id

    return VideoSceneResponseWithTranscript(scenes=final_scene_list)


def _extract_transcripts_for_scenes(
    video_segment_result: dict, video_scene_response: VideoSceneResponse
) -> VideoSceneResponseWithTranscript:
    """
    Extract transcripts for the scenes.

    Args:
        video_segment_result (dict): The video segment result.
        video_scene_response (VideoSceneResponse): The video scene response.

    Returns:
        VideoSceneResponseWithTranscript: The list of scenes with transcripts.
    """
    if len(video_scene_response.scenes) == 0:
        return VideoSceneResponseWithTranscript(scenes=[])

    contents = video_segment_result["result"]["contents"]

    transcripts = ["" for _ in range(len(video_scene_response.scenes))]

    for idx, scene in enumerate(video_scene_response.scenes):
        for segment in scene.segmentIDs:
            for phrase in contents[segment.id]["transcriptPhrases"]:
                start_time = phrase["startTimeMs"]
                end_time = phrase["endTimeMs"]
                transcript_text = f"{start_time}ms --> {end_time}ms :" + phrase["text"]
                if transcript_text not in transcripts[idx]:
                    transcripts[idx] += transcript_text + "\n"

    scenes_with_transcript = []
    for idx, scene in enumerate(video_scene_response.scenes):
        scenes_with_transcript.append(
            VideoSceneWithTranscript(
                startTimeMs=scene.startTimeMs,
                endTimeMs=scene.endTimeMs,
                description=scene.description,
                segmentIDs=scene.segmentIDs,
                transcript=transcripts[idx],
            )
        )
    return VideoSceneResponseWithTranscript(scenes=scenes_with_transcript)


def generate_chapters(
    scene_result: VideoSceneResponse, openai_assistant: OpenAIAssistant
) -> VideoChapterResponse:
    """
    Generate chapters from the scenes using the OpenAI assistant.

    Args:
        scene_result (VideoSceneResponse): The list of scenes.
        openai_assistant (OpenAIAssistant): The OpenAI assistant client.

    Returns:
        VideoChapterResponse: The list of chapters.
    """
    scenes = scene_result.scenes
    if len(scenes) == 0:
        return []

    scene_descriptions = ""
    for idx, scene in enumerate(scenes):
        description_and_transcript = (
            f"Scene Index {idx} -- From {scene.startTimeMs}ms to {scene.endTimeMs}ms: {scene.description} "
        )
        if scene.transcript != "":
            description_and_transcript += f" ---- Transcript: {scene.transcript}\n\n"
        scene_descriptions += description_and_transcript
    chapter_generation_prompt = Template(CHAPTER_GENERATION_PROMPT).substitute(
        descriptions=scene_descriptions
    )
    chapter_response = VideoChapterResponse(chapters=[])
    chapter_response = openai_assistant.get_structured_output_answer(
        "", chapter_generation_prompt, VideoChapterResponse
    )
    return chapter_response


def aggregate_tags(
    video_segment_result: dict, openai_assistant: OpenAIAssistant
) -> VideoTagResponse:
    """
    Generate tags from the video segment result using the OpenAI assistant.

    Args:
        video_segment_result (dict): The video segment result.
        openai_assistant (OpenAIAssistant): The AI assistant client.

    Returns:
        VideoTagResponse: List of tags.
    """
    contents = video_segment_result["result"]["contents"]
    tags = []

    for content in contents:
        value_str = content["fields"]["tags"]["valueString"]
        segment_tags = list(map(str.lower, value_str.split(',')))
        tags.extend(segment_tags)

    tags_dedup = set(map(lambda x: re.sub(r'^ ', '', x), tags))
    tag_dedup_prompt = Template(DEDUP_PROMPT).substitute(tag_list=tags_dedup)

    tag_response = VideoTagResponse(tags=[])
    tag_response = openai_assistant.get_structured_output_answer(
        "", tag_dedup_prompt, VideoTagResponse
    )

    return tag_response


def get_highlight_plan(
    openai_assistant: OpenAIAssistant,
    segments: list,
    video_type: str,
    clip_density: float,
    target_duration_s: int,
    personalization: str,
) -> Optional[HighlightPlan]:
    """
    Generate a highlight plan using OpenAI structured output.

    Args:
        openai_assistant (OpenAIAssistant): The OpenAI assistant client.
        segments (list): The list of segment dicts.
        video_type (str): The type of video.
        clip_density (float): The clip density.
        target_duration_s (int): The target duration in seconds.
        personalization (str): Personalization string.

    Returns:
        Optional[HighlightPlan]: The structured highlight plan, or None if failed.
    """
    # Read the prompt template from the fixed path
    with open(HIGHLIGHT_PLAN_PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    filled_prompt = (
        prompt_template
        .replace("{{video_type}}", video_type)
        .replace("{{clip_density}}", str(clip_density))
        .replace("{{target_duration_s}}", str(target_duration_s))
        .replace("{{personalization}}", str(personalization))
    )
    
    # Prepare user message
    user_input = {
        "video_type": video_type,
        "clip_density": clip_density,
        "target_duration_s": target_duration_s,
        "personalization": personalization,
        "Segments": segments
    }
    user_message = json.dumps(user_input, indent=2)

    try:
        result = openai_assistant.get_structured_output_answer(
            system_prompt=filled_prompt,
            user_prompt=user_message,
            response_format=HighlightPlan,
            temperature=0.0,
            is_model_after_2024=True,
        )
        return result
    except Exception as e:
        print(f"Failed to get highlight plan: {e}")
        return None


def convert_properties_dict_to_list(data: Any) -> Any:
    """
    Recursively converts all 'properties' dicts in a JSON schema
    to lists of {key, value} objects to match SegmentAnalyzer's KeyValueField model.

    Args:
        data (Any): The input data (dict, list, or other).

    Returns:
        Any: The converted data with 'properties' as lists.
    """
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if k == "properties" and isinstance(v, dict):
                # convert dict -> list of {key, value}
                new_data[k] = [{"key": key, "value": value} for key, value in v.items()]
            else:
                new_data[k] = convert_properties_dict_to_list(v)
        return new_data
    elif isinstance(data, list):
        return [convert_properties_dict_to_list(item) for item in data]
    else:
        return data
    

def convert_properties_list_to_dict(data: dict) -> dict:
    """
    Recursively converts all 'properties' lists in a JSON schema
    to dicts to match the expected schema format.

    Args:
        data (dict): The input data.

    Returns:
        dict: The converted data with 'properties' as dicts.
    """
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if k == "properties" and isinstance(v, list):
                # convert list of {key, value} -> dict
                new_data[k] = {item["key"]: item["value"] for item in v}
            else:
                new_data[k] = convert_properties_list_to_dict(v)
        return new_data
    elif isinstance(data, list):
        return [convert_properties_list_to_dict(item) for item in data]
    else:
        return data


def get_converted_properties_list_schema_str(schema: dict) -> str:
    """
    Convert a schema's 'properties' dicts to lists and return as a JSON string.

    Args:
        schema (dict): The schema dictionary.

    Returns:
        str: The JSON string of the converted schema.
    """
    converted_schema = convert_properties_dict_to_list(schema)
    return json.dumps(converted_schema, indent=2)

def get_converted_properties_dict_schema_dict(result: SegmentAnalyzer) -> dict:
    """
    Convert a SegmentAnalyzer model's properties lists to dicts.

    Args:
        result (SegmentAnalyzer): The SegmentAnalyzer model.

    Returns:
        dict: The converted schema dictionary.
    """
    result_dict = result.model_dump()
    return convert_properties_list_to_dict(result_dict)

def generate_schema_llm(
    openai_assistant: OpenAIAssistant,
    video_type: str,
    example_path: str,
    clip_density: float = 1.0,
    target_duration_s: int = 300,
    personalization: str = "none"
) -> dict:
    """
    Generate a schema for the given video type using OpenAI structured output.

    Args:
        openai_assistant (OpenAIAssistant): The OpenAI assistant client.
        video_type (str): The type of video.
        example_path (str): Path to the example schema file.
        clip_density (float, optional): Target clip density. Defaults to 1.0.
        target_duration_s (int, optional): Target total duration in seconds. Defaults to 300.
        personalization (str, optional): Personalization string. Defaults to "none".

    Returns:
        dict: The generated schema as a dictionary.
    """
    # OpenAI structured output doesn't support Dict, so convert 'properties' dicts to lists.
    # Then convert back after getting the response.
    with open(example_path, "r", encoding="utf-8") as f:
        example_schema = json.load(f)
    converted_example_schema_str = get_converted_properties_list_schema_str(example_schema)

    prompt = (
        "You are an expert in Azure Content Understanding custom analyzer schemas.\n"
        f"Here is an example schema for {video_type}:\n\n"
        f"{converted_example_schema_str}\n\n"
        f"Generate a complete, valid analyzer schema JSON for '{video_type}' videos, "
        f"matching the same format and level of detail.\n"
        f"- Target clip density: {clip_density} clips per minute.\n"
        f"- Target total duration: {target_duration_s} seconds.\n"
        f"- Personalization: {personalization}\n"
        f"Return only the schema JSON."
    )

    result = openai_assistant.get_structured_output_answer(
        system_prompt="Generate Azure Content Understanding custom analyzer schemas.",
        user_prompt=prompt,
        response_format=SegmentAnalyzer,
        is_model_after_2024=True,
    )
    result_dict_converted = get_converted_properties_dict_schema_dict(result)

    return result_dict_converted

def add_field_llm(
    openai_assistant: OpenAIAssistant,
    current_schema: dict,
    field_description: str
) -> dict:
    """
    Use OpenAI structured output to add a field to the schema.

    Args:
        openai_assistant (OpenAIAssistant): The OpenAI assistant client.
        current_schema (dict): The current schema dictionary.
        field_description (str): Description of the field to add.

    Returns:
        dict: The updated schema dictionary.
    """
    # OpenAI structured output doesn't support Dict, so convert 'properties' dicts to lists.
    # Then convert back after getting the response.
    converted_schema_str = get_converted_properties_list_schema_str(current_schema)

    prompt = (
        f"You are an expert in JSON schema design. Given the current schema and a user description, "
        f"add a new field to the Segments.items.properties section. Return only the updated JSON schema.\n\n"
        f"Current schema:\n{converted_schema_str}\n\n"
        f"User request: {field_description}"
    )
    result = openai_assistant.get_structured_output_answer(
        system_prompt="Add a field to the schema.",
        user_prompt=prompt,
        response_format=SegmentAnalyzer,
        is_model_after_2024=True,
    )

    new_schema = get_converted_properties_dict_schema_dict(result)

    return new_schema
