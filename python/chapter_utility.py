from typing import Any, Union, Tuple
import json
from string import Template

from openai import AzureOpenAI
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel, Field

import prompts


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
        scenes (list[VideoScene]): The list of scenes in the chapter
    """

    startTimeMs: int = Field(
        ..., description="The start time stamp of the chapter in miliseconds."
    )
    endTimeMs: int = Field(
        ..., description="The end time stamp of the chapter in miliseconds."
    )
    title: str = Field(..., description="The title of the chapter.")
    scenes: list[VideoScene] = Field(
        ..., description="The list of scenes in the chapter."
    )


class VideoChapterResponse(BaseModel):
    """The video chapter response analyzer
    Attributes:
        chapters (list[VideoChapter]): The list of chapters in the video
    """

    chapters: list[VideoChapter] = Field(
        ..., description="The list of chapters in the video."
    )


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

        self.analyzer = deployment_name

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3)
    )
    def _chat_completion_request(self, messages, tools=None, tool_choice=None):
        try:
            response = self.client.chat.completions.create(
                analyzer=self.analyzer,
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
    ):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})

            completion = self.client.beta.chat.completions.parse(
                analyzer=self.analyzer,
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
    """Get the token count of a text.
    Args:
        text (str): The text
        model_name (str): The analyzer name
    Returns:
        int: The token count
    """
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    return len(tokens)


def _get_next_processing_segments(
    contents: list, start_idx: int, token_size_threshold: int = 4000
) -> Tuple[int, str]:
    """Get the next set of processing segments
    Args:
        contents (list): The list of segments
        start_idx (int): The start index
    Returns:
        Tuple[int, str]: The end index and the segment contents
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
    """Generate scenes from the video segment result
    Args:
        video_segment_result (dict): The video segment result
        openai_assistant (shared_functions.AiAssistant): The AI assistant client
    Returns:
        list: The list of scenes with transcripts
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
        scene_generation_prompt = Template(prompts.SCENE_GENERATION_PROMPT).substitute(
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
    """Extract transcripts for the scenes
    Args:
        video_segment_result (dict): The video segment result
        video_scene_response (VideoSceneResponse): The video scene response
    Returns:
        list: The list of scenes with transcripts
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
    """Generate chapters from the scenes
    Args:
        scenes (VideoSceneResponse): The list of scenes
        openai_assistant (shared_functions.AiAssistant): The OpenAI assistant client
    Returns:
        list: The list of chapters
    """
    scenes = scene_result.scenes
    if len(scenes) == 0:
        return []

    scene_descriptions = ""
    for scene in scenes:
        description_and_transcript = (
            f"From {scene.startTimeMs}ms to {scene.endTimeMs}ms: {scene.description} "
        )
        if scene.transcript != "":
            description_and_transcript += f" ---- Transcript: {scene.transcript}\n\n"
        scene_descriptions += description_and_transcript
    chapter_generation_prompt = Template(prompts.CHAPTER_GENERATION_PROMPT).substitute(
        descriptions=scene_descriptions
    )
    chapter_response = VideoChapterResponse(chapters=[])
    chapter_response = openai_assistant.get_structured_output_answer(
        "", chapter_generation_prompt, VideoChapterResponse
    )
    return chapter_response
