from typing import Any, Union, Tuple, List
import json
import re
from string import Template

from openai import AzureOpenAI
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel, Field

CUSTOM_SEGMENT_GENERATION_PROMPT = """
You are given the grounding data including descriptions, the transcripts and the additional data fields of chunks in a video.

Your task is to group the chunks into (1) TV Programs, (2) Promos, or (3) Commercial segments.

1. Tv Programs are the full-length TV program and the currently running shows or movies. 
	Instructions for TV Programs: 
		- TV Programs are the main running shows. The actor or cast members are usually present in the segments. For movies, they normally come with subtitles.
		- TV Programs follow the storyline of the show. If any inital segments are classified as TV Programs but do not follow the storyline or having different titles, they should belongs to either Promo or Commercial segments. Group them with the neighboring segments with most similar description or transcripts.
		- When a title card appears that indicates a start of program, do not mark it as promo segment. 
		- StarPlus TV is the TV channel, do not use it for the TV program title
		- **Strictly follow it: If a title card appears with a different name than the program which is running previously, consider it as a start of new program, do not make any assumptions that it's a part of previous program, or just a character introduction in previous program. 
		- **IMPORTANT* The 2 adjacent segments cannot be both TV Programs. If they have similar storyline, group them together. If there is a distinct storyline, the segment should be Promos or Commercials.
		- TV Programs are not short clips or segments. They are normally longer than Promos and Commercials segments.
2. Promos are short promotional clips for upcoming programs on the same or partner channels. These usually mention air times/days and do not promote brands or products.
	Instructions for Promos: -  
		**IMPORTANT** DO NOT classify promotion banners as Promos and DO NOT separate them as the promo segments. Only classify segments as Promos if they involve a distinct visual change or interruption that clearly separates them from the ongoing program 
		- If a segment promotes a program but also mentions multiple brands or sponsors, classify it as a Commercial. 
		- Identify the name of the program being promoted. 
		- A Promo can be for the same program that is currently running, if it mentions airing time or day, and appears as a separate segment.
		- When a title card appears that indicates the start of program, do not mark it as promo segment. 
3. Commercials: A paid advertisement promoting a brand, product, or service. 
	Instructions for Commercials: 
		- Commercials promote a product, service, or brand. 
		- Maintain consistent naming for Brand and Product across all occurrences of the same ad. 
		- If the same ad appears at multiple times in the video, at differnet timestamps, **treat each occurrence as a separate segment** — do not merge them. Just ensure that their metadata (title, brand, category, sector) remains consistent.
		- Make sure none of the commercials are missed, shorter commercials of 3-5 seconds should also be captured.
		- Generally, In early morning shows, there are no commercials, if found validate it properly and then mark it as Commercial.
		- DO NOT combine multiple commercials into a single segment. Each product, service, or brand should be treated as a separate segment, only combine segments if they're in the same ads.


Output the customized segments with following infor:
1. start time and end time for each segment in the hh:mm:ss.ms format. The customized segments must be CONTINUOUS which means the end time of the previous segment is the start time of the next segment and no time gap or overlap between the segments. 
2. SegmentClassificationReason: Explain the reason for the classification of the segment
3. SegmentClassification: One value from TV Program, Promos or Commercial
4. Title: the title of the segment using the following instructions: For Commercials: Brand + Product. For Promos: Name of the program being promoted. For Programs: Actual name of the program. Do not use any special characters in the title, except for spaces. For example, 'Colgate MaxFresh' is correct, while 'Colgate MaxFresh!' is incorrect.Try to combine the title of the segments with similar titles or related product. StarPlus TV is the TV channel, do not use it for the TV program title.
5. People: List the people in the segment if available

Besides descriptions and transcripts, use face or person identification to help with the logical grouping as the same people will be likely in the same segment
Group the segments with similar titles or related products together. For unknown segments or unknown data fields, the segments were extracted locally and thus missed the global context, group it with the neighboring segments with most similar description or transcript
Do not rely solely on the existing `segmentClassification` and `title` fields; they may contain errors. You can refer the title and segmentClassification from adjacent segments.  
**IMPORTANT** If the title of a segment is unknown or not clear, the segment should be grouped with adjacent segments and find the title from the adjacent segments with similar people or content or storyline.

                        

Here are the detailed descriptions, transcripts and the data fields of original segments:

${grounding_data}
"""


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

### The video customized segment structure output ###

class VideoCustomSegment(BaseModel):

    startTime: str = Field(
        ..., description="The start time of the segment in hh:mm:ss.ms format"
    )
    endTime: str = Field(
        ..., description="The end time of the segment in hh:mm:ss.ms format"
    )
    People: str = Field(
        ..., description="The list of people ids in the segment.")
    SegmentClassification: str = Field(..., description="Classify the segment type. There are 3 possible values: TV Program, Promos, Commercial. DO NOT classify and DO NOT separate a visual overlay or promotional banners as Promos. Only classify segments as Promos if they involve a distinct visual change or interruption that clearly separates them from the ongoing program")
    SegmentClassificationReason: str = Field(..., description="The reason for the classification of the segment")
    Title: str = Field(..., description="Generate the title of the segment using the following instructions: For Commercials: Brand + Product. For Promos: Name of the program being promoted. For Programs: Actual name of the program. Do not use any special characters in the title, except for spaces. For example, 'Colgate MaxFresh' is correct, while 'Colgate MaxFresh!' is incorrect.Try to combine the title of the segments with similar titles or related product. For the unknown title, it might comes from the adjacent segments, so try to combine them.")
    SegmentDescription: str = Field(..., description="Description of the segment content")
    
class VideoCustomSegmentList(BaseModel):
    """The video customized segment list 
    Attributes:
        segments (list[VideoCustomSegment]): The list of segments
    """

    segments: List[VideoCustomSegment] = Field(
        ..., description="The list of customized segments."
    )

### The video scene structure output ###
class VideoScene(BaseModel):
    """The video scene 
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
    ):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})

            if "o3-" in self.model or "o4-" in self.model:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    reasoning_effort="low",
                    max_completion_tokens=100000
                )
            else:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    max_tokens=16000,
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

def convert_seconds_to_hhmmssms(seconds: float) -> str:
    """Convert seconds hh:mm:ss:ms string"""
    if seconds < 0:
        return "00:00:00.000"

    int_seconds = int(seconds)

    # Calculate milliseconds
    milliseconds = round((seconds - int_seconds) * 1000)

    # Calculate hours, minutes, and seconds
    hours = int_seconds // 3600
    int_seconds %= 3600
    minutes = int_seconds // 60
    int_seconds %= 60

    return f"{hours:02}:{minutes:02}:{int_seconds:02}.{milliseconds:03}"

def _get_key_value(obj: dict) -> str:
    if "valueString" in obj:
        return obj["valueString"]
    if "valueObject" in obj:
        output_str = ""
        for k, v in obj["valueObject"].items():
            v_str = _get_key_value(v)
            output_str += f"{k}: {v_str}\n"
        return output_str
    if "valueArray" in obj:
        output_str = ""
        for item in obj["valueArray"]:
            v_str = _get_key_value(item)
            output_str += f"{v_str}\n"
        return output_str
    


def _get_next_processing_segments(
    contents: list, start_idx: int, token_size_threshold: int = 100000, duration_threshold: int = 910000
) -> Tuple[int, str]:
    """Get the next set of processing segments
    Args:
        contents (list): The list of segments
        start_idx (int): The start index
        duration_threshold (int): The duration of the processing segments each time in milliseconds
    Returns:
        Tuple[int, str]: The end index and the segment contents
    """
    end_idx = start_idx
    segment_contents = ""
    numb_tokens = 0
    start_processing_time = contents[end_idx]["startTimeMs"]
    while end_idx <= len(contents)-1:
        start_time = convert_seconds_to_hhmmssms(float(contents[end_idx]["startTimeMs"]) / 1000)
        end_time = convert_seconds_to_hhmmssms(float(contents[end_idx]["endTimeMs"]) / 1000)
        descriptions = "\n\n\n"
        if "subsegments" in contents[end_idx]["fields"]:
            subsegments = contents[end_idx]["fields"]["subsegments"]
            for segment in subsegments["valueArray"]:
                stime = segment["valueObject"]["startTime"]["valueString"]
                etime = segment["valueObject"]["endTime"]["valueString"]
                descriptions += (
                    f"Segment From {stime} to {etime}: \n "
                )
                for k, v in segment["valueObject"].items():
                    if k != "startTime" and k != "endTime" and k!= "SegmentClassificationReason":
                        v_str = _get_key_value(v)
                        descriptions += f"{k} : {json.dumps(v_str, ensure_ascii=False)}\n"
        else:
            descriptions += f"Segment From {start_time} to {end_time}: \n "
            for k, v in contents[end_idx]["fields"].items():
                v_str = _get_key_value(v)
                descriptions += f"{k} : {json.dumps(v_str, ensure_ascii=False)}\n"

        description_tokens = get_token_count(descriptions)
        numb_tokens += description_tokens
        transcripts = "---- Transcript: \n"
        for item in contents[end_idx]["transcriptPhrases"]:
            t_stime = convert_seconds_to_hhmmssms(float(item["startTimeMs"]) / 1000)
            t_etime = convert_seconds_to_hhmmssms(float(item["endTimeMs"]) / 1000)
            transcripts += (
                str(t_stime)
                + "--> "
                + str(t_etime)
                + ": "
                + item["text"]
            )
        numb_tokens += get_token_count(transcripts)
        segment_contents += descriptions + transcripts
        if contents[end_idx]["endTimeMs"] - start_processing_time > duration_threshold:
            # stop processing if the processing over 15 minutes
            end_idx += 1
            break
        end_idx += 1

    return end_idx, segment_contents

def _format_segment_time(custom_segments:VideoCustomSegmentList, is_seconds: bool) -> list:
    """Format the segment time
    Args:
        custom_segment_list (VideoCustomSegmentList): The list of customized segments
    Returns:
        The list of customized segments with formated time
    """
    custom_segment_list = []
    for idx, segment in enumerate(custom_segments.segments):
        if isinstance(segment.endTime, int):
            if is_seconds:
                # the time is in seconds
                start_time = convert_seconds_to_hhmmssms(float(segment.startTime))
                end_time = convert_seconds_to_hhmmssms(float(segment.endTime) )
            else:
                # the time is in milliseconds
                start_time = convert_seconds_to_hhmmssms(float(segment.startTime) / 1000)
                end_time = convert_seconds_to_hhmmssms(float(segment.endTime) / 1000)
            segment.startTime = start_time
            segment.endTime = end_time
        custom_segment_list.append(segment)
    return custom_segment_list

def generate_custom_segments(
    video_segment_result: dict, openai_assistant: OpenAIAssistant
) -> VideoCustomSegmentList:
    """Generate customized segments from the video segment result
    Args:
        video_segment_result (dict): The video segment result
        openai_assistant (shared_functions.AiAssistant): The AI assistant client
        segment_definition (str): The customized segment definition
    Returns:
        list: The list of customized segments
    """
    contents = video_segment_result["result"]["contents"]

    start_idx = 0
    end_idx = 0
    custom_segment_list = []
    while end_idx < len(contents):
        # Generate the scenes from the pre-processed list
        end_idx, next_segment_content = _get_next_processing_segments(
            contents, start_idx
        )
        segment_generation_prompt = Template(CUSTOM_SEGMENT_GENERATION_PROMPT).substitute(
            grounding_data=next_segment_content,
        )

        custom_segment_response = openai_assistant.get_structured_output_answer(
            "", segment_generation_prompt, VideoCustomSegmentList
        )

        # post_process the customized segments
        is_seconds = True if (isinstance(custom_segment_response.segments[0].endTime, int) and custom_segment_response.segments[0].endTime < 1000) else False
        custom_segment_list.extend(_format_segment_time(custom_segment_response, is_seconds))
            
        start_idx = end_idx

    
    # 2nd round to merge results from 15 minute chunks
    grounding_data = ""
    for idx, segment in enumerate(custom_segment_list):
        grounding_data += f"Segment {idx}: From {segment.startTime} to {segment.endTime}: \n "
        grounding_data += f"SegmentClassification: {segment.SegmentClassification}\n"
        grounding_data += f"SegmentClassificationReason: {segment.SegmentClassificationReason}\n"
        grounding_data += f"Title: {segment.Title}\n"
        grounding_data += f"SegmentDescription: {segment.SegmentDescription}\n"
        grounding_data += f"People: {segment.People}\n"

    segment_generation_prompt = Template(CUSTOM_SEGMENT_GENERATION_PROMPT).substitute(
        grounding_data=grounding_data
    )
    
    custom_segment_response = openai_assistant.get_structured_output_answer(
        "", segment_generation_prompt, VideoCustomSegmentList
    )
    
    is_seconds = True if (isinstance(custom_segment_response.segments[0].endTime, int) and custom_segment_response.segments[0].endTime < 1000) else False

    final_custom_segment_list = _format_segment_time(custom_segment_response, is_seconds)

    return VideoCustomSegmentList(segments=final_custom_segment_list)


def generate_scenes(
    video_segment_result: dict, openai_assistant: OpenAIAssistant, segment_definition: str
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
        scene_generation_prompt = Template(CUSTOM_SEGMENT_GENERATION_PROMPT).substitute(
            grounding_data=next_segment_content,
            segment_definition=segment_definition
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
    """Generate tags from the video segment result
    Args:
        video_segment_result (dict): The video segment result
        openai_assistant (shared_functions.AiAssistant): The AI assistant client
    Returns:
        VideoTagResponse: list of tags
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
