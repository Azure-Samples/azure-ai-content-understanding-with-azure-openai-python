import json
import logging
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
from moviepy import vfx


def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get video duration and basic info using ffprobe.

    Args:
        video_path (str or Path): Path to the video file.

    Returns:
        dict: Dictionary containing duration, fps, width, height, and has_audio.
    """
    try:
        clip = VideoFileClip(str(video_path))
        duration = clip.duration
        fps = clip.fps
        width, height = clip.size
        has_audio = clip.audio is not None
        clip.close()
        
        return {
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'has_audio': has_audio
        }
    except Exception as e:
        logging.error(f"Error getting video info: {e}")
        return {'duration': 0, 'fps': 30, 'width': 1920, 'height': 1080, 'has_audio': False}



def extract_clip(
    video_path: Union[str, Path],
    start_time: float,
    end_time: float,
    output_path: Union[str, Path],
    speed_factor: float = 1.0
) -> bool:
    """
    Extract a clip from a video file, optionally changing its speed.

    Args:
        video_path (str or Path): Path to the source video.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        output_path (str or Path): Path to save the extracted clip.
        speed_factor (float, optional): Speed multiplier for the clip. Defaults to 1.0.

    Returns:
        bool: True if extraction succeeded, False otherwise.
    """
    try:
        clip = VideoFileClip(str(video_path))

        # Use slicing instead of subclip
        clip = clip[start_time:end_time]

        # Apply speed change if needed
        if speed_factor != 1.0:
            clip = clip.speedx(factor=speed_factor)

        # Write output
        clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            logger=None
        )

        clip.close()
        return True

    except Exception as e:
        logging.error(f"Error extracting clip {start_time}-{end_time}: {e}")
        return False


def add_fade_effects(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    fadein_duration: float = 0.5,
    fadeout_duration: float = 0.5
) -> bool:
    """
    Add fade in/out effects to a video.

    Args:
        video_path (str or Path): Path to the source video.
        output_path (str or Path): Path to save the output video.
        fadein_duration (float, optional): Duration of fade-in in seconds. Defaults to 0.5.
        fadeout_duration (float, optional): Duration of fade-out in seconds. Defaults to 0.5.

    Returns:
        bool: True if fade effects were applied successfully, False otherwise.
    """
    try:
        clip = VideoFileClip(str(video_path))

        # Apply FadeIn
        if fadein_duration > 0:
            fadein_effect = FadeIn(duration=fadein_duration)
            clip = fadein_effect.apply(clip)

        # Apply FadeOut
        if fadeout_duration > 0:
            fadeout_effect = FadeOut(duration=fadeout_duration)
            clip = fadeout_effect.apply(clip)

        # Write the output video
        clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            logger=None
        )
        clip.close()
        return True
    except Exception as e:
        logging.error(f"Error adding fade effects: {e}")
        return False


def concatenate_videos(
    video_paths: List[Union[str, Path]],
    output_path: Union[str, Path]
) -> bool:
    """
    Concatenate multiple videos using MoviePy v2.x (pure Python).

    Args:
        video_paths (list): List of paths to video files to concatenate.
        output_path (str or Path): Path to save the concatenated video.

    Returns:
        bool: True if concatenation succeeded, False otherwise.
    """
    try:
        clips = [VideoFileClip(str(path)) for path in video_paths]

        # Concatenate clips (method="compose" handles different resolutions/FPS)
        final_clip = concatenate_videoclips(clips, method="compose")

        # Write the output video
        final_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            logger=None
        )

        # Close all clips to free resources
        for clip in clips:
            clip.close()
        final_clip.close()

        return True

    except Exception as e:
        logging.error(f"Error concatenating videos: {e}")
        return False


def resize_video(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    height: int = 720
) -> bool:
    """
    Resize a video to the specified height while maintaining aspect ratio (MoviePy v2.x).

    Args:
        video_path (str or Path): Path to the source video.
        output_path (str or Path): Path to save the resized video.
        height (int, optional): Desired output height in pixels. Defaults to 720.

    Returns:
        bool: True if resizing succeeded, False otherwise.
    """
    try:
        clip = VideoFileClip(str(video_path))

        # Apply Resize effect via class-based vfx
        clip_resized = clip.with_effects([vfx.Resize(height=height)])

        # Write the output video
        clip_resized.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            logger=None
        )

        clip.close()
        clip_resized.close()
        return True

    except Exception as e:
        logging.error(f"Error resizing video: {e}")
        return False


def stitch_video(
    video_path: Union[str, Path],
    plan_path: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs
) -> Optional[Union[str, Path]]:
    """
    Stitch together highlight clips using MoviePy.

    Args:
        video_path (str or Path): Path to source video file.
        plan_path (str or Path): Path to reasoning JSON (FinalHighlightResult.json).
        output_path (str or Path): Output filename.
        **kwargs: Optional parameters:
            - transition (str): 'cut' or 'fade' (default: 'cut')
            - speed_ramp (bool): Add speed ramp effect (default: False)
            - min_clip_s (float): Minimum clip duration in seconds (default: 0.5)
            - resolution (int): Output height in pixels (default: 720)
            - dry_run (bool): Just print clips without processing (default: False)

    Returns:
        str or Path or None: Path to the output video, or None on failure.
    """
    # Set defaults for optional parameters
    transition = kwargs.get('transition', 'cut')
    speed_ramp = kwargs.get('speed_ramp', False)
    min_clip_s = kwargs.get('min_clip_s', 0.5)
    resolution = kwargs.get('resolution', 720)
    dry_run = kwargs.get('dry_run', False)
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Load & Validate Inputs
    video_path = Path(video_path)
    plan_path = Path(plan_path)

    if not video_path.is_file():
        logging.critical(f"Input video not found: {video_path}")
        return None
    
    if not plan_path.is_file():
        logging.critical(f"Input plan not found: {plan_path}")
        return None

    with open(plan_path, encoding='utf-8') as f:
        plan = json.load(f)
    selected_clips = plan.get('SelectedClips', [])
    
    # Get video info
    video_info = get_video_info(video_path)
    logging.info(f"Video info: {video_info}")

    # Clean and Prepare Clips
    cleaned_clips = []
    for clip in selected_clips:
        start = max(0, clip.get('StartTimeMs', 0) / 1000)
        end = min(video_info['duration'], clip.get('EndTimeMs', 0) / 1000)
        if end - start < min_clip_s:
            logging.warning(f"Clip {clip.get('SegmentId')} too short, skipping.")
            continue
        cleaned_clips.append({**clip, 'start': start, 'end': end})

    # Sort clips by start time
    cleaned_clips.sort(key=lambda c: c['start'])

    if not cleaned_clips:
        logging.error('No valid clips to process.')
        return None

    # Dry run - just print clips
    if dry_run:
        for c in cleaned_clips:
            print(f"{c['SegmentId']}: {c['start']}s → {c['end']}s")
        return None

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        clip_files = []
        
        # Extract individual clips
        for i, clip_spec in enumerate(cleaned_clips):
            clip_file = temp_dir / f"clip_{i:03d}.mp4"
            
            # Apply speed ramp effect if requested
            if speed_ramp and clip_spec['end'] - clip_spec['start'] > 2.0:
                # Split clip: normal speed for first 75%, half speed for last 25%
                duration = clip_spec['end'] - clip_spec['start']
                split_point = clip_spec['start'] + duration * 0.75
                
                # Extract normal speed part
                normal_file = temp_dir / f"clip_{i:03d}_normal.mp4"
                if extract_clip(video_path, clip_spec['start'], split_point, normal_file):
                    # Extract slow motion part
                    slow_file = temp_dir / f"clip_{i:03d}_slow.mp4"
                    if extract_clip(video_path, split_point, clip_spec['end'], slow_file, speed_factor=0.5):
                        # Concatenate normal + slow parts
                        if concatenate_videos([normal_file, slow_file], clip_file):
                            clip_files.append(clip_file)
                        else:
                            logging.warning(f"Failed to create speed ramp for clip {i}, using normal speed")
                            if extract_clip(video_path, clip_spec['start'], clip_spec['end'], clip_file):
                                clip_files.append(clip_file)
                    else:
                        # Fallback to normal speed
                        if extract_clip(video_path, clip_spec['start'], clip_spec['end'], clip_file):
                            clip_files.append(clip_file)
                else:
                    logging.error(f"Failed to extract clip {i}")
            else:
                # Normal clip extraction
                if extract_clip(video_path, clip_spec['start'], clip_spec['end'], clip_file):
                    clip_files.append(clip_file)
                else:
                    logging.error(f"Failed to extract clip {i}")
        
        if not clip_files:
            logging.error("No clips were successfully extracted")
            return None
        
        # Apply fade transitions if requested
        if transition == 'fade' and len(clip_files) > 1:
            fade_files = []
            for i, clip_file in enumerate(clip_files):
                fade_file = temp_dir / f"fade_{i:03d}.mp4"
                
                # Add fade effects (except first and last get partial fades)
                fadein = 0.5 if i > 0 else 0
                fadeout = 0.5 if i < len(clip_files) - 1 else 0
                
                if add_fade_effects(clip_file, fade_file, fadein, fadeout):
                    fade_files.append(fade_file)
                else:
                    fade_files.append(clip_file)  # Use original if fade fails
            
            clip_files = fade_files
        
        # Concatenate all clips
        temp_output = temp_dir / "concatenated.mp4"
        if not concatenate_videos(clip_files, temp_output):
            logging.error("Failed to concatenate clips")
            return None
        
        # Apply final processing (resize if needed)
        final_output = Path(output_path)
        if resolution and resolution != video_info['height']:
            if not resize_video(temp_output, final_output, resolution):
                logging.error("Failed to resize video")
                return None
        else:
            # Just copy the file
            shutil.copy2(temp_output, final_output)
    
    logging.info(f"Highlight video saved to {output_path}")
    return output_path
