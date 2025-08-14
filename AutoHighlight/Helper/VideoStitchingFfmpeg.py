import json
import pathlib
import logging
import subprocess
import tempfile
import os
import shutil
from datetime import datetime

def get_video_info(video_path):
    """Get video duration and basic info using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', 
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        audio_stream = None
        for stream in info['streams']:
            if stream['codec_type'] == 'video' and video_stream is None:
                video_stream = stream
            elif stream['codec_type'] == 'audio' and audio_stream is None:
                audio_stream = stream
        
        duration = float(info['format']['duration'])
        fps = eval(video_stream['r_frame_rate']) if video_stream else 30
        width = video_stream['width'] if video_stream else 1920
        height = video_stream['height'] if video_stream else 1080
        has_audio = audio_stream is not None
        
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

def extract_clip(video_path, start_time, end_time, output_path, speed_factor=1.0):
    """Extract a clip from video using ffmpeg"""
    try:
        duration = end_time - start_time
        
        # Base ffmpeg command
        cmd = ['ffmpeg', '-y', '-i', str(video_path)]
        
        # Seek to start time and set duration
        cmd.extend(['-ss', str(start_time), '-t', str(duration)])
        
        # Apply speed change if needed
        if speed_factor != 1.0:
            video_filter = f"setpts={1/speed_factor}*PTS"
            audio_filter = f"atempo={speed_factor}"
            cmd.extend(['-filter:v', video_filter, '-filter:a', audio_filter])
        
        # Output settings
        cmd.extend([
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'medium',
            '-crf', '23',
            str(output_path)
        ])
        
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        logging.error(f"Error extracting clip {start_time}-{end_time}: {e}")
        return False

def add_fade_effects(video_path, output_path, fadein_duration=0.5, fadeout_duration=0.5):
    """Add fade in/out effects to a video"""
    try:
        video_info = get_video_info(video_path)
        duration = video_info['duration']
        
        # Create fade filters
        fade_filters = []
        if fadein_duration > 0:
            fade_filters.append(f"fade=t=in:st=0:d={fadein_duration}")
        if fadeout_duration > 0:
            fade_start = duration - fadeout_duration
            fade_filters.append(f"fade=t=out:st={fade_start}:d={fadeout_duration}")
        
        filter_str = ",".join(fade_filters)
        
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-filter:v', filter_str,
            '-c:a', 'copy',  # Copy audio without re-encoding
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        logging.error(f"Error adding fade effects: {e}")
        return False

def concatenate_videos(video_paths, output_path):
    """Concatenate multiple videos using ffmpeg"""
    try:
        # Create temporary file list for ffmpeg concat
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            concat_file = f.name
            for video_path in video_paths:
                f.write(f"file '{os.path.abspath(video_path)}'\n")
        
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file,
            '-c:v', 'libx264', '-c:a', 'aac', '-preset', 'medium', '-crf', '23',
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Clean up temp file
        os.unlink(concat_file)
        return True
    except Exception as e:
        logging.error(f"Error concatenating videos: {e}")
        if 'concat_file' in locals():
            try:
                os.unlink(concat_file)
            except:
                pass
        return False

def resize_video(video_path, output_path, height=720):
    """Resize video to specified height maintaining aspect ratio"""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-filter:v', f'scale=-2:{height}',
            '-c:a', 'copy',
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        logging.error(f"Error resizing video: {e}")
        return False

def stitch_video(video_path, plan_path, output_path, **kwargs):
    """
    Stitch together highlight clips using FFmpeg.
    
    Args:
        video_path (str): Path to source video file
        plan_path (str): Path to reasoning JSON (FinalHighlightResult.json)
        output_path (str): Output filename
        **kwargs: Optional parameters:
            - transition (str): 'cut' or 'fade' (default: 'cut')
            - speed_ramp (bool): Add speed ramp effect (default: False)
            - min_clip_s (float): Minimum clip duration in seconds (default: 0.5)
            - resolution (int): Output height in pixels (default: 720)
            - dry_run (bool): Just print clips without processing (default: False)
    """
    # Set defaults for optional parameters
    transition = kwargs.get('transition', 'cut')
    speed_ramp = kwargs.get('speed_ramp', False)
    min_clip_s = kwargs.get('min_clip_s', 0.5)
    resolution = kwargs.get('resolution', 720)
    dry_run = kwargs.get('dry_run', False)
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Load & Validate Inputs
    video_path = pathlib.Path(video_path)
    plan_path = pathlib.Path(plan_path)

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
        temp_dir = pathlib.Path(temp_dir)
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
        final_output = pathlib.Path(output_path)
        if resolution and resolution != video_info['height']:
            if not resize_video(temp_output, final_output, resolution):
                logging.error("Failed to resize video")
                return None
        else:
            # Just copy the file
            shutil.copy2(temp_output, final_output)
    
    logging.info(f"Highlight video saved to {output_path}")
    return output_path

def main():
    """Command-line interface for backward compatibility"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stitch together highlight clips using FFmpeg.")
    parser.add_argument('--video', required=True, help='Path to source video file')
    parser.add_argument('--plan', required=True, help='Path to reasoning JSON (FinalHighlightResult.json)')
    parser.add_argument('--out', default='highlight.mp4', help='Output filename')
    parser.add_argument('--transition', choices=['cut', 'fade'], default='cut')
    parser.add_argument('--speed-ramp', action='store_true', default=False)
    parser.add_argument('--min-clip-s', type=float, default=0.5)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resolution', type=int, choices=[720, 1080], default=720)
    
    args = parser.parse_args()
    
    return stitch_video(
        video_path=args.video,
        plan_path=args.plan,
        output_path=args.out,
        transition=args.transition,
        speed_ramp=args.speed_ramp,
        min_clip_s=args.min_clip_s,
        dry_run=args.dry_run,
        resolution=args.resolution
    )

if __name__ == '__main__':
    main()
