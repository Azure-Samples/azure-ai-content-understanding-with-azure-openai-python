import argparse
import json
import pathlib
import sys
import logging
from moviepy.editor import VideoFileClip, TextClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip, CompositeAudioClip
import PIL
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QFileDialog, 
                             QCheckBox, QComboBox, QSlider, QProgressBar,
                             QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from moviepy.video.fx import all as vfx

# Patch for MoviePy compatibility with Pillow >=10.0.0
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# 1. Argument Parsing
parser = argparse.ArgumentParser(description="Stitch together highlight clips based on model output.")
parser.add_argument('--video', required=False, default=r'C:\Users\t-kjindel\Downloads\videoplayback (3).mp4', help='Path to source video file')
parser.add_argument('--plan', required=False, default=r'C:\Users\t-kjindel\OneDrive - Microsoft\Desktop\Highlights Generation\final_highlight_result.json', help='Path to reasoning JSON (final_highlight_result.json)')
parser.add_argument('--out', default='highlight.mp4', help='Output filename')
parser.add_argument('--captions', dest='captions', action='store_true')
parser.add_argument('--no-captions', dest='captions', action='store_false')
parser.set_defaults(captions=True)
parser.add_argument('--transition', choices=['cut', 'fade'], default='cut')
parser.add_argument('--bg-music', help='Path to background music (optional)')
parser.add_argument('--music-vol', type=float, default=0.2, help='Background music volume (0-1)')
parser.add_argument('--speed-ramp', dest='speed_ramp', action='store_true')
parser.add_argument('--no-speed-ramp', dest='speed_ramp', action='store_false')
parser.set_defaults(speed_ramp=False)
parser.add_argument('--min-clip-s', type=float, default=0.5)
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--resolution', type=int, choices=[720, 1080], default=720)
parser.add_argument('--fps', type=int, help='Frames per second')
args = parser.parse_args()

def stitch_video(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 2. Load & Validate Inputs
    video_path = pathlib.Path(args.video)
    plan_path = pathlib.Path(args.plan)

    if not video_path.is_file():
        logging.critical(f"Input video not found: {args.video}")
        # Use return instead of sys.exit for library usage
        return
    
    if not plan_path.is_file():
        logging.critical(f"Input plan not found: {args.plan}")
        return

    with open(plan_path, encoding='utf-8') as f:
        plan = json.load(f)
    selected_clips = plan.get('SelectedClips', [])
    # Explicitly set audio=True to ensure audio is loaded
    try:
        video = VideoFileClip(str(video_path), audio=True)
        if video.audio is None:
            logging.warning("Source video doesn't contain audio or audio track could not be loaded.")
    except Exception as e:
        logging.error(f"Error loading video with audio: {str(e)}")
        # Try again without audio as fallback
        video = VideoFileClip(str(video_path), audio=False)
        logging.warning("Loaded video without audio due to error.")

    # 3. Clean and Prepare Clips
    cleaned_clips = []
    for clip in selected_clips:
        start = max(0, clip.get('StartTimeMs', 0) / 1000)
        end = min(video.duration, clip.get('EndTimeMs', 0) / 1000)
        if end - start < args.min_clip_s:
            logging.warning(f"Clip {clip.get('SegmentId')} too short, skipping.")
            continue
        # Pass through all fields for make_subclip
        cleaned_clips.append({**clip, 'start': start, 'end': end})

    # Sort clips by start time to check for overlaps/gaps
    cleaned_clips.sort(key=lambda c: c['start'])

    # --- Overlap/Gap Detection ---
    GAP_THRESHOLD_S = 1.0 # Warn if gap is larger than 1 second
    if cleaned_clips:
        for i in range(1, len(cleaned_clips)):
            current_clip = cleaned_clips[i]
            prev_clip = cleaned_clips[i-1]
            
            overlap = prev_clip['end'] - current_clip['start']
            if overlap > 0:
                logging.warning(f"Overlap detected between clip {prev_clip.get('SegmentId')} (ends at {prev_clip['end']:.2f}s) and clip {current_clip.get('SegmentId')} (starts at {current_clip['start']:.2f}s). Overlap: {overlap:.2f}s")

            gap = current_clip['start'] - prev_clip['end']
            if gap > GAP_THRESHOLD_S:
                logging.warning(f"Large gap detected between clip {prev_clip.get('SegmentId')} (ends at {prev_clip['end']:.2f}s) and clip {current_clip.get('SegmentId')} (starts at {current_clip['start']:.2f}s). Gap: {gap:.2f}s")

    if not cleaned_clips:
        logging.error('No valid clips to process.')
        # Use return instead of sys.exit
        return

    # 4. Subclip Factory

    def make_subclip(clip_spec, video_obj, current_args):
        try:
            # Ensure audio is included when creating subclips
            clip = video_obj.subclip(clip_spec['start'], clip_spec['end'])
            
            # Verify clip is valid
            if clip is None or not hasattr(clip, 'duration'):
                logging.error(f"Invalid clip generated for segment {clip_spec.get('SegmentId')}")
                return None
                
            # Debug audio presence
            if clip.audio is None and video_obj.audio is not None:
                logging.warning(f"Audio missing in subclip for segment {clip_spec.get('SegmentId')}, attempting to restore")
                try:
                    # Try to extract audio from source video for this segment
                    audio_clip = video_obj.audio.subclip(clip_spec['start'], clip_spec['end'])
                    clip = clip.set_audio(audio_clip)
                except Exception as audio_e:
                    logging.error(f"Failed to restore audio for segment {clip_spec.get('SegmentId')}: {audio_e}")
                
            if current_args.speed_ramp:
                # Slow-mo last 25%
                dur = clip.duration
                if dur > 0: # Prevent error on zero-duration clips
                    ramp_start = dur * 0.75
                    normal = clip.subclip(0, ramp_start)
                    slow = clip.subclip(ramp_start, dur).fx(vfx.speedx, 0.5)
                    
                    # Verify both subclips are valid before concatenating
                    if normal is None or not hasattr(normal, 'duration') or slow is None or not hasattr(slow, 'duration'):
                        logging.error(f"Invalid subclips for speed ramping in segment {clip_spec.get('SegmentId')}")
                        return clip  # Return original clip if speed ramping fails
                    
                    # Use simple concatenation for internal speed ramping without transitions
                    try:
                        clip = concatenate_videoclips([normal, slow], method="compose")
                    except Exception as e:
                        logging.error(f"Error during speed ramping concatenation: {e}")
                        return clip  # Return original clip if concatenation fails
            return clip
        except Exception as e:
            logging.error(f"Failed to process clip {clip_spec.get('SegmentId')}: {e}")
            return None

    # 5. Build Clip List
    clips = []
    for clip_spec in cleaned_clips:
        subclip = make_subclip(clip_spec, video, args)
        if subclip:
            clips.append(subclip)

    if not clips:
        logging.error('All clips failed to process.')
        # Use return instead of sys.exit
        return
        
    # Extra validation: ensure all elements in clips are valid video clips
    valid_clips = []
    for i, clip in enumerate(clips):
        if clip is None or not hasattr(clip, 'duration'):
            logging.error(f"Removing invalid clip at position {i}")
            continue
        valid_clips.append(clip)
    
    # Update clips array with only valid clips
    clips = valid_clips
    
    if not clips:
        logging.error('No valid clips remain after validation.')
        return
        
    logging.info(f"Proceeding with {len(clips)} valid clips")

    # 6. Concatenate with Transitions
    fade_dur = 0.5
    
    # Log audio status of clips before concatenation
    for i, clip in enumerate(clips):
        has_audio = clip.audio is not None
        logging.info(f"Clip {i}: has_audio={has_audio}")
    
    # Handle the case when there's only one clip
    if len(clips) == 1:
        logging.info("Only one clip available, skipping transitions")
        final = clips[0]
    else:
        if args.transition == 'fade':
            # Fix: Use a proper crossfadeout/crossfadein function instead of passing fade_dur directly
            from moviepy.video.compositing.transitions import crossfadeout, crossfadein
            
            try:
                # Define the crossfade function with proper error handling
                def crossfade_clips(clip1, clip2):
                    # Ensure both clips have proper duration attribute
                    if not hasattr(clip1, 'duration') or not hasattr(clip2, 'duration'):
                        logging.warning("Clips don't have proper duration attribute, using direct concatenation")
                        return clip1.set_end(clip1.end).set_start(0) + clip2.set_start(clip1.duration)
                    
                    # Apply crossfade if possible
                    try:
                        return crossfadeout(clip1, fade_dur) + crossfadein(clip2, fade_dur)
                    except Exception as e:
                        logging.warning(f"Crossfade failed: {e}, falling back to direct concatenation")
                        return clip1.set_end(clip1.end).set_start(0) + clip2.set_start(clip1.duration)
                
                # Explicitly enable audio in concatenation
                final = concatenate_videoclips(clips, method='compose', transition=crossfade_clips)
            except Exception as e:
                logging.error(f"Error with fade transition: {e}. Falling back to cut transition.")
                final = concatenate_videoclips(clips, method='compose')
        else:
            # Explicitly enable audio in concatenation
            final = concatenate_videoclips(clips, method='compose')

    # 7. Audio Mixing
    # (Background music removed as requested)
    # if args.bg_music:
    #     bg = AudioFileClip(args.bg_music).volumex(args.music_vol).set_duration(final.duration)
    #     final_audio = CompositeAudioClip([final.audio.volumex(0.6), bg])
    #     final = final.set_audio(final_audio)

    # 8. Dry-Run & Logging
    if args.dry_run:
        for c in cleaned_clips:
            print(f"{c['SegmentId']}: {c['start']}s → {c['end']}s")
        # Use return instead of sys.exit
        return

    # 9. Export
    if args.resolution:
        final = final.resize(height=args.resolution)

    # Ensure audio is present in final video
    if final.audio is None and video.audio is not None:
        logging.warning("Audio missing in final video. Attempting to restore audio...")
        try:
            # Need to re-open the video to get the audio
            with VideoFileClip(str(video_path), audio=True) as video_handle:
                if video_handle.audio is not None:
                    # Create a composite audio from original audio segments
                    audio_clips = []
                    for clip_spec in cleaned_clips:
                        try:
                            audio_clip = video_handle.audio.subclip(clip_spec['start'], clip_spec['end'])
                            audio_clips.append(audio_clip)
                        except Exception as e:
                            logging.error(f"Error extracting audio for segment: {e}")
                    
                    if audio_clips:
                        from moviepy.audio.AudioClip import concatenate_audioclips
                        try:
                            final_audio = concatenate_audioclips(audio_clips)
                            final = final.set_audio(final_audio)
                            logging.info("Successfully restored audio from original segments")
                        except Exception as e:
                            logging.error(f"Error concatenating audio clips: {e}")
                            # Fall back to using the entire video's audio
                            final = final.set_audio(video_handle.audio)
                    else:
                        # Fall back to using the entire video's audio
                        final = final.set_audio(video_handle.audio)
                else:
                    logging.warning("Source video doesn't have audio")
        except Exception as e:
            logging.error(f"Error restoring audio: {str(e)}")

    # Log audio status before writing
    has_audio = final.audio is not None
    logging.info(f"Final video has_audio={has_audio}")

    # Enable verbose output for ffmpeg to debug audio issues
    final.write_videofile(
        args.out,
        codec="libx264",
        audio_codec="aac",
        fps=args.fps or video.fps,
        threads=4,
        preset="medium",
        ffmpeg_params=["-crf", "23"],
        audio=True,
        temp_audiofile='temp-audio.m4a',
        remove_temp=True,
        verbose=True,
        logger=None  # Use default logger to see ffmpeg output
    )
    # 10. Done
    logging.info(f"Highlight video saved to {args.out}")
    video.close()


class StitcherThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        try:
            # This is a simplified progress reporting.
            # For accurate progress, moviepy's logger needs to be parsed.
            self.progress.emit(10)
            stitch_video(self.args)
            self.progress.emit(100)
            self.finished.emit(f"Success! Video saved to {self.args.out}")
        except Exception as e:
            self.error.emit(str(e))


class StitcherUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Video Highlight Stitcher')
        self.setGeometry(100, 100, 500, 400)

        layout = QVBoxLayout()

        # File selection
        layout.addLayout(self.create_file_selector('Video File:', 'video_path', r'C:\Users\t-kjindel\Downloads\videoplayback (3).mp4'))
        layout.addLayout(self.create_file_selector('Plan File:', 'plan_path', r'C:\Users\t-kjindel\OneDrive - Microsoft\Desktop\Highlights Generation\final_highlight_result.json'))
        layout.addLayout(self.create_file_saver('Output File:', 'out_path', 'highlight.mp4'))

        # Options
        self.captions_check = QCheckBox('Add Captions', self)
        self.captions_check.setChecked(True)
        layout.addWidget(self.captions_check)

        self.speed_ramp_check = QCheckBox('Add Speed Ramp Effect', self)
        layout.addWidget(self.speed_ramp_check)

        # Transition
        self.transition_combo = QComboBox(self)
        self.transition_combo.addItems(['cut', 'fade'])
        layout.addWidget(QLabel('Transition:'))
        layout.addWidget(self.transition_combo)
        
        # Resolution
        self.resolution_combo = QComboBox(self)
        self.resolution_combo.addItems(['720', '1080'])
        layout.addWidget(QLabel('Resolution (height):'))
        layout.addWidget(self.resolution_combo)

        # Start button
        self.start_button = QPushButton('Generate Highlight Video', self)
        self.start_button.clicked.connect(self.start_stitching)
        layout.addWidget(self.start_button)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def create_file_selector(self, label, path_attr, default_path=""):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        path_edit = QLineEdit(default_path, self)
        setattr(self, path_attr, path_edit)
        layout.addWidget(path_edit)
        button = QPushButton('Browse', self)
        button.clicked.connect(lambda: self.browse_file(path_edit))
        layout.addWidget(button)
        return layout

    def create_file_saver(self, label, path_attr, default_path=""):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        path_edit = QLineEdit(default_path, self)
        setattr(self, path_attr, path_edit)
        layout.addWidget(path_edit)
        button = QPushButton('Browse', self)
        button.clicked.connect(lambda: self.save_file(path_edit))
        layout.addWidget(button)
        return layout

    def browse_file(self, path_edit):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "All Files (*)")
        if fname:
            path_edit.setText(fname)

    def save_file(self, path_edit):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save file', '', "MP4 Files (*.mp4)")
        if fname:
            path_edit.setText(fname)

    def start_stitching(self):
        # Simple namespace to hold the arguments
        class Args:
            pass
        
        args = Args()
        args.video = self.video_path.text()
        args.plan = self.plan_path.text()
        args.out = self.out_path.text()
        args.captions = self.captions_check.isChecked()
        args.transition = self.transition_combo.currentText()
        args.speed_ramp = self.speed_ramp_check.isChecked()
        args.resolution = int(self.resolution_combo.currentText())
        # Add other args from the parser with default values
        args.bg_music = None
        args.music_vol = 0.2
        args.min_clip_s = 0.5
        args.dry_run = False
        args.fps = None

        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)

        self.thread = StitcherThread(args)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_finished(self, message):
        self.start_button.setEnabled(True)
        QMessageBox.information(self, "Success", message)

    def on_error(self, message):
        self.start_button.setEnabled(True)
        QMessageBox.critical(self, "Error", message)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = StitcherUI()
    ex.show()
    sys.exit(app.exec_())