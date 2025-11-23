from concurrent import futures
import dataclasses
import functools
import hashlib
import json
import logging
import os

import cv2
import moviepy
from numpy import typing as npt
import numpy as np
from PIL import Image
from PIL import ImageDraw

from . import interval_scanner
from . import movie_compiler_utils
from .. import video_config
from ..video_flow_nodes import ocr_detector
from ..video_flow_nodes import role_based_captioner

from typing import Callable, TypedDict


@dataclasses.dataclass
class CaptionOptions:
    position_prop: tuple[float, float]
    caption_width_prop: float
    # Anchor ref.: https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html
    anchor: str
    align: str
    # RGBA. E.g. (0, 0, 0, 192) to create a dark black.
    background_color: tuple[int, int, int, int]


@dataclasses.dataclass
class MovieOptions:
    """Class to define movie rendering options for the movie."""

    # Will be resized if wrong size.
    resize_to: tuple[int, int]
    # Caption positioning options.
    caption: CaptionOptions

    # Coords for the title.
    text_title_pos: tuple[float, float]
    # Coords for the description.
    text_desc_pos: tuple[float, float]
    # Title text and bar color.
    text_color: tuple[int, int, int]

    # Background color for the timer bar for clips.
    @property
    def bar_color(self) -> tuple[int, int, int]:
        result = tuple(int(c * 0.5) for c in self.text_color)
        assert len(result) == 3
        return result


# Number of seconds added to each end of a clip for fading.
DEFAULT_FADE_TIME = 0.5

# 1 worker ~15 it/s, 4 workers ~28 it/s combined.
_HIGHLIGHT_MAX_WORKERS = 4

# Number of threads for moviepy. This did not seem to make a difference in speed.
_SAVE_MAX_WORKERS = 1

# Length of silence to fast-forward.
_FFWD_MIN_LENGTH = 10.0
# How long after speech stops, or before speech ends, to start marking a silence.
_FFWD_SILENCE_MARGIN = 0.5
# Number of seconds to leave at the start or end of clip. Must be positive.
_FFWD_CLIP_MARGIN = 2.0
# Speed of fast forward.
_FFWD_SPEED = 2.5
# _FFWD_COLOR = (255, 235, 59)
_FFWD_COLOR = (255, 179, 0)  # FFB300


class HighlightsT(TypedDict):
    """Used to describe inputs for MovieCompiler."""

    description: str
    start_time: float
    end_time: float
    captions: list[role_based_captioner.RoleAwareCaptionT]


@functools.cache
def get_movie_duration(movie_path: str) -> float:
    video = moviepy.VideoFileClip(movie_path)
    assert isinstance(video.duration, float)
    return video.duration


def _get_temp_file_name() -> str:
    return video_config.random_temp_fname("_highlights_temp", ".mp4")


def save_video_clip(final_video: moviepy.VideoClip, output_file: str) -> None:
    final_video.write_videofile(
        output_file,
        # codec="libx264",  # Without nvidia.
        codec="h264_nvenc",  # Or, "nvenc_hevc" for H.265.
        audio_codec="aac",
        # threads=1,  # Seting 1 may avoid CPU threading confusion.
        preset="fast",  # Optional NVENC presets: "slow", "fast", "ll", "hq", etc.
        # See options in `ffmpeg -h encoder=h264_nvenc`.
        ffmpeg_params=[
            "-rc",  # Rate control for nvidia.
            "vbr",  # Cariable bitrate mode
            "-cq",
            "23",  # Constant quality.
            # "-b:v",
            # "0",  # Disable bitrate target for constant quality.
        ],
        threads=_SAVE_MAX_WORKERS,
    )


def _ffwd_silence(
    input_clip_file: str,
    start: float,
    end: float,
    captions: list[role_based_captioner.RoleAwareCaptionT],
):
    """Detects and fast forwards silence. Overwrites original file."""
    # Find the negative spaces in the captions.
    silences: list[tuple[float, float]] = []
    silence_start: float = 0
    for caption in captions:
        caption_start, caption_end = caption["interval"]
        caption_start -= _FFWD_SILENCE_MARGIN
        caption_end += _FFWD_SILENCE_MARGIN
        silence_end = caption_start
        # Append only between (start, end).
        if silence_end >= start and silence_start <= end:
            silences.append(
                (
                    max(start + _FFWD_CLIP_MARGIN, silence_start),
                    min(end - _FFWD_CLIP_MARGIN, silence_end),
                )
            )
        silence_start = caption_end
    silences.append((silence_start, end))

    # Keep only long silences - which need to be fast forwarded.
    silences = [x for x in silences if x[1] - x[0] >= _FFWD_MIN_LENGTH]

    # Return as is, with no ffwd, if there is no silence.
    if not silences:
        logging.info("No silence to ffwd.")
        return

    # Subtract `start` from all the times.
    silences = [(x[0] - start, x[1] - start) for x in silences]

    logging.info(f"Silence detected at: {silences}.")

    clip = moviepy.VideoFileClip(input_clip_file)

    speed_fx = moviepy.video.fx.MultiplySpeed(_FFWD_SPEED)  # type: ignore
    txt_ffwd = moviepy.TextClip(
        text=f"FFWD >> {_FFWD_SPEED:0.1f}x",
        font_size=36,
        color=f"rgb{_FFWD_COLOR}",
        margin=(10, 10, 10, 10),
    )
    # Create an array of subclips to cover the entire duration, but ffwd silences.
    subclips: list[moviepy.VideoClip] = []
    t = 0.0
    for silence_start, silence_end in silences:
        subclips.append(clip.subclipped(t, silence_start))
        clip_to_ffwd = clip.subclipped(silence_start, silence_end)
        sped_up_audio = movie_compiler_utils.audio_speed(
            clip_to_ffwd.audio, _FFWD_SPEED
        )
        sped_up = speed_fx.apply(clip_to_ffwd.with_audio(None))
        sped_up = sped_up.with_audio(sped_up_audio)
        sped_up = moviepy.CompositeVideoClip(
            [
                sped_up,
                txt_ffwd.with_duration(sped_up.duration).with_position(
                    ("right", "top")
                ),
            ]
        )
        subclips.append(sped_up)
        t = silence_end
    subclips.append(clip.subclipped(t, clip.duration))

    logging.info("Fast forwarding all silences...")
    final_clip = moviepy.concatenate_videoclips(subclips)
    temp_file2 = _get_temp_file_name()
    save_video_clip(final_clip, temp_file2)
    os.rename(temp_file2, input_clip_file)


class MovieCompiler:
    def __init__(self, options: MovieOptions) -> None:
        self._movie_options = options
        self._clip_files: list[str] = []
        os.makedirs(video_config.tempdir(), exist_ok=True)

        self._executor = futures.ThreadPoolExecutor(max_workers=_HIGHLIGHT_MAX_WORKERS)
        self._executor_tasks: list[futures.Future[None]] = []

    def _frame_hook(
        self,
        getframe,
        t: float,
        duration: float,
        scanner: interval_scanner.IntervalScanner[
            role_based_captioner.RoleAwareCaptionT
        ],
        start: float,
        options: MovieOptions,
        blur_scanner: (
            interval_scanner.IntervalScanner[ocr_detector.DetectionInterval] | None
        ),
        frame_processor: (
            Callable[[npt.NDArray[np.uint8], float], npt.NDArray[np.uint8]] | None
        ) = None,
    ) -> npt.NDArray[np.uint8]:
        # Note: The visualization node uses cv2. Here we use Pillow, to
        # demonstrate yet another way to render on movie frames.

        frame: npt.NDArray[np.uint8] = getframe(t).copy()  # type: ignore

        if blur_scanner is not None:
            # Apply blur.
            frame = movie_compiler_utils.do_blur(frame, start + t, blur_scanner)

        if frame_processor is not None:
            frame = frame_processor(frame, start + t)

        # Resize if the frame is not of the right size.
        required_size = options.resize_to
        if frame.shape[:2] != required_size:
            # Resize the frame. CUBIC takes slightly longer but is better for up-sizing
            # which is what we will typically do.
            frame = np.array(
                cv2.resize(frame, required_size, interpolation=cv2.INTER_CUBIC),
                dtype=np.uint8,
            )

        image = Image.fromarray(frame).convert("RGBA")
        draw = ImageDraw.Draw(image)

        # Starts from 0 like f(x) = x, but tapers off and never crosses max_bar_length.
        bar_length = frame.shape[1]
        bar_y = 0
        bar_width = 3

        draw.line(
            [(0, bar_y), (bar_length, bar_y)], fill=options.bar_color, width=bar_width
        )
        draw.line(
            [(0, bar_y), ((bar_length * t) // duration, bar_y)],
            fill=options.text_color,
            width=bar_width,
        )

        # A green-blue pallette for captions to be drawn with ImageDraw.
        caption_pallette = {
            "Teacher": (152, 251, 152),  # Pale green.
            "Student": (173, 216, 230),  # Light blue.
        }

        captions = scanner.containing_timestamp(t + start)
        if captions:
            caption = max(captions, key=lambda x: x["interval"][0])
            caption_color = caption_pallette.get(caption["speaker"], (255, 255, 255))
            caption_text = caption["text"]
            image = movie_compiler_utils.multiline_text(
                image,
                caption_text,
                caption_color,
                self._movie_options.caption.background_color,
                # Caption bounding and alignment parameters.
                # position_prop=(0.02, 0.92),  # Left, bottom.
                # caption_width_prop=0.3,  # 30% of the width.
                # anchor="ld",
                # align="left",
                position_prop=self._movie_options.caption.position_prop,
                caption_width_prop=self._movie_options.caption.caption_width_prop,
                anchor=self._movie_options.caption.anchor,
                align=self._movie_options.caption.align,
            )

        return np.array(image.convert("RGB"))

    def add_highlight_group(
        self,
        source_movie_file: str,
        title: str,
        highlights: list[HighlightsT],
    ) -> None:
        logging.info(f"Processing {source_movie_file}...")
        for index, highlight in enumerate(highlights):
            self.add_highlight(
                source_movie_file,
                title,
                highlight,
                title_fade_in=(index == 0),
                title_fade_out=(index == len(highlights) - 1),
            )
            if video_config.TESTING_MODE:
                if index >= 1:
                    break

    def add_highlight(
        self,
        source_movie_file: str,
        title: str,
        highlight: HighlightsT,
        title_fade_in: bool,
        title_fade_out: bool,
        fade_in_time: float = DEFAULT_FADE_TIME,
        fade_out_time: float = DEFAULT_FADE_TIME,
        temp_clip_hash: str | None = None,
        blur_json_file: str | None = None,
        # Optional hook to do arbitrary drawing, blurring, or other processing on the frame.
        frame_processor: (
            Callable[[npt.NDArray[np.uint8], float], npt.NDArray[np.uint8]] | None
        ) = None,
    ) -> None:
        logging.info(
            f"Processing highlight: {highlight['description']!r} [{highlight['start_time']} - {highlight['end_time']}]"
        )

        # Use a deterministic name for this particular source and highlight.
        # This will help us to ensure that the work will resume if movie
        # rendering is interrupted.
        movie_base_name = hashlib.sha256(
            f"{temp_clip_hash!r} {source_movie_file!r} {highlight!r}".encode("utf-8")
        ).hexdigest()[:16]
        output_file = str(video_config.tempdir() / f"{movie_base_name}.mp4")
        self._clip_files.append(output_file)
        logging.info(f"Temporary clip file: {output_file}")
        if os.path.isfile(output_file):
            # Exists from an interrupted session.
            logging.info(f"Re-using file from previous run: {output_file}")
            return

        future = self._executor.submit(
            self._add_highlight_async,
            source_movie_file=source_movie_file,
            title=title,
            highlight=highlight,
            title_fade_in=title_fade_in,
            title_fade_out=title_fade_out,
            fade_in_time=fade_in_time,
            fade_out_time=fade_out_time,
            output_file=output_file,
            blur_json_file=blur_json_file,
            frame_processor=frame_processor,
        )
        self._executor_tasks.append(future)

    def _add_highlight_async(
        self,
        source_movie_file: str,
        title: str,
        highlight: HighlightsT,
        title_fade_in: bool,
        title_fade_out: bool,
        fade_in_time: float,
        fade_out_time: float,
        output_file: str,
        blur_json_file: str | None = None,
        frame_processor: (
            Callable[[npt.NDArray[np.uint8], float], npt.NDArray[np.uint8]] | None
        ) = None,
    ) -> None:
        # Cannot print em-dash.
        highlight["description"] = highlight["description"].replace("\u2014", " - ")
        # Cannot print hyphen.
        highlight["description"] = highlight["description"].replace("\u2010", "-")

        logging.info("Creating subclip of the movie.")
        source_movie = moviepy.VideoFileClip(source_movie_file)
        logging.info(f"Source movie duration: {source_movie.duration}")
        start, end = (highlight["start_time"], highlight["end_time"])
        # Add padding to account for fade-in / fade-out effects.
        # Region excludes the end, i.e. [start, end). Especially important if fade_out is 0.
        fps = source_movie.fps
        assert isinstance(fps, float)
        start, end = (
            max(0, start - fade_in_time),
            end - 1 / fps / 2 + fade_out_time,
        )
        assert isinstance(source_movie.duration, float)
        if end >= source_movie.duration:
            logging.warning(f"Truncating {end=} to {source_movie.duration=}")
            end = source_movie.duration
        if end <= start + 0.05:
            logging.warning(f"Skipping bad segment: {(start, end)}")
            return

        blur_scanner: (
            interval_scanner.IntervalScanner[ocr_detector.DetectionInterval] | None
        ) = None
        if blur_json_file is not None:
            with open(blur_json_file, "r") as blur_file:
                blur_stream: list[ocr_detector.DetectionInterval] = json.load(blur_file)
            blur_scanner = interval_scanner.IntervalScanner(blur_stream)

        duration = end - start
        clip = source_movie.subclipped(start, end)

        clip = clip.transform(
            functools.partial(
                self._frame_hook,
                duration=duration,
                scanner=interval_scanner.IntervalScanner(highlight["captions"]),
                start=start,
                options=self._movie_options,
                blur_scanner=blur_scanner,
                frame_processor=frame_processor,
            )
        )

        fadein = moviepy.video.fx.FadeIn(fade_in_time)  # type: ignore
        fadeout = moviepy.video.fx.FadeOut(fade_out_time)  # type: ignore
        afadein = moviepy.audio.fx.AudioFadeIn(fade_in_time)  # type: ignore
        afadeout = moviepy.audio.fx.AudioFadeOut(fade_out_time)  # type: ignore
        clip = fadein.apply(clip)
        clip = fadeout.apply(clip)
        clip = afadein.apply(clip)
        clip = afadeout.apply(clip)

        logging.info("Creating text clip.")
        txt_desc = moviepy.TextClip(
            text=highlight["description"],
            # font="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
            font_size=36,
            color=f"rgb{self._movie_options.text_color}",
            margin=(5, 5, 5, 10),
            bg_color=self._movie_options.caption.background_color,
        )
        txt_desc = txt_desc.with_duration(clip.duration)
        txt_desc = fadein.apply(txt_desc)
        txt_desc = fadeout.apply(txt_desc)

        txt_title = moviepy.TextClip(
            text=title,
            # font="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
            font_size=36,
            color=f"rgb{self._movie_options.text_color}",
            margin=(5, 5, 0, 10),
            bg_color=self._movie_options.caption.background_color,
        )
        txt_title = txt_title.with_duration(clip.duration)
        if title_fade_in:
            txt_title = fadein.apply(txt_title)
        if title_fade_out:
            txt_title = fadeout.apply(txt_title)

        # Overlay the text on the video
        composite = moviepy.CompositeVideoClip(
            [
                clip,
                txt_title.with_position(self._movie_options.text_title_pos),
                txt_desc.with_position(self._movie_options.text_desc_pos),
            ]
        )

        # Since we do not want to resume from a half finished movie after
        # interruption, write to temporary file first and move once
        # succeeded.
        # Randomize it, so that this is thread safe.
        temp_file = _get_temp_file_name()
        save_video_clip(composite, temp_file)

        # moviepy does not compose effects well, so we operate on the saved file.
        _ffwd_silence(temp_file, start, end, highlight["captions"])

        os.rename(temp_file, output_file)
        logging.info(f"Wrote and renamed to {output_file!r}")
        self._log_tasks_status(add_one=True)

    def _log_tasks_status(self, add_one: bool) -> None:
        # One line log of how many done of how many total.
        done = sum(future.done() for future in self._executor_tasks)
        # Called from within a task, at the end.
        if add_one:
            done += 1
        logging.info(f"Movie compiler: done {done} of {len(self._executor_tasks)}")

    def _delete_clips(self) -> None:
        for fname in self._clip_files:
            os.remove(fname)

    def combine(self, out_file: str) -> None:
        logging.info("Waiting for task completion...")
        completed = 0
        for future in self._executor_tasks:
            future.result()
            completed += 1
            logging.info(f"Completed 1-{completed} of {len(self._executor_tasks)}")

        logging.info("Concatenating clips...")
        movie_compiler_utils.concatenate_movies(self._clip_files, out_file)
        self._delete_clips()
