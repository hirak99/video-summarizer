# Self contained methods for drawing etc. used by movie_compiler.py.
import os
import subprocess
import tempfile

import cv2
import moviepy
from numpy import typing as npt
import numpy as np
from PIL import Image
from PIL import ImageDraw

from . import interval_scanner
from ..video_flow_nodes import ocr_detector


def audio_speed(clip: moviepy.AudioClip, speed: float) -> moviepy.AudioClip:
    """Change audio speed without changing pitch."""
    if speed == 1:
        return clip

    with tempfile.TemporaryDirectory(prefix="_audio_speedup_") as tempdir:
        fname = os.path.join(tempdir, "original.wav")
        clip.write_audiofile(fname)
        # Speedup with atempo.
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            fname,
            "-filter:a",
            f"atempo={speed}",
            os.path.join(tempdir, "speedup.wav"),
        ]
        subprocess.run(cmd, check=True)
        return moviepy.AudioFileClip(os.path.join(tempdir, "speedup.wav"))


def concatenate_movies(movie_paths: list[str], output_path: str) -> None:
    ffmpeg_bin = os.environ.get("FFMPEG_BINARY", "ffmpeg")

    # For the concatanating with "demuxer" method, we need to have instructions
    # in a text file.
    # See https://trac.ffmpeg.org/wiki/Concatenate#samecodec
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt"
    ) as list_file:
        for path in movie_paths:
            list_file.write(f"file '{os.path.abspath(path)}'\n")
        list_filename = list_file.name

    try:
        # Run ffmpeg concat demuxer.
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_filename,
            "-c",
            "copy",
            output_path,
        ]
        subprocess.run(cmd, check=True)
        print(f"Concatenation complete. Output saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution: {e}")
    finally:
        os.remove(list_filename)


def do_blur(
    frame: npt.NDArray[np.uint8],
    t: float,
    blur_scanner: interval_scanner.IntervalScanner[ocr_detector.DetectionInterval],
) -> npt.NDArray[np.uint8]:
    blur_intervals = blur_scanner.overlapping_intervals(max(0, t - 1), t + 1)
    if not blur_intervals:
        return frame

    for blur_interval in blur_intervals:

        # Create a simple subtle animation effect with the blur strength.
        start, end = blur_interval["interval"]

        # Determine how far the interval is from t, if t is not within it yet.
        # It is 0 if within the interval, otherwise a positive number.
        distance_to_interval = max(0, start - t, t - end)

        # Kernel size should be interpolated between 25 if distance is 0, and 15 if distance is 1.
        max_ksize = 25
        min_ksize = 15
        ksize = int(max_ksize + (min_ksize - max_ksize) * min(distance_to_interval, 1))
        # Kernel size must be odd.
        if ksize % 2 == 0:
            ksize += 1

        for det in blur_interval["detections"]:
            # Det has keys top, left, width, height. We want ROI for blur.
            x1, y1, w, h = det["left"], det["top"], det["width"], det["height"]
            x2, y2 = x1 + w, y1 + h
            # Apply Gaussian blur to the ROI.
            margin = 10
            x1 = max(x1 - margin, 0)
            y1 = max(y1 - margin, 0)
            x2 = min(x2 + margin, frame.shape[1])
            y2 = min(y2 + margin, frame.shape[0])
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(
                frame[y1:y2, x1:x2], (ksize, ksize), 0
            )

    return frame


def multiline_text(
    image: Image.Image,
    caption_text: str,
    caption_color: tuple[int, int, int],
    # The width prop and height prop, where the caption will be rendered.
    position_prop: tuple[float, float],
    caption_width_prop: float,
    anchor: str | None,
    align: str,
) -> Image.Image:
    caption_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    caption_draw = ImageDraw.Draw(caption_image)

    position = (
        int(image.width * position_prop[0]),
        int(image.height * position_prop[1]),
    )

    def caption_bbox(caption_text: str):
        return caption_draw.multiline_textbbox(
            position,
            caption_text,
            font_size=36,
            anchor=anchor,
            align=align,
        )

    # Function to split text into multiple lines based on maximum width
    def wrap_text(caption_text: str, max_width: int) -> str:
        words = caption_text.split(" ")
        lines = []
        current_line = ""

        for word in words:
            # Try to add the word to the current line
            test_line = current_line + (word if not current_line else " " + word)
            test_bbox = caption_bbox(test_line)

            # If adding this word exceeds max width, start a new line
            if test_bbox[2] - test_bbox[0] > max_width:
                if current_line:  # Don't add empty line
                    lines.append(current_line)
                current_line = word  # Start a new line with the current word
            else:
                current_line = test_line

        if current_line:  # Add any remaining text
            lines.append(current_line)

        return "\n".join(lines)

    # Get the width available for the text box
    max_caption_width = int(image.width * caption_width_prop)

    caption_text = wrap_text(caption_text, max_caption_width)

    # Update bbox with the wrapped caption text
    bbox = caption_bbox(caption_text)

    caption_draw.rectangle(
        [bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5], (0, 0, 0, 128)
    )

    # Draw caption text as subtitle.
    caption_draw.multiline_text(
        position,
        caption_text,
        fill=caption_color,
        font_size=36,
        anchor=anchor,
        align=align,
    )
    return Image.alpha_composite(image, caption_image)
