import dataclasses
import enum
import json
import logging
import os
import random

import cv2
import moviepy  # type: ignore
import numpy as np

from . import speaker_assigner
from ...flow import process_node
from ..utils import interval_scanner
from ..utils import movie_compiler

from typing import Any, override


class _Alignment(enum.Enum):
    LEFT = 1
    RIGHT = 2
    CENTER = 3


@dataclasses.dataclass
class _RenderWord:
    word: str
    color: tuple[int, int, int]
    underline: bool


def _cv2_aligned_text(
    img,
    text: str,
    anchor_point,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    color=(0, 0, 0),
    thickness=2,
    horizontal_alignment: _Alignment = _Alignment.RIGHT,
    shadow_color: tuple[int, int, int] | None = None,
) -> None:
    _cv2_per_word_text(
        img,
        [_RenderWord(word=text, color=color, underline=False)],
        anchor_point,
        font,
        font_scale,
        thickness,
        horizontal_alignment,
        shadow_color,
    )


def _cv2_per_word_text(
    img,
    words: list[_RenderWord],
    anchor_point,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    thickness=2,
    horizontal_alignment: _Alignment = _Alignment.RIGHT,
    shadow_color: tuple[int, int, int] | None = None,
) -> None:
    text = " ".join(word.word for word in words)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    match horizontal_alignment:
        case _Alignment.LEFT:
            x = anchor_point[0]
        case _Alignment.RIGHT:
            x = anchor_point[0] - text_width
        case _Alignment.CENTER:
            x = anchor_point[0] - text_width // 2
        case _:
            raise ValueError(f"Unknown alignment: {horizontal_alignment}")
    y = anchor_point[1] + text_height // 2

    def make_text(text, coords, color):
        cv2.putText(
            img,
            text,
            coords,
            font,
            font_scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    text_so_far = ""
    for word in words:
        (pre_width, _), _ = cv2.getTextSize(text_so_far, font, font_scale, thickness)

        if shadow_color is not None:
            make_text(word.word, (x + pre_width + 3, y + 3), shadow_color)
        make_text(word.word, (x + pre_width, y), word.color)

        if word.underline:
            (upto_width, _), _ = cv2.getTextSize(
                text_so_far + word.word, font, font_scale, thickness
            )
            cv2.line(
                img,
                (x + pre_width, y + text_height - baseline),
                (x + upto_width, y + text_height - baseline),
                word.color,
                thickness,
            )

        text_so_far += word.word + " "

    return img


def _brighten_color(rgb, factor=0.3):
    r, g, b = rgb
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return (r, g, b)


class _Visualizer:
    def __init__(
        self,
        captions: list[speaker_assigner.SpeakerCaptionT],
        diarization: list[dict[str, Any]],
        speaker_roles: dict[str, str],
    ):
        self._captions = interval_scanner.IntervalScanner(captions)
        self._diarization: dict[str, list[tuple[float, float]]] = {}
        for item in diarization:
            speaker = item["speaker"]
            interval = item["interval"]
            if speaker not in self._diarization:
                self._diarization[speaker] = []
            self._diarization[speaker].append(interval)

        # Has the current interval index whose end >= last scanned t.
        self._tracking_index: dict[str, int] = {}
        for speaker in self._diarization:
            self._diarization[speaker].sort()
            self._tracking_index[speaker] = 0

        self._speaker_roles: dict[str, str] = speaker_roles

        # Sort by 'Teacher', 'Student'; reverse to have 'Teacher' first.
        speakers: list[str] = list(self._diarization.keys())
        sorted_speakers: list[str] = sorted(
            speakers, key=lambda x: self._speaker_roles.get(x, ""), reverse=True
        )
        self._speaker_index: dict[str, int] = {
            speaker: i for i, speaker in enumerate(sorted_speakers)
        }

    def render(self, getframe, t: float):
        # copy() because getframe() is readonly.
        frame: np.ndarray = getframe(t).copy()  # type: ignore

        # Show the timestamp.
        cv2.putText(
            frame, f"{t:0.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0)
        )

        margin_right = 200
        first_speaker_center = 200
        speaker_gap = 50

        # Draw a vertical line at margin_right.
        cv2.line(
            frame,
            (margin_right, first_speaker_center - speaker_gap // 2),
            (
                margin_right,
                first_speaker_center
                + speaker_gap // 2
                + (len(self._tracking_index) - 1) * speaker_gap,
            ),
            # (170, 68, 0),
            (0, 255, 0),
            2,
        )

        pallette = [(230, 40, 40), (70, 70, 230)]
        # pallette = [(220, 85, 57), (255, 191, 71)]

        captions = self._captions.containing_timestamp(t)

        if captions:
            # If there are multiple, likely the overlap is just rounding error.
            # Take the last caption.
            caption = max(captions, key=lambda x: x["interval"][0])

            render_words: list[_RenderWord] = []
            for word in caption["words"]:
                if word["speaker"] not in self._speaker_index:
                    # This may happen if caption has unknown speaker.
                    color = (255, 255, 255)
                else:
                    # Use a slightly different color from diarization, to indicate
                    # they are different processes.
                    color = _brighten_color(
                        pallette[self._speaker_index[word["speaker"]]], 0.5
                    )
                render_words.append(
                    _RenderWord(
                        word=word["text"],
                        color=color,
                        underline=word["interval"][0] <= t <= word["interval"][1],
                    )
                )

            _cv2_per_word_text(
                frame,
                render_words,
                (frame.shape[1] // 2, frame.shape[0] - 100),
                shadow_color=(0, 0, 0),
                thickness=2,
                font_scale=0.8,
                horizontal_alignment=_Alignment.CENTER,
            )

        for speaker in self._tracking_index.keys():
            speaker_index = self._speaker_index[speaker]
            index = self._tracking_index[speaker]
            diarization = self._diarization[speaker]
            while index < len(diarization) and t >= diarization[index][1]:
                index += 1
            self._tracking_index[speaker] = index
            if index >= len(diarization):
                continue

            speaker_center_y = first_speaker_center + speaker_gap * speaker_index

            speaker_color = pallette[speaker_index]

            is_speaking = diarization[index][0] <= t <= diarization[index][1]

            if is_speaking:
                cv2.circle(
                    frame,
                    center=(margin_right, speaker_center_y),
                    radius=random.randint(6, 7),
                    color=speaker_color,
                    thickness=-1,
                )

            # Add speaker name as a right aligned text.
            _cv2_aligned_text(
                frame,
                self._speaker_roles.get(speaker, ""),
                (margin_right - 10, speaker_center_y),
                color=speaker_color,
                thickness=3 if is_speaking else 2,
                font_scale=0.8,
            )

            # Draw a line for each of the intervals.
            pixels_per_second = 20
            for interval in diarization[index:]:
                x1 = int((interval[0] - t) * pixels_per_second) + margin_right
                x2 = int((interval[1] - t) * pixels_per_second) + margin_right
                if x1 > frame.shape[1]:
                    break
                x1 = max(x1, margin_right)
                x2 = min(x2, frame.shape[1])
                cv2.line(
                    frame,
                    (x1, speaker_center_y),
                    (x2, speaker_center_y),
                    speaker_color,
                    3,
                )

        return frame


# Note: The following must be exported in bash for nvenc.
# export FFMPEG_BINARY="/usr/bin/ffmpeg"


class CaptionVisualizer(process_node.ProcessNode):
    @override
    def process(
        self,
        source_file: str,
        word_captions_file: str,
        diarization_file: str,
        identified_roles: dict[str, str],
        out_file_stem: str,
    ) -> str:
        out_file = out_file_stem + ".visualization.mp4"
        with open(word_captions_file) as f:
            captions: list[speaker_assigner.SpeakerCaptionT] = json.load(f)
        with open(diarization_file) as f:
            diarization = json.load(f)
        visualizer = _Visualizer(captions, diarization, identified_roles)

        video = moviepy.VideoFileClip(source_file)
        clip = video.transform(visualizer.render)

        # First few seconds to check teacher/student identification.
        clip = clip.subclipped(0, 30)

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        movie_compiler.save_video_clip(clip, out_file)
        logging.info(f"Written to {out_file!r}")
        return out_file
