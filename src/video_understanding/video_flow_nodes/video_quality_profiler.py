import json
import logging
import time

import moviepy
import numpy as np
import numpy.typing as npt
from pyannote import audio  # type: ignore

from ...flow import process_node

from typing import override, TypedDict

# Difference than this will be considered same frame.
_RMSE_THRESHOLD = 0.0001

# Minimum stillness duration to be considered stutter.
_STUTTER_THRESHOLD = 0.3

# How much non_still time should count as good to stand on its own.
_NON_STILL_TIME_THRESHOLD = 10.0

# Ignore choppy segments if it is shorter than this length.
_CHOPPINESS_LENGTH_THRESHOLD = 1.5


class BadSegment(TypedDict):
    interval: tuple[float, float]
    reason: str


class _ChoppinessDetector:
    def __init__(self):
        self._still_segments: list[tuple[float, float]] = []
        self._same_since = 0

    def report_frame(self, time: float, is_same: bool) -> None:
        if not is_same:
            self._same_since = time
            return

        if self._still_segments:
            last_sgement = self._still_segments[-1]
            if last_sgement[0] == self._same_since:
                self._still_segments.pop()

        still_segment = (self._same_since, time)
        self._still_segments.append(still_segment)

    def get_bad_segments(self) -> list[BadSegment]:

        # First remove small choppy segments.
        long_choppy_segments = (
            segment
            for segment in self._still_segments
            if segment[1] - segment[0] > _STUTTER_THRESHOLD
        )

        # Then merge segments with short gaps of non-choppiness.
        processed_segments: list[tuple[float, float]] = []
        for segment in long_choppy_segments:
            if processed_segments:
                last_segment = processed_segments[-1]
                # Check if the gap of non-choppiness is too small.
                if segment[0] - last_segment[1] <= _NON_STILL_TIME_THRESHOLD:
                    # Merge with previous.
                    segment = (last_segment[0], segment[1])
                    # Drop the last segment, as we will add merged segment.
                    processed_segments.pop()

            processed_segments.append(segment)

        # Then drop all the short choppy sections.
        processed_segments = [
            segment
            for segment in processed_segments
            if segment[1] - segment[0] > _CHOPPINESS_LENGTH_THRESHOLD
        ]

        return [
            {
                "interval": segment,
                "reason": "choppiness",
            }
            for segment in processed_segments
        ]


class VideoQualityProfiler(process_node.ProcessNode):
    @override
    def process(self, source_file: str, out_file_stem: str) -> str:
        clip = moviepy.VideoFileClip(source_file)

        choppiness = _ChoppinessDetector()

        last_frame: npt.NDArray[np.uint8] | None = None

        frame_time: np.float64
        frame: npt.NDArray[np.uint8]
        start_time = time.time()
        for frame_count, (frame_time, frame) in enumerate(
            clip.iter_frames(with_times=True)
        ):
            if last_frame is not None:
                # Check if the video frames are the same or very similar.
                # Compute RMSE.
                rmse = ((frame - last_frame) ** 2).mean() ** 0.5
                choppiness.report_frame(
                    float(frame_time), is_same=rmse < _RMSE_THRESHOLD
                )
                if frame_count % 100 == 0:
                    speed_factor = frame_time / max(time.time() - start_time, 0.001)
                    logging.info(
                        f"frame: {frame_count} time: {frame_time:0.2f}/{clip.duration}"
                        f" Speed: {speed_factor:0.2f}x Last RMSE: {rmse:0.5f}"
                    )
                    segements_for_logging = choppiness.get_bad_segments()
                    if segements_for_logging:
                        logging.info(f"So far: {segements_for_logging}")

            last_frame = frame

        out_file = out_file_stem + ".inadmissible_video_segments.json"
        with open(out_file, "w") as f:
            json.dump(choppiness.get_bad_segments(), f, indent=2)
            logging.info(f"Video Quality Profile written to {out_file!r}")

        return out_file
