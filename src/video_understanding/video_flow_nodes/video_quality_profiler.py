import json
import logging

import moviepy
import numpy as np
import numpy.typing as npt
from pyannote import audio  # type: ignore

from ...flow import process_node

from typing import override, TypedDict

# Difference than this will be considered same frame.
_RMSE_THRESHOLD = 0.0001

# How long a period in minimum is needed to be considered inadmissible.
_STILL_TIME_THRESHOLD = 2.5

# How much non_still time should count as good to stand on its own.
_NON_STILL_TIME_THRESHOLD = 5.0


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
            elif time - last_sgement[1] <= _NON_STILL_TIME_THRESHOLD:
                self._same_since = last_sgement[0]
                self._still_segments.pop()

        still_segment = (self._same_since, time)
        self._still_segments.append(still_segment)

    def get_bad_segments(self) -> list[BadSegment]:
        return [
            {
                "interval": segment,
                "reason": "choppiness",
            }
            for segment in self._still_segments
            if segment[1] - segment[0] > _STILL_TIME_THRESHOLD
        ]


class VideoQualityProfiler(process_node.ProcessNode):
    @override
    def process(self, source_file: str, out_file_stem: str) -> str:
        clip = moviepy.VideoFileClip(source_file)

        choppiness = _ChoppinessDetector()

        last_frame: npt.NDArray[np.uint8] | None = None

        frame_time: npt.NDArray[np.float64]
        frame: npt.NDArray[np.uint8]
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
                    logging.info(
                        f"frame: {frame_count} time: {frame_time:0.2f} RMSE: {rmse:0.5f}"
                    )
                    segements_for_logging = choppiness.get_bad_segments()
                    if segements_for_logging:
                        logging.info(f"Stillness so far: {segements_for_logging}")

            last_frame = frame

        out_file = out_file_stem + ".inadmissible_video_segments.json"
        with open(out_file, "w") as f:
            json.dump(choppiness.get_bad_segments(), f, indent=2)

        return out_file
