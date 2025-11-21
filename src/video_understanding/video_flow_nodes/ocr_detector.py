import json
import logging

import moviepy  # type: ignore
from numpy import typing as npt
import numpy as np

from ...flow import process_node
from ..detectors import detection_utils
from ..detectors import easyocr_custom
from ..detectors import tesseract_custom

from typing import override, TypedDict

# Tesseract requires hand-holding - and seems to miss text if it is colored.
# EasyOCR seems more accurate.
_EASYOCR = True

# How long to skip ahead if detected. We can afford to skip more, since we can still blur if the number goes off and comes back.
_DETECTED_SCAN_INTERVAL = 1.0
# How long to skip ahead if undetected.
_UNDETECTED_SCAN_INTERVAL = 0.5
# Note that we will blur around the detected interval a bit, which should nicely cover things up.
_REFINE_INTERVAL = 1 / 30.0


class DetectionInterval(TypedDict):
    interval: tuple[float, float]
    detections: list[detection_utils.Bbox]


_TimedDetectionT = tuple[float, list[detection_utils.DetectionResult]]


class OcrDetector(process_node.ProcessNode):
    def __init__(
        self,
    ):
        self._easyocr = None
        if _EASYOCR:
            self._easyocr = easyocr_custom.Detector()

    def _detect(
        self, clip: moviepy.VideoFileClip, t: float
    ) -> list[detection_utils.DetectionResult]:
        frame: npt.NDArray[np.uint8] = clip.get_frame(t)  # type: ignore
        if self._easyocr is not None:
            return list(self._easyocr.phone_numbers(frame))
        else:
            return list(tesseract_custom.iterate_phone_numbers(frame))

    def _detect_time_series(
        self,
        clip: moviepy.VideoFileClip,
    ) -> list[_TimedDetectionT]:
        timed_detections: list[_TimedDetectionT] = []
        t = 0.0
        assert isinstance(clip.duration, float)
        while t < clip.duration:
            timed_detections.append((t, self._detect(clip, t)))

            logging.info(
                f"OCR progress: t = {t} / {clip.duration} - detections so far: {sum(len(x[1]) for x in timed_detections)}"
            )
            if timed_detections[-1][1]:
                t += _DETECTED_SCAN_INTERVAL
            else:
                t += _UNDETECTED_SCAN_INTERVAL

        return timed_detections

    def _iterative_refine(
        self, clip: moviepy.VideoFileClip, timed_detections: list[_TimedDetectionT]
    ) -> None:
        # Scans through the detection list. Whenever two consecutive detections
        # are apart by _SCAN_TIME_SMALL at index i and i+1, inserts the middle,
        # starts the process again at ith index.
        i = 0
        while i < len(timed_detections) - 1:
            t1, detections1 = timed_detections[i]
            t2, detections2 = timed_detections[i + 1]
            if (
                t2 - t1 > _REFINE_INTERVAL
                and not detection_utils.result_list_almost_equal(
                    detections1, detections2
                )
            ):
                logging.info(f"Not equal: {(t1, detections1)}")
                logging.info(f"      And: {(t2, detections2)}")
                mid_time = (t1 + t2) / 2
                logging.info(f"Iterative refinement at {mid_time}")

                mid_detections = self._detect(clip, mid_time)
                timed_detections.insert(i + 1, (mid_time, mid_detections))
                # Continue at i to check for the newly inserted detection.
                continue
            i += 1

    @override
    def process(
        self,
        source_file: str,
        out_file_stem: str,
    ) -> str:
        movie = moviepy.VideoFileClip(source_file)

        timed_detections = self._detect_time_series(movie)
        self._iterative_refine(movie, timed_detections)

        results: list[DetectionInterval] = []
        for t, detections in timed_detections:
            if not results or not detection_utils.boxes_almost_equal(
                results[-1]["detections"], [detection.bbox for detection in detections]
            ):
                # Append new segment.
                results.append(
                    {
                        "interval": (t, t),
                        "detections": [detection.bbox for detection in detections],
                    }
                )
            else:
                # Extend previous.
                results[-1]["interval"] = (results[-1]["interval"][0], t)

        # Drop entries with no detections.
        results = [x for x in results if x["detections"]]

        out_file_name = out_file_stem + ".ocr_detections.json"

        logging.info(f"Writing to {out_file_name!r}")
        with open(out_file_name, "w") as f:
            json.dump(results, f)

        return out_file_name
