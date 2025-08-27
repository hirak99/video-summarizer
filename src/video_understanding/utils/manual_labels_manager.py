import logging
import os

import cv2
from numpy import typing as npt
import numpy as np

from . import interval_scanner
from . import manual_label_types
from .. import video_config

from typing import TypedDict

# For now, we only use the labels done by the user "hermes".
# Later on we can add support for replication etc.
_MANUAL_LABELER = "hermes"

# How much before start time does blur start.
_BLUR_BEFORE = 0.0
# How much is blur lingered after it ends.
_BLUR_AFTER = 0.0
# Blur padding in pixels. Should be 0 for precise manual labeling.
_BLUR_PADDING = 0

# Constants for annotation names that we will process. These must match with the labeling config.
_ANNOTATION_BLUR = "Blur"
_ANNOTATION_TEACHER = "Teacher Window"
_ANNOTATION_STUDENT = "Student Window"


def _labels_file(video_file: str) -> str:
    labels_file = os.path.join(
        video_config.MANUAL_LABELS_DIR, os.path.basename(video_file) + ".json"
    )
    return labels_file


def _load_labels_all_users(labels_file: str) -> manual_label_types.UserAnnotations:
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            return manual_label_types.UserAnnotations.model_validate_json(f.read())
    logging.info(f"No file {labels_file!r}")
    return manual_label_types.UserAnnotations(by_user={})


# Simple function to get labels from a JSON file
def _load_labels(labels_file: str) -> list[manual_label_types.AnnotationProps]:
    all_users = _load_labels_all_users(labels_file)
    if _MANUAL_LABELER in all_users.by_user:
        return all_users.by_user[_MANUAL_LABELER]
    return []


# Annotations for IntervalScanner.
class _AnnotationIntervalScanner(TypedDict):
    interval: tuple[float, float]
    label: manual_label_types.BoxLabel


class VideoAnnotation:
    def __init__(self, movie_path: str):
        self._labels_file = _labels_file(movie_path)
        self._labels = _load_labels(self._labels_file)
        logging.info(f"# manual annotations loaded: {len(self._labels)}")
        if not self._labels:
            logging.warning(f"No manual annotations loaded for {movie_path}.")

    def _get_scanner(
        self, ann_type: str
    ) -> interval_scanner.IntervalScanner[_AnnotationIntervalScanner]:
        annotations = [ann for ann in self._labels if ann.name == ann_type]

        intervals: list[_AnnotationIntervalScanner] = []
        for ann in annotations:
            interval = (ann.label.start, ann.label.end)
            intervals.append({"interval": interval, "label": ann.label})

        return interval_scanner.IntervalScanner(intervals)

    def get_teacher_scanner(
        self,
    ) -> interval_scanner.IntervalScanner[_AnnotationIntervalScanner]:
        return self._get_scanner(_ANNOTATION_TEACHER)

    def get_student_scanner(
        self,
    ) -> interval_scanner.IntervalScanner[_AnnotationIntervalScanner]:
        return self._get_scanner(_ANNOTATION_STUDENT)


class AnnotationBlur(VideoAnnotation):
    """Implements frame_hook() to process video frames using manual labels."""

    def __init__(self, movie_path: str):
        super().__init__(movie_path)

        # Set up the scanners one time to scan for intersecting annotations to blur.
        self._scanner_blur = self._get_scanner(_ANNOTATION_BLUR)
        self._scanner_teacher = self._get_scanner(_ANNOTATION_TEACHER)
        self._scanner_student = self._get_scanner(_ANNOTATION_STUDENT)

        # Cached blur image, to allow lazy computation of blurred image only if needed.
        self._blur_cache_t: float = -1.0
        self._blur_cache: cv2.typing.MatLike | None = None

    def _get_scanner(
        self, ann_type: str
    ) -> interval_scanner.IntervalScanner[_AnnotationIntervalScanner]:
        annotations = [ann for ann in self._labels if ann.name == ann_type]

        intervals: list[_AnnotationIntervalScanner] = []
        for ann in annotations:
            interval = (ann.label.start, ann.label.end)
            intervals.append({"interval": interval, "label": ann.label})

        return interval_scanner.IntervalScanner(intervals)

    def _handle_windows(
        self,
        frame: npt.NDArray[np.uint8],
        t: float,
    ) -> npt.NDArray[np.uint8]:
        """Handle teacher and student windows."""

        frame_has_no_labels = True

        # Black background.
        new_frame = np.zeros(frame.shape, dtype=np.uint8)
        for scanner in [self._scanner_teacher, self._scanner_student]:

            intervals = scanner.overlapping_intervals(
                max(0, t - _BLUR_BEFORE), t + _BLUR_AFTER
            )
            if not intervals:
                continue

            frame_has_no_labels = False

            # Just use the first, since there should be at most one teacher / student window.
            to_unblur = intervals[0]["label"]

            # Copy from the unblurred frame to the new_frame.
            x1, y1, w, h = (
                to_unblur.x,
                to_unblur.y,
                to_unblur.width,
                to_unblur.height,
            )
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
            new_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        # On no labels, we will not blur anything.
        # This is a workaround to handle incomplete labeling.
        # TODO: Drop this logic once we ensure full labeling.
        if frame_has_no_labels:
            return frame

        return np.array(new_frame, dtype=np.uint8)

    def _handle_blur(
        self,
        frame: npt.NDArray[np.uint8],
        t: float,
    ) -> npt.NDArray[np.uint8]:
        """Handle blur regions."""

        blur_intervals = self._scanner_blur.overlapping_intervals(
            max(0, t - _BLUR_BEFORE), t + _BLUR_AFTER
        )

        if not blur_intervals:
            # Skip the computation of blur entirely.
            return frame

        blurred_frame = self._get_blurred_frame(frame, t)

        for blur_interval in blur_intervals:
            ann_label = blur_interval["label"]
            x1, y1, w, h = ann_label.x, ann_label.y, ann_label.width, ann_label.height
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

            # Copy the area to be blurred from the blurred frame.
            x1 = max(x1 - _BLUR_PADDING, 0)
            y1 = max(y1 - _BLUR_PADDING, 0)
            x2 = min(x2 + _BLUR_PADDING, frame.shape[1])
            y2 = min(y2 + _BLUR_PADDING, frame.shape[0])

            # Substitute the blurred pixels back onto the original frame
            frame[y1:y2, x1:x2] = blurred_frame[y1:y2, x1:x2]

        return frame

    def _get_blurred_frame(
        self, frame: npt.NDArray[np.uint8], t: float
    ) -> cv2.typing.MatLike:
        """Return blurred(frame). Arg `t` is only used for caching."""

        if self._blur_cache_t == t and self._blur_cache is not None:
            return self._blur_cache

        # Blur radius as 10% of the width.
        ksize = int(frame.shape[1] * 0.1)
        # Kernel size must be odd.
        if ksize % 2 == 0:
            ksize += 1

        # Create a blurred version of the entire frame once
        self._blur_cache = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        self._blur_cache_t = t

        return self._blur_cache

    def process_frame(
        self,
        frame: npt.NDArray[np.uint8],
        t: float,
    ) -> npt.NDArray[np.uint8]:
        """Blur all annotations marked for blur on the given frame at time t."""

        if not self._labels:
            # Not ideal, but if annotations are not done do not blur anything.
            return frame

        frame = self._handle_blur(frame, t)
        frame = self._handle_windows(frame, t)

        return frame
