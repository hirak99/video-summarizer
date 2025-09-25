import collections
import logging
import os

import cv2
import numpy as np
import numpy.typing as npt
from pyannote import audio  # type: ignore
import ultralytics

_YOLO_WEIGHTS = os.path.expanduser(
    "~/code/bounding-box-model/_yolo_models/multi-class/weights/best.pt"
)

# The model was built with these class names.
_EXPECTED_CLASSES = ["teacher", "student"]


class _Detection:
    teacher: tuple[float, float, float, float] | None = None
    student: tuple[float, float, float, float] | None = None


class TeacherStudentDetector:
    def __init__(self) -> None:
        self._yolo = ultralytics.YOLO(_YOLO_WEIGHTS)

    def detect(self, frame: npt.NDArray[np.uint8], log_str: str) -> _Detection:
        detection = _Detection()

        # Note: There is an opportunity to remove this and speed up the
        # inference, if we just train on BGR instead of RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._yolo([frame_rgb], verbose=False)
        assert len(results) == 1  # Single image should result in single result.
        result = results[0]
        cls_id_counts = collections.Counter(int(x) for x in result.boxes.cls)
        for cls, box in zip(result.boxes.cls, result.boxes.xyxy):
            id = int(cls)
            cls_name = result.names[id]
            if cls_name not in _EXPECTED_CLASSES:
                continue
            if cls_id_counts[id] > 1:
                logging.warning(
                    f"Ignoring {log_str} {id=}, {cls_name=}: count={cls_id_counts[id]}"
                )
                continue
            xyxy = tuple(x.round(2) for x in box.cpu().numpy())
            if cls_name == "teacher":
                detection.teacher = xyxy
            elif cls_name == "student":
                detection.student = xyxy

        return detection
