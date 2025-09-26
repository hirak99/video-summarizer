import collections
import enum
import logging
import os

from cv2 import typing as cvt
from PIL import Image
from pyannote import audio  # type: ignore
import ultralytics

_YOLO_WEIGHTS = os.path.expanduser(
    "~/code/bounding-box-model/_yolo_models/multi-class/weights/best.pt"
)

# The model was built with these class names.
_EXPECTED_CLASSES = ["teacher", "student"]


class DetectionType(enum.Enum):
    TEACHER = "teacher"
    STUDENT = "student"


class YoloWindowDetector:
    def __init__(self) -> None:
        self._yolo = ultralytics.YOLO(_YOLO_WEIGHTS)

    def detect(
        self, frame_rgb: cvt.MatLike | Image.Image, log_str: str
    ) -> dict[DetectionType, tuple[int, int, int, int]]:
        """Detects teacher/student boxes.

        Args:
            frame_rgb: Note that if PIL.Image is passed, it automatically converts to RGB.
                But if passed as MatLike, conversion must be performed before passing.

        Returns:
            All detected windows in xyxy form.
        """
        detections: dict[DetectionType, tuple[int, int, int, int]] = {}

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
            xyxy = tuple(int(x + 0.5) for x in box.cpu().numpy())
            assert len(xyxy) == 4
            if cls_name == "teacher":
                detections[DetectionType.TEACHER] = xyxy
            elif cls_name == "student":
                detections[DetectionType.STUDENT] = xyxy

        return detections

    def crop_to_detections(
        self, image: Image.Image, log_str: str
    ) -> dict[DetectionType, Image.Image]:
        detections = self.detect(image, log_str)
        cropped_images: dict[DetectionType, Image.Image] = {}
        for detection_type, box in detections.items():
            cropped_images[detection_type] = image.crop(box)
        return cropped_images
