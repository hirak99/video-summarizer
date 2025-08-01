import dataclasses
import enum
import re

from typing import TypedDict

# Assume US phone number.
_PHONE_REGEX = re.compile(r"\+?\d{0,3} ?\(\d{3}\) ?\d{3} ?\-?\d{4}")
# _PHONE_REGEX = re.compile(r"\+?\d{0,3} ?\(?\d{3}\)? ?\d{3} ?\-?\d{4}")


def is_phone_number(text: str) -> bool:
    return bool(_PHONE_REGEX.match(text))


class DetectedObject(enum.Enum):
    PHONE_NUMBER = "tesseract_phone_number"
    # Add other detected objects here if needed.


class Bbox(TypedDict):
    top: int
    left: int
    width: int
    height: int


@dataclasses.dataclass
class DetectionResult:
    # Represents the type of detection.
    name: DetectedObject
    detail: str
    bbox: Bbox


def _box_almost_equal(box1: Bbox, box2: Bbox) -> bool:
    pixel_threshold = 5
    return (
        abs(box1["top"] - box2["top"]) < pixel_threshold
        and abs(box1["left"] - box2["left"]) < pixel_threshold
        and abs(box1["width"] - box2["width"]) < pixel_threshold
        and abs(box1["height"] - box2["height"]) < pixel_threshold
    )


def boxes_almost_equal(boxes1: list[Bbox], boxes2: list[Bbox]) -> bool:
    if len(boxes1) != len(boxes2):
        return False

    # Sort the boxes by top and left.
    boxes1.sort(key=lambda x: (x["top"], x["left"]))
    boxes2.sort(key=lambda x: (x["top"], x["left"]))

    for box1, box2 in zip(boxes1, boxes2):
        if not _box_almost_equal(box1, box2):
            return False

    return True


def result_list_almost_equal(
    detections1: list[DetectionResult], detections2: list[DetectionResult]
) -> bool:
    if len(detections1) != len(detections2):
        return False

    # Sort the boxes by top and left.
    detections1.sort(key=lambda x: (x.name, x.detail, x.bbox["top"], x.bbox["left"]))
    detections2.sort(key=lambda x: (x.name, x.detail, x.bbox["top"], x.bbox["left"]))

    for det1, det2 in zip(detections1, detections2):
        if det1.name != det2.name:
            return False
        if det1.detail != det2.detail:
            return False
        if not _box_almost_equal(det1.bbox, det2.bbox):
            return False

    return True
