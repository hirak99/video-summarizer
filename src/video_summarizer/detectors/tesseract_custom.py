import dataclasses

import cv2
from numpy import typing as npt
import numpy as np
import pytesseract  # type: ignore

from . import detection_utils

from typing import Iterator, TypedDict


class _OcrData(TypedDict):
    text: list[str]
    left: list[int]
    top: list[int]
    width: list[int]
    height: list[int]


@dataclasses.dataclass
class _TextBox:
    text: str
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self):
        return self.left + self.width

    def append(self, box):
        # print('  append', self, box)
        self.text += box.text
        self.top = min(self.top, box.top)
        self.width = box.left + box.width - self.left


def _do_ocr(frame: npt.NDArray[np.uint8]) -> _OcrData:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    inverted = cv2.bitwise_not(gray)
    return pytesseract.image_to_data(
        inverted,
        output_type=pytesseract.Output.DICT,
        # config=r"--oem 3 -c tessedit_char_whitelist=0123456789+()- ",
    )


def _is_adjacent(box1: _TextBox, box2: _TextBox) -> bool:
    """Determines if two textboxes may be considered as part of the same text."""
    return -2 < box2.left - box1.right <= 8 and abs(box1.top - box2.top) <= 5


def _iterate_texts(data: _OcrData) -> Iterator[_TextBox]:
    """Finds all texts.

    Tesseract tends to split up text a lot. This method joins adjacent OCR
    readings that should be part of the same text.
    """
    cumulative_text: _TextBox | None = None
    for index, text in enumerate(data["text"]):
        if not text:
            # Tesseract outputs all potential text from its detection model, even if OCR failed later.
            continue
        kwargs = {
            field: data[field][index]
            for field in ["text", "left", "top", "width", "height"]
        }
        box = _TextBox(**kwargs)
        if not cumulative_text or not _is_adjacent(cumulative_text, box):
            if cumulative_text:
                yield cumulative_text
            cumulative_text = box
        else:
            cumulative_text.append(box)
    if cumulative_text:
        yield cumulative_text


def iterate_phone_numbers(
    frame: npt.NDArray[np.uint8],
) -> Iterator[detection_utils.DetectionResult]:
    data = _do_ocr(frame)
    for box in _iterate_texts(data):
        if detection_utils.is_phone_number(box.text):
            yield detection_utils.DetectionResult(
                name=detection_utils.DetectedObject.PHONE_NUMBER,
                detail=box.text,
                bbox={
                    "left": box.left,
                    "top": box.top,
                    "width": box.width,
                    "height": box.height,
                },
            )
