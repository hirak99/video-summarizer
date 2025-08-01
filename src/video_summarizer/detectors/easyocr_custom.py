import easyocr  # type: ignore
from numpy import typing as npt
import numpy as np

from . import detection_utils

from typing import Iterator


class Detector:
    def __init__(self):
        lang_list = ["en"]
        self._reader = easyocr.Reader(lang_list, gpu=True)

    def phone_numbers(
        self, image: npt.NDArray[np.uint8]
    ) -> Iterator[detection_utils.DetectionResult]:
        for bbox, text, prob in self._reader.readtext(image):  # type: ignore
            del prob
            if detection_utils.is_phone_number(text):
                yield detection_utils.DetectionResult(
                    name=detection_utils.DetectedObject.PHONE_NUMBER,
                    detail=text,
                    bbox={
                        "left": int(bbox[0][0]),
                        "top": int(bbox[0][1]),
                        "width": int(bbox[2][0]) - int(bbox[0][0]),
                        "height": int(bbox[2][1]) - int(bbox[0][1]),
                    },
                )
