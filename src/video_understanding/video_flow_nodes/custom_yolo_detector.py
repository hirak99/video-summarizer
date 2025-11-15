# IMPORTANT
# WIP, incomplete. Probably will drop this. Instead, carry out inference on the fly in vision_processor.
# If we do this -
# We will need to add code complexity to store, load, and scan the detections.
# The complexity can be avoided since detection on image is very cheap.
import logging

import cv2
import moviepy
import numpy as np
import numpy.typing as npt
from pyannote import audio  # type: ignore
import pydantic

from ...flow import process_node
from ..utils import yolo_window_detector

from typing import override

# Number of pixels within which we merge with the last detection.
_MERGE_THRESHOLD_PX = 5

# Number of seconds of absent images to be merged / filled in.
_MERGE_THRESHOLD_S = 1.0


class _YoloDetection(pydantic.BaseModel):
    # Includes both ends.
    frame_range: tuple[int, int]
    interval: tuple[float, float]
    xyxy: tuple[float, float, float, float]


class YoloDetections:
    def __init__(self) -> None:
        self._last_frame = -1
        self._by_cls: dict[str, list[_YoloDetection]] = {}
        self._pydantic_root = pydantic.RootModel[dict[str, list[_YoloDetection]]]

    def add(
        self,
        cls_name: str,
        frame: int,
        time: float,
        xyxy: tuple[float, float, float, float],
    ):
        dets = self._by_cls.setdefault(cls_name, [])

        merge = False
        if dets:
            last_det = dets[-1]
            if time - last_det.interval[1] <= _MERGE_THRESHOLD_S:
                # Check if coordinates are almost same.
                max_diff = max(abs(a - b) for a, b in zip(xyxy, last_det.xyxy))
                if max_diff < _MERGE_THRESHOLD_PX:
                    merge = True

            if merge:
                last_det.frame_range = (last_det.frame_range[0], frame)
                last_det.interval = (
                    last_det.interval[0],
                    time,
                )
                # Compute weighted average of the xyxy, by number of frames.
                last_frames = last_det.frame_range[1] - last_det.frame_range[0] + 1
                new_xyxy = tuple(
                    (last * last_frames + current) / (last_frames + 1)
                    for last, current in zip(last_det.xyxy, xyxy)
                )
                assert len(new_xyxy) == 4
                last_det.xyxy = new_xyxy

        if not merge:
            this_det = _YoloDetection(
                frame_range=(frame, frame), interval=(time, time), xyxy=xyxy
            )
            dets.append(this_det)

    def model_dump_json(self) -> str:
        return self._pydantic_root.model_validate(self._by_cls).model_dump_json()


class CustomYoloDetector(process_node.ProcessNode):
    def __init__(self) -> None:
        self._detector = yolo_window_detector.YoloWindowDetector()

    @override
    def process(
        self,
        source_file: str,
        checksum: dict[str, str],  # Only for graph dependency.
        out_file_stem: str,
    ) -> str:
        clip = moviepy.VideoFileClip(source_file)

        detections = YoloDetections()

        frame_time: np.float64
        frame: npt.NDArray[np.uint8]
        for frame_count, (frame_time, frame) in enumerate(
            clip.iter_frames(with_times=True)
        ):
            result = self._detector.detect(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"{frame_count=} {frame_time=}"
            )

            for cls_enum, xyxy in result.items():
                detections.add(
                    cls_name=cls_enum.value,
                    frame=frame_count,
                    time=float(frame_time),
                    xyxy=xyxy,
                )

            if frame_count % 100 == 0:
                logging.info(f"frame: {frame_count}")

            if frame_count >= 300:
                print(detections.model_dump_json())
                print(f"{out_file_stem=}")
                raise RuntimeError("forced break")

        out_file = out_file_stem + ".yolo_windows.json"
        with open(out_file, "w") as f:
            f.write(detections.model_dump_json())
            logging.info(f"Yolo outputs written to {out_file!r}")

        return out_file
