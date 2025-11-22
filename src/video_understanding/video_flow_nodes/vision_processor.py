# Note: Since this processor is expensive, we save partial work as we process so
# that it can be resumed if interrupted.
#
# The tradeoff - if the VERSION changes and a partial file exists, the results from
# it will still be loaded. This can affect if the LLM being used was changed.
#
# So temp dir should be cleared if VERSION changes for a clean run.
#
import hashlib
import json
import logging
import os
import random

import moviepy  # type: ignore
from numpy import typing as npt
import numpy as np
from PIL import Image
import pydantic

from . import role_based_captioner
from .. import prompt_templates
from .. import video_config
from ...flow import process_node
from ..llm_service import abstract_llm
from ..llm_service import llm_utils
from ..llm_service import vision
from ..utils import interval_scanner
from ..utils import manual_labels_manager
from ..utils import misc_utils
from ..utils import templater
from ..utils import yolo_window_detector

from typing import Any, override

# Number of seconds to probe the video for.
_RESOLUTION_S = 5.0

# How many seconds of caption to use.
_CAPTION_SECS = 30.0

# Probability of logging VLM calls.
_LOG_PROBABILITY = 0.05


class SceneDescriptionT(pydantic.BaseModel):
    time: float
    # Hashes of the prompt and the image used.
    context_hash: str
    # List of bullet points describing the scene.
    scene: list[str]
    # List of bullet points describing changes from the last time point.
    actions: list[str]


# This is the format saved in the .json file.
class SceneListT(pydantic.BaseModel):
    model: str
    chronology: list[SceneDescriptionT]


def _sha_256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:10]


def _get_caption_lines(
    captions: list[role_based_captioner.RoleAwareCaptionT],
    last_results: SceneListT,
) -> list[str]:
    lines = []
    chronology = last_results.chronology
    chronology_index = -1

    if captions:
        lines += [
            "",
            "For additional context, below is an excerpt of conversation immediately before this frame -",
        ]
        first_scene_added = True
        for caption in captions:
            start_time = caption["interval"][0]
            while (
                chronology_index + 1 < len(chronology)
                and chronology[chronology_index + 1].time < start_time
            ):
                chronology_index += 1
                this_scene = chronology[chronology_index]
                if this_scene.time < start_time - _CAPTION_SECS:
                    # TODO: Test this.
                    continue
                # Append only if something changed, or this is the first scene.
                # TODO: Test this condition.
                if this_scene.actions or not first_scene_added:
                    first_scene_added = True
                    lines.append(f"{this_scene.time:0.1f}s")
                    if this_scene.scene:
                        lines[-1] += " | Student's View: " + " ".join(this_scene.scene)
                    if this_scene.actions:
                        lines[-1] += " | Student's Actions: " + " ".join(
                            this_scene.actions
                        )

            lines.append(
                f"{start_time:0.1f}s | {caption['speaker']}: \"{caption['text']}\"",
            )
        lines.append("")
    return lines


def _get_prompt(
    source_movie: str,
    captions: list[role_based_captioner.RoleAwareCaptionT],
    last_results: SceneListT,
) -> str:
    lines: list[str] = templater.fill(
        prompt_templates.SCENE_PROMPT_TEMPLATE_PART1,
        {
            "source_movie": source_movie,
            "optional_caption_lines": "\n".join(
                _get_caption_lines(captions, last_results)
            ),
        },
    )

    if last_results.chronology:
        prompt_args = {
            "last_frame_time": f"{last_results.chronology[-1].time:.1f}",
            "last_frame_scene_json": json.dumps(
                last_results.chronology[-1].scene, indent=2
            ),
        }
        lines += templater.fill(
            prompt_templates.SCENE_PROMPT_TEMPLATE_PART2_OTHER_FRAMES,
            prompt_args,
        )
    else:
        lines.extend(prompt_templates.SCENE_PROMPT_TEMPLATE_PART2_FIRST_FRAME)
    return "\n".join(lines)


class _VisionProcessor:
    def __init__(
        self,
        vision_model: abstract_llm.AbstractLlm,
        role_aware_summary: list[role_based_captioner.RoleAwareCaptionT],
        movie_path: str,
        out_file_stem: str,
    ):
        self._yolo_detector = yolo_window_detector.YoloWindowDetector()

        # Used to name files generated.
        self._timestamp = misc_utils.timestamp_str()

        self._vision = vision_model
        self._source_movie = movie_path

        self._out_file_stem = out_file_stem

        self._clip = moviepy.VideoFileClip(movie_path)

        labels = manual_labels_manager.VideoAnnotation(movie_path)
        self._student_labels = labels.get_student_scanner()

        self._scene_descriptions: SceneListT = SceneListT(
            model=vision_model.model_description(), chronology=[]
        )

        self._caption_scanner = interval_scanner.IntervalScanner(role_aware_summary)

        # If a partial output file exists, load it and delete it. We will use it
        # as a cache.
        self._prior_work: dict[str, SceneDescriptionT] = {}
        self._partial_file = video_config.tempdir() / (
            "_vision_processor." + os.path.basename(self._out_file_stem) + ".partial"
        )
        self._partial_load()

    def _partial_load(self) -> None:
        if os.path.exists(self._partial_file):
            logging.info(f"Partial file found: {self._partial_file}. Loading...")
            with open(self._partial_file, "r") as file:
                scenes = SceneListT.model_validate_json(file.read())
            if scenes.model != self._scene_descriptions.model:
                # Do not use results if the model has changed since then.
                return
            for scene in scenes.chronology:
                self._prior_work[scene.context_hash] = scene
            os.remove(self._partial_file)

    def _partial_save(self) -> None:
        logging.info(f"Saving to {self._partial_file!r}.")
        with open(self._partial_file, "w") as file:
            file.write(self._scene_descriptions.model_dump_json(indent=2))

    def process(self) -> str:
        t = 5.0
        clip_duration: Any = self._clip.duration
        assert isinstance(clip_duration, float)

        # Create a subdirectory for each call.
        log_dir = misc_utils.file_stem_to_log_stem(self._out_file_stem)
        log_dir = os.path.dirname(log_dir)
        log_dir = os.path.join(log_dir, f"vlm_{self._timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        logging.info(f"Logging VLM calls to: {log_dir!r}")

        while t < clip_duration - _RESOLUTION_S:
            self._process_frame(t, log_dir)
            self._partial_save()
            t += _RESOLUTION_S

        # We're done. Rename the partial save to actual save.

        # Since this will be a costly operation, preserve the previous outputs.
        # Create a new file every time an output is ready.
        outfname = f"{self._out_file_stem}.scene_understanding_{self._timestamp}.json"
        if os.path.exists(outfname):
            os.remove(outfname)
        os.rename(self._partial_file, outfname)
        logging.info(f"Saved from partial to: {outfname}")
        return outfname

    def _crop_to_windows(
        self, frame: npt.NDArray[np.uint8], t: float
    ) -> dict[yolo_window_detector.DetectionType, Image.Image]:

        image = Image.fromarray(frame)
        label = list(self._student_labels.containing_timestamp(t))
        if label:
            box = label[0]["label"]
            cropped_images: dict[yolo_window_detector.DetectionType, Image.Image] = {}
            cropped_images[yolo_window_detector.DetectionType.STUDENT] = image.crop(
                (box.x, box.y, box.x + box.width, box.y + box.height)
            )
            logging.info(f"Found and using human-labeled student window at {t}.")
            return cropped_images

        return self._yolo_detector.crop_to_detections(image, f"{t=}")

    def _process_frame(self, t: float, log_dir: str) -> None:
        frame = self._clip.get_frame(t)
        if frame is None:
            logging.warning(f"Could not find frame at {t}.")
            return None
        cropped = self._crop_to_windows(frame, t)
        student_image = cropped.get(yolo_window_detector.DetectionType.STUDENT)
        if student_image is None:
            logging.warning(f"Could not find student window at {t}.")
            return None

        captions = self._caption_scanner.overlapping_intervals(t - _CAPTION_SECS, t)
        prompt = _get_prompt(self._source_movie, captions, self._scene_descriptions)

        image_b64 = vision.to_base64(student_image)
        context_hash = _sha_256(prompt + image_b64)

        # Skip if we are re-running the process and there is a hit from previous partial file.
        if context_hash in self._prior_work:
            # Additionally check the time, though it may be unnecessary in statistical sense.
            if abs(self._prior_work[context_hash].time - t) <= 1e-5:
                logging.info(f"Cache hit to previous partial work at time {t}.")
                self._scene_descriptions.chronology.append(
                    self._prior_work[context_hash]
                )
                return

        call_count = len(self._scene_descriptions.chronology)

        if random.random() < _LOG_PROBABILITY:
            log_stem = os.path.join(log_dir, f"call_#{call_count:05d}")

            # Save the image into logs.
            with open(log_stem + ".png", "wb") as file:
                student_image.save(file)
            logging.info(f"Saved image to {log_stem}.png")

            # To be saved by the LLM call.
            log_file = log_stem + ".txt"
            logging.info("Logging to " + log_file)
        else:
            log_file = None

        def validate_as_list(result: Any) -> SceneDescriptionT | None:
            result["time"] = t
            result["context_hash"] = context_hash
            if "actions" not in result:
                result["actions"] = []
            return SceneDescriptionT.model_validate(result)

        result: SceneDescriptionT = self._vision.do_prompt_and_parse(
            prompt=prompt,
            max_tokens=4096,
            image_b64=image_b64,
            transformers=[llm_utils.parse_as_json, validate_as_list],
            log_file=log_file,
        )
        self._scene_descriptions.chronology.append(result)


class VisionProcess(process_node.ProcessNode):
    def __init__(self) -> None:
        self._model = vision.OpenAiVision("gpt-4.1")

    @override
    def process(
        self, source_file: str, role_aware_summary_file: str, out_file_stem: str
    ) -> str:
        with open(role_aware_summary_file, "r") as file:
            role_aware_summary: list[role_based_captioner.RoleAwareCaptionT] = (
                json.load(file)
            )
        processor = _VisionProcessor(
            vision_model=self._model,
            role_aware_summary=role_aware_summary,
            movie_path=source_file,
            out_file_stem=out_file_stem,
        )
        outfname = processor.process()
        return outfname

    @override
    def finalize(self) -> None:
        # Needed if we use local models.
        self._model.finalize()
