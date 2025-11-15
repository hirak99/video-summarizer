import datetime
import json

from . import role_based_captioner
from . import video_flow_types
from . import video_quality_profiler
from . import vision_processor
from .. import prompt_templates
from .. import video_config
from ...flow import process_node
from ..llm_service import llm
from ..llm_service import llm_utils
from ..utils import file_conventions
from ..utils import prompt_utils
from ..utils import templater

from typing import override


def _student_evaluation_prompt(
    compilation_type: video_flow_types.CompilationType,
    source_file: str,
    task_description: str,
    role_aware_summary: list[role_based_captioner.RoleAwareCaptionT],
    scene_understanding: vision_processor.SceneListT | None,
    bad_segments: list[video_quality_profiler.BadSegment],
) -> list[str]:
    """Stores student evaluations as a json file and returns the path."""
    match compilation_type:
        case video_flow_types.CompilationType.STUDENT_HIRING:
            prompt_template = prompt_templates.STUDENT_HIRING_PROMPT_TEMPLATE
        case video_flow_types.CompilationType.STUDENT_RESUME:
            prompt_template = prompt_templates.STUDENT_RESUME_PROMPT_TEMPLATE
        case video_flow_types.CompilationType.TEACHER_HIRING:
            prompt_template = prompt_templates.TEACHER_HIRING_PROMPT_TEMPLATE
        case video_flow_types.CompilationType.FTP_HIGHLIGHTS:
            prompt_template = prompt_templates.FTP_PROMPT_TEMPLATE
        case _:
            raise ValueError(f"Unknown compilation type: {compilation_type}")

    return templater.fill(
        prompt_template,
        {
            "task_description": task_description,
            "caption_lines": "\n".join(
                prompt_utils.caption_lines_for_prompt(
                    source_file=source_file,
                    role_aware_summary=role_aware_summary,
                    scene_understanding=scene_understanding,
                    bad_segments=bad_segments,
                )
            ),
        },
    )


class HighlightsSelector(process_node.ProcessNode):
    def __init__(self):
        # self._llm_instance = llm.LocalLlmInstance()
        # self._llm_instance = llm.OpenAiLlmInstance("gpt-3.5-turbo")
        # self._llm_instance = llm.OpenAiLlmInstance("gpt-4.1")
        self._llm_instance = llm.OpenAiLlmInstance("o4-mini")

    @override
    def process(
        self,
        compilation_type: video_flow_types.CompilationType,
        source_file: str,
        role_aware_summary_file: str,
        # Note: Remove `| None` if we roll out ENABLE_VISION as True. For that we need to ensure the presence of manual labels.
        scene_understanding_file: str | None,
        bad_video_segments_file: str,
        out_file_stem: str,
    ) -> str:
        """Evaluates a video for different purposes as indicated in the params.

        Args:
            prompt_template: The prompt can vary based on what this node is required to evaluate, for example a prompt for student hiring, a different prompt for student resume.
            source_file: The video file.
            role_aware_summary_file: Transcriptions from a prior node.
            scene_understanding_file: Scene understanding from a prior node.
            out_file_stem: The output file name stem. Changes based on the video being processed.
            out_file_suffix: The output file name suffix. Should be a constant for the node, but different nodes with different prompts should have different suffixes.

        Returns:
            Full path of the json file with output.
        """
        out_file_suffix = f".highlights_for_{compilation_type.value}.json"
        out_file_name = out_file_stem + out_file_suffix
        with open(role_aware_summary_file, "r") as f:
            role_aware_summary: list[role_based_captioner.RoleAwareCaptionT] = (
                json.load(f)
            )

        task_description = file_conventions.filename_to_task(source_file)

        scene_understanding: vision_processor.SceneListT | None = None
        if video_config.ENABLE_VISION:
            if scene_understanding_file is None:
                raise ValueError(
                    "Vision is enabled, so scene_understanding_file must be provided."
                )
            with open(scene_understanding_file, "r") as f:
                scene_understanding = vision_processor.SceneListT.model_validate_json(
                    f.read()
                )

        with open(bad_video_segments_file, "r") as f:
            bad_segments: list[video_quality_profiler.BadSegment] = json.load(f)

        prompt = _student_evaluation_prompt(
            compilation_type=compilation_type,
            source_file=source_file,
            task_description=task_description,
            role_aware_summary=role_aware_summary,
            scene_understanding=scene_understanding,
            bad_segments=bad_segments,
        )

        # Current date-time as "yyyymmdd-hhmmss".
        datetime_str = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")

        response_list = self._llm_instance.do_prompt_and_parse(
            "\n".join(prompt),
            transformers=[llm_utils.remove_thinking, llm_utils.parse_as_json],
            max_tokens=4096,
            log_file=f"{out_file_name}.llm_log.v{video_config.VERSION}.{datetime_str}.txt",
        )

        for response in response_list:
            if compilation_type in {
                video_flow_types.CompilationType.STUDENT_RESUME,
                video_flow_types.CompilationType.FTP_HIGHLIGHTS,
            }:
                # Check that "example_of" is not populated.
                if "example_of" in response:
                    raise ValueError(f"{compilation_type=} but has 'example_of'.")
                # Mark "example_of" as "strength" for STUDENT_RESUME.
                if compilation_type == video_flow_types.CompilationType.STUDENT_RESUME:
                    response["example_of"] = "strength"
            else:
                # Check that "example_of" is populated.
                if "example_of" not in response:
                    raise ValueError(
                        f"{compilation_type=} but 'example_of' is not populated."
                    )
            # Confirm that responses follow the correct format.
            # TODO: This should ideally be moved to a prompt transformer, so that automatic retry can happen.
            video_flow_types.HighlightsT(**response)

        with open(out_file_name, "w") as f:
            json.dump(response_list, f)
        return out_file_name
