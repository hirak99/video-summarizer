import json

from . import role_based_captioner
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


def _summarize_prompt(
    source_file: str,
    task_description: str,
    role_aware_summary: list[role_based_captioner.RoleAwareCaptionT],
    scene_understanding: vision_processor.SceneListT | None,
    bad_segments: list[video_quality_profiler.BadSegment],
) -> list[str]:
    """Stores student evaluations as a json file and returns the path."""
    return templater.fill(
        prompt_templates.SESSION_SUMMARIZE_PROMPT_TEMPLATE,
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


class SessionSummarizer(process_node.ProcessNode):
    def __init__(self):
        # self._llm_instance = llm.LocalLlmInstance()
        # self._llm_instance = llm.OpenAiLlmInstance("gpt-3.5-turbo")
        # self._llm_instance = llm.OpenAiLlmInstance("gpt-4.1")
        self._llm_instance = llm.OpenAiLlmInstance("o4-mini")

    @override
    def process(
        self,
        source_file: str,
        role_aware_summary_file: str,
        # Note: Remove `| None` if we roll out ENABLE_VISION as True. For that we need to ensure the presence of manual labels.
        scene_understanding_file: str | None,
        bad_video_segments_file: str,
        out_file_stem: str,
    ) -> str:
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

        prompt = _summarize_prompt(
            source_file=source_file,
            task_description=task_description,
            role_aware_summary=role_aware_summary,
            scene_understanding=scene_understanding,
            bad_segments=bad_segments,
        )

        response_str = self._llm_instance.do_prompt_and_parse(
            "\n".join(prompt),
            transformers=[llm_utils.parse_as_markdown],
            max_tokens=4096,
        )

        out_file_suffix = f".session_summary.md"
        out_file_name = out_file_stem + out_file_suffix
        with open(out_file_name, "w") as f:
            f.write(response_str)
        return out_file_name
