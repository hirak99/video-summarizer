import datetime
import json

from . import role_based_captioner
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

_FILE_SUFFIX = ".student_eval.json"


def _student_evaluation_prompt(
    source_file: str,
    task_description: str,
    role_aware_summary: list[role_based_captioner.RoleAwareCaptionT],
    scene_understanding: vision_processor.SceneListT | None,
) -> list[str]:
    """Stores student evaluations as a json file and returns the path."""
    return templater.fill(
        prompt_templates.STUDENT_EVAL_PROMPT_TEMPLATE,
        {
            "task_description": task_description,
            "caption_lines": "\n".join(
                prompt_utils.caption_lines_for_prompt(
                    source_file, role_aware_summary, scene_understanding
                )
            ),
        },
    )


class StudentEvaluator(process_node.ProcessNode):
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
        # Note: Remove `| None` if we roll out ENABLE_VISION as True.
        scene_understanding_file: str | None,
        out_file_stem: str,
    ) -> str:
        out_file_name = out_file_stem + _FILE_SUFFIX
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

        prompt = _student_evaluation_prompt(
            source_file, task_description, role_aware_summary, scene_understanding
        )

        # Current date-time as "yyyymmdd-hhmmss".
        datetime_str = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")

        response = self._llm_instance.do_prompt_and_parse(
            "\n".join(prompt),
            transformers=[llm.remove_thinking, llm_utils.parse_as_json],
            max_tokens=4096,
            log_file=f"{out_file_name}.llm_log.v{video_config.VERSION}.{datetime_str}.txt",
        )
        with open(out_file_name, "w") as f:
            json.dump(response, f)
        return out_file_name
