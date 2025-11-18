import functools
import json
import logging

from . import word_caption_utils
from .. import prompt_templates
from ...flow import process_node
from ..llm_service import llm
from ..llm_service import llm_utils
from ..utils import file_conventions
from ..utils import templater

from typing import Any, override


def _caption_to_str(captions: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    """Convert a caption dictionary to a text representation."""

    speaker_aliases: dict[str, str] = {}
    # Diarization produces "SPEAKER_00", "SPEAKER_01", etc.
    # Example:
    # {
    #   "SPEAKER_00": "Person A",
    #   "SPEAKER_01": "Person B",
    # }

    for i, speaker in enumerate(word_caption_utils.all_speakers(captions)):
        speaker_aliases[speaker] = f"Person {chr(i + ord('A'))}"

    combined = word_caption_utils.merge_word_captions(
        captions, speaker_aliases, unknown="Either"
    )

    lines: list[str] = []
    for caption in combined:
        speaker = caption["speaker"]
        lines.append(f"{speaker}: {caption['text']}")
    return "\n".join(lines), speaker_aliases


def _result_parser(
    response: dict[str, str], *, speaker_names: list[str]
) -> dict[str, str] | None:
    """Converts LLM response.

    Args:
        response: Example {"Person A": "Student", "Person B": "Teacher"}.
        speaker_names: All known speaker names, e.g. "Person A", "Person B".
    Returns:
        Same as response, but drops unknown keys (e.g. "Person C").
    """
    # Valid keys in LLM response. E.g. "Person A".
    for speaker_name in speaker_names:
        if speaker_name not in response:
            logging.warning(f"Response is missing speaker: {speaker_name}")
            return None
        # Capitalize the values to ensure consistency.
        response[speaker_name] = response[speaker_name].strip().capitalize()
        if response[speaker_name] not in ["Teacher", "Student"]:
            logging.error(
                f"Response value for {speaker_name} is not valid: {response[speaker_name]}"
            )
            return None
    # Remove any keys that are not in the valid keys.
    result = {k: v for k, v in response.items() if k in speaker_names}
    if not result:
        logging.error("No valid keys (e.g. 'Person A') found in the LLM response")
        return None
    return result


class RoleIdentifier(process_node.ProcessNode):

    def __init__(self):
        self._llm_instance = llm.OpenAiLlmInstance("o4-mini")

    @override
    def process(
        self, source_file: str, word_captions_file: str, out_file_stem: str
    ) -> dict[str, str]:
        """Returns a simple dict identifying speaker rolse.

        Returns:
            A dict for example {"SPEAKER_00": "Teacher", "SPEAKER_01": "Student"}.
        """
        with open(word_captions_file, "r") as file:
            data = json.load(file)
        # print(_to_caption_text(data))

        caption_text, speaker_aliases = _caption_to_str(data)

        # Unalias the speaker names to the original names.
        # E.g. {"Person A": "SPEAKER_00", ...}.
        speaker_unalias = {v: k for k, v in speaker_aliases.items()}

        role_dict_example = json.dumps({k: "ROLE" for k in speaker_unalias.keys()})
        logging.info(f"{role_dict_example=}")

        prompt_lines = templater.fill(
            prompt_templates.ROLE_IDENTIFIER_PROMPT_TEMPLATE,
            {
                "caption_text": caption_text,
                "task_description": file_conventions.filename_to_task(source_file),
                "role_dict_example": role_dict_example,
            },
        )
        response = self._llm_instance.do_prompt_and_parse(
            "\n".join(prompt_lines),
            transformers=[
                llm_utils.remove_thinking,
                llm_utils.parse_as_json,
                functools.partial(
                    _result_parser, speaker_names=list(speaker_unalias.keys())
                ),
            ],
            max_tokens=8192,
            log_file=f"{out_file_stem}.role_assigner_llm_debug.txt",
            log_additional_info="\n".join([f"Speaker Aliases: {speaker_aliases!r}"]),
        )
        logging.info(f"Parsed LLM Response: {response!r}")
        response = {speaker_unalias[k]: v for k, v in response.items()}
        logging.info(f"Unaliased speaker names: {response!r}")
        return response

    def finalize(self) -> None:
        self._llm_instance.finalize()
