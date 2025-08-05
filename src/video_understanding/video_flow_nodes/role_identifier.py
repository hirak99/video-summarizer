import functools
import json
import logging
import os

from . import word_caption_utils
from ...flow import process_node
from ..llm_service import llm
from ..llm_service import llm_utils

from typing import Any, override


def _caption_to_str(captions: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    """Convert a caption dictionary to a text representation."""

    speaker_aliases: dict[str, str] = {}
    # Diarization produces "SPEAKER_00", "SPEAKER_01", etc.
    # Convert "SPEAKER_00" to "Person A" and so on, to help the LLM.

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
    response: dict[str, str], *, speaker_aliases: dict[str, str]
) -> dict[str, str] | None:
    # Valid keys in LLM response. E.g. "Person A".
    valid_keys = speaker_aliases.values()
    for valid_key in valid_keys:
        if valid_key not in response:
            logging.warning(f"Response is missing key: {valid_key}")
            # The LLM can produce ..., "Person B": "Teacher" even when there is
            # no Person B and only the teacher spoke. So we allow non-existent
            # keys, and ignore them.
            continue
        # Capitalize the values to ensure consistency.
        response[valid_key] = response[valid_key].strip().capitalize()
        if response[valid_key] not in ["Teacher", "Student"]:
            logging.error(
                f"Response value for {valid_key} is not valid: {response[valid_key]}"
            )
            return None
    # Remove any keys that are not in the valid keys.
    result = {k: v for k, v in response.items() if k in valid_keys}
    if not result:
        logging.error("No valid keys (e.g. 'Person A') found in the LLM response")
        return None
    return result


class RoleIdentifier(process_node.ProcessNode):

    def __init__(self):
        self._llm_instance = llm.OpenAiLlmInstance("o4-mini")

    @override
    def process(self, word_captions_file: str, out_file_stem: str) -> dict[str, str]:
        """Returns a simple dict identifying speaker rolse.

        Returns:
            A dict for example {"SPEAKER_00": "Teacher", "SPEAKER_01": "Student"}.
        """
        with open(word_captions_file, "r") as file:
            data = json.load(file)
        # print(_to_caption_text(data))

        caption_text, speaker_aliases = _caption_to_str(data)

        prompt_lines: list[str] = [
            # "You are an AI assistant that identifies roles in a conversation. "
            # "Analyze the following conversation and identify the roles of each speaker. "
            # "Provide a brief description of each role based on their speech patterns and content.\n\n"
            "Scan the following transciption of a teaching session, and identify which of Person A or Person B was the teacher, and which was the student.",
            "",
            "---",
            # "Person A: I am teacher.",
            # "Person B: I am student.",
            f"{caption_text}",
            "---",
            f"The session name is {os.path.basename(out_file_stem)}",
            "Analyze all the lines above, and determine who is the teacher and who is the student.",
            "The diarization may not be 100% perfect, there can be very few lines incorrectly captioned.",
            "Restrict your response to only one line of json, with the follwing format:",
            "" '{"Person A": role, "Person B": role}',
            "" 'The `role` can be either "Teacher" or "Student".',
        ]
        response = self._llm_instance.do_prompt_and_parse(
            "\n".join(prompt_lines),
            transformers=[
                llm.remove_thinking,
                llm_utils.parse_as_json,
                functools.partial(_result_parser, speaker_aliases=speaker_aliases),
            ],
            max_tokens=8192,
            log_file=f"{out_file_stem}.role_assigner_llm_debug.txt",
            log_additional_info="\n".join([f"Speaker Aliases: {speaker_aliases!r}"]),
        )
        logging.info(f"Parsed LLM Response: {response!r}")
        # Unalias the speaker names to the original names.
        speaker_unalias = {v: k for k, v in speaker_aliases.items()}
        response = {speaker_unalias[k]: v for k, v in response.items()}
        logging.info(f"Unaliased speaker names: {response!r}")
        return response

    def finalize(self) -> None:
        self._llm_instance.finalize()
