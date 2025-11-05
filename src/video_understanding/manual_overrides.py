import copy
import logging
import re

from ..domain_specific import manual_override_defs
from ..video_understanding.video_flow_nodes import role_based_captioner


# Used in two places -
# (1) student_evaluator - If rerun, will ignore this clip.
# (2) hiring_highlight_compiler - Will always ignore this clip.
def is_clip_ineligible(filename: str, start: float, end: float) -> bool:
    for fname_part, interval in manual_override_defs.INELIGIBLE_VIDEO_SECTIONS:
        if fname_part in filename:
            # Exclude any intersection with ineligible interval.
            if start < interval[1] and end > interval[0]:
                logging.warning(f"Ineligible clip: {filename} - {start} to {end}")
                return True
    return False


def word_replace(
    orig_caption: list[role_based_captioner.RoleAwareCaptionT],
) -> list[role_based_captioner.RoleAwareCaptionT]:
    caption = copy.deepcopy(orig_caption)
    for c in caption:
        for word, replacement in manual_override_defs.WORD_REPLACEMENTS.items():
            original = c["text"]
            # Use regex to replace at word boundaries.
            # Also manage case, "word" -> "replacement" and "Word" -> "Replacement."
            c["text"] = re.sub(
                r"\b" + re.escape(word) + r"\b",
                replacement,
                c["text"],
                flags=re.IGNORECASE,
            )
            if original != c["text"]:
                logging.info(f"Word replacement: '{original}' became '{c['text']}'")

    return caption
