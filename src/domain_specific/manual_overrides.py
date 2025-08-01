# All manual overrides should be stored here, to isolate business logic.

import logging

# Used in hiring_highlight_curator.
# Add segment hashes to drop here.
# But prefer handling this in other ways - e.g. with is_clip_ineligible.
BAD_HIRING_SEGMENTS: list[str] = [
    # Example: A hash like below.
    "5d4c942",
]


def _mmss(mmss: str) -> int:
    # Parse time given as "10:17" into int 10 * 60 + 17.
    mm, ss = mmss.split(":")
    return int(mm) * 60 + int(ss)


# Video segments that should not be processed.
_INELIGIBLE_VIDEO_SECTIONS: list[tuple[str, tuple[int, int]]] = [
    # The line below is only an example, and should not match any actual file.
    ("EXAMPLE_VIDEO_FILE_BASENAME", (_mmss("10:00"), _mmss("10:30"))),
]


# Used in two places -
# (1) student_evaluator - If rerun, will ignore this clip.
# (2) hiring_highlight_compiler - Will always ignore this clip.
def is_clip_ineligible(filename: str, start: float, end: float) -> bool:
    for fname_part, interval in _INELIGIBLE_VIDEO_SECTIONS:
        if fname_part in filename:
            # Exclude any intersection with ineligible interval.
            if start < interval[1] and end > interval[0]:
                logging.warning(f"Ineligible clip: {filename} - {start} to {end}")
                return True
    return False
