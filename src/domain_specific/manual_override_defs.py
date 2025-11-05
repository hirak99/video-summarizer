# All manual overrides should be stored here, to isolate business logic.

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
INELIGIBLE_VIDEO_SECTIONS: list[tuple[str, tuple[int, int]]] = [
    # The line below is only an example, and should not match any actual file.
    ("EXAMPLE_VIDEO_FILE_BASENAME", (_mmss("10:00"), _mmss("10:30"))),
]

# Notes:
# 1. Replacements will only happen at word boundaries.
# 2. If multiple forms are to be replaced - e.g. capitalized, non-capitalized, all forms
#    should be provided here.
WORD_REPLACEMENTS: dict[str, str] = {
    "wordtoreplace": "replacement",
    "Wordtoreplace": "Replacement",
}
