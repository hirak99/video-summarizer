# Declares important directory locations.
# See also domain_config.py for domain-specific directives.
import logging
import os
import pathlib
import random
import string

# History:
# 6.3 - Shorten to 5 minutes.
# 6.4 - (1) FFWD silences, (2) semi-transparent black box below captions.
# 6.5 - Ask for student-eval explanation. Should improve diagnosability, and
#   quality of the highlights.
# 6.6 - Reduce overlap_threshold to 0.0s and select which to keep based on points.
# 6.7 - Prompt refinement to (a) include full conversation for weaknesses, (b) include only valid clarifications, and (c) stress further on not including equipment issues.
# 6.8 -
#   * Ask to double check the time intervals, since it often gets that wrong.
#     Example last interval here: https://drive.google.com/file/d/198LIeJ6vsK83pNWYgcnq5I66_c4BAm25/view?usp=drive_link
#   * Split long captions that don't fit on screen into two lines.
# 6.9 - Encourage variety across sessions in selection.
# 7.0 - Minor tweak to student evaluator prompt.
# 7.1 - Integrated manual annotations for blurring.
# 7.1.1 - For hiring highlights, use black bg. Store labels file in chosen_highlights log.
# 7.2 - Split caption into as many lines as needed.
# 7.3 - Implement fix for Whisper's rolling captions.
# 7.4 - Scene understanding implementation done.
# 7.4.1 - Skip fading for consecutive clips.
# 7.5 - Correct long (initial) silence in some Whisper transcriptions.
# 7.6 - Prompt tweak to prioritize teacher's response in clarifications. Also reduced the fade time from 1.0s to 0.5s.
# 7.7 - Added resume prompt. Also now we capitalize comments - so no more fully lowercase comments.
# 7.7.1 - Updated resume prompt.
# 7.7.2 - Further updated resume prompt.
# 7.8 - Implemented automatic inadmissible segment ("choppiness") detection on videos.
# 7.8.1 - Iteration of prompts.
# 7.8.2 - Iteration of prompts.
# 7.9 - Automated detection of teacher/student windows with a custom YOLO model.
VERSION = "7.9"

_HOME = pathlib.Path(os.environ["HOME"])
VIDEOS_DIR = _HOME / "data/videos"
WORKSPACE_DIR = _HOME / "data/workspace"

RESULTS_DIR = _HOME / "data/results"
VIDEO_SUMMARIES_DIR = RESULTS_DIR / "video_summaries"

# Where the labels should be read from.
# To use new labels, freeze and set the path.
MANUAL_LABELS_DIR = _HOME / "data/manual_labeling/frozen/latest"

# Used to keep temporary movies and such.
# Read with the tempdir() function here, which also creates it.
_TEMP_DIR = _HOME / "data/_tmp"


# Process partial payload to test during development.
TESTING_MODE = False

# Development flag, set via video_flow.py.
ENABLE_VISION = True


def tempdir() -> pathlib.Path:
    os.makedirs(_TEMP_DIR, exist_ok=True)
    return _TEMP_DIR


def random_temp_fname(prefix: str, extension: str) -> str:
    random_string = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(16)
    )
    if extension and not extension.startswith("."):
        raise ValueError(
            f"If present, extension should start with ., got: {extension!r}"
        )
    return str(tempdir() / f"{prefix}_{random_string}{extension}")


# This is meant to be nagging reminder of flags that should not be on during prod.
# Call this between nodes, and when any large amount of logging is done.
def repeated_warnings():
    if TESTING_MODE:
        logging.warning("TESTING_MODE is True")
