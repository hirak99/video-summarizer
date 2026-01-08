import enum

from typing import Literal, NotRequired, TypedDict


class ProgramType(enum.Enum):
    UNKNOWN = "unknown"
    # Pediatric Medical Assistant.
    PMA = "PMA"
    # First Time Parent.
    FTP = "FTP"
    # Miscellaneous.
    MISC = "MISC"


# Indicates what kind of highlights to capture from the video.
# Multiple of these can be added to a single graph.
class CompilationType(enum.Enum):
    UNKNOWN = "unknown"

    # These nodes are only for PMA.
    STUDENT_HIRING = "student_hiring"
    STUDENT_RESUME = "student_resume"
    TEACHER_HIRING = "teacher_hiring"

    # These nodes are only for FTP.
    FTP_HIGHLIGHTS = "ftp_highlights"


class HighlightsT(TypedDict):
    example_of: NotRequired[Literal["strength", "weakness"]]
    comment: str
    start: float
    end: float
    explanation: str
    importance: int
