import enum

from typing import Literal, TypedDict


class ProgramType(enum.Enum):
    UNKNOWN = "unknown"
    # Pediatric Medical Assistant.
    PMA = "PMA"
    # First Time Parent.
    FTP = "FTP"


# Indicates what kind of highlights to capture from the video.
# Multiple of these can be added to a single graph.
class CompilationType(enum.Enum):
    UNKNOWN = "unknown"
    STUDENT_HIRING = "student_hiring"
    STUDENT_RESUME = "student_resume"
    TEACHER_HIRING = "teacher_hiring"
    FTP_HIGHLIGHTS = "ftp_highlights"


class HighlightsT(TypedDict):
    example_of: Literal["strength", "weakness"]
    comment: str
    start: float
    end: float
    explanation: str
    importance: int
