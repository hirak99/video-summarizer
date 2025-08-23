import enum
from typing import Literal, TypedDict


class CompilationType(enum.Enum):
    UNKNOWN = "unknown"
    HIRING = "hiring"
    RESUME = "resume"


class StudentEvalT(TypedDict):
    example_of: Literal["strength", "weakness"]
    comment: str
    start: float
    end: float
    explanation: str
    importance: int
