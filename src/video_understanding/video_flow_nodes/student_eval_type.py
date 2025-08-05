from typing import Literal, TypedDict


class StudentEvalT(TypedDict):
    example_of: Literal["strength", "weakness"]
    comment: str
    start: float
    end: float
    explanation: str
    importance: int
