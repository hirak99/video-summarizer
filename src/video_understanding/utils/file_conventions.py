import dataclasses
import functools
import logging
import os
import re

_SESSION_RE_CONTENT_TO_TASK = {
    "EKG": "pediatric EKG training",
    "Hearing": "pediatric Hearing Test training",
    "(color vision|Color Screening)": "pediatric Color Vision Tess training",
    "Heel stick": "pediatric Heel Stick blood sample collection training",
    "temperature": "pediatric temperature check training",
    "finger stick": "pediatric finger stick blood sample collection training",
    "PPD": "pediatric PPD skin test for tuberculosis training",
    "PPE donning doffing": "pediatric personal protective equipment gear donning and droffing training",
    "Weight": "pediatric weight measurement training",
    "height": "pediatric height measurement training",
    "Heart Rate": "pediatric heart rate measurement training",
    "Hand hygiene": "pediatric hand hygiene training",
}


# Files need to be sorted in the order of whichever word is found in the file
# name from the list below.
_FILE_SORT_ORDER = [
    "hygiene",
    "ppe",
    "temperature",
    "heart rate",
    "respiration",
    "blood pressure",
    "weight",
    "head circumference",
    "height",
    "ppd",
    "finger stick",
    "heel stick",
    "ekg",
    "hearing",
    "color vision",
]


@dataclasses.dataclass
class FileNameComponents:
    date: str
    student: str
    teacher: str
    session: str

    @classmethod
    def from_pathname(cls, pathname: str) -> "FileNameComponents":
        file_re = r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<session>.+)_(?P<uid1>[ESPG][0-9]+?)-(?P<uid2>[ESPG][0-9]+?)\.(?P<ext>[a-zA-Z0-9]+)"
        match = re.fullmatch(file_re, os.path.basename(pathname))

        if not match:
            raise ValueError(f"Invalid file name.")

        parent_basename = os.path.basename(os.path.dirname(pathname))
        if parent_basename != match.group("uid2"):
            raise ValueError(f"Parent dir doesn't match student name.")

        return cls(
            date=match.group("date"),
            student=match.group("uid2"),
            teacher=match.group("uid1"),
            session=match.group("session"),
        )


def filename_to_task(file_path: str) -> str:
    session_name = FileNameComponents.from_pathname(file_path).session

    for keyword, tasc_description in _SESSION_RE_CONTENT_TO_TASK.items():
        if re.search(keyword, session_name, re.IGNORECASE):
            return tasc_description

    # Fall back to heuristically create a description.
    return f"pediatric {session_name} training"


@functools.lru_cache(maxsize=2048)
def sort_key(file_name: str) -> int:
    """Returns the index of the first matching keyword, or a large number if none is found."""
    lower_file_name = os.path.basename(file_name).lower()
    for index, keyword in enumerate(_FILE_SORT_ORDER):
        if keyword in lower_file_name:
            return index
    logging.warning(f"No sorting keyword found in file name: {file_name}")
    return len(_FILE_SORT_ORDER)  # Put files with no matching keyword at the end.
