import functools
import logging
import os

import regex

_FILE_REGEX_TO_TASK = {
    "EKG": "Pediatric EKG training session",
    "Hearing": "Pediatric Hearing Test training session",
    "(color vision|Color Screening)": "Pediatric Color Vision Tess training session",
    "Heel stick": "Pediatric Heel Stick blood sample collection training session",
    "temperature": "Pediatric temperature check training session",
    "finger stick": "Pediatric finger stick blood sample collection training session",
    "PPD": "Pediatric PPD skin test for tuberculosis training session",
    "PPE donning doffing": "Pediatric personal protective equiment gear donning and droffing training session",
    "Weight": "Pediatric weight measurement training session",
    "height": "Pediatric height measurement training session",
    "Heart Rate": "Pediatric heart rate measurement training session",
    "Hand hygiene": "Pediatric hand hygiene training session",
}


# Files need to be sorted in the order of whichever word is found in the file
# name from the list below.
_FILE_SORT_ORDER = [
    "hygiene",
    "ppe",
    "temperature",
    "blood pressure",
    "heart rate",
    "respiration",
    "weight",
    "head circumference",
    "height",
    "skill",
    "ekg",
    "hearing",
    "color vision",
    "finger stick",
    "heel stick",
]


def filename_to_task(file_path: str) -> str:
    basename = os.path.basename(file_path)
    for regex_str, description in _FILE_REGEX_TO_TASK.items():
        if regex_str in basename:
            return description

    # Fall back to heuristically deducing the metadata.
    logging.info(f"No metadata found for file: {file_path}")
    m = regex.fullmatch(
        # It's not always E000x-S00xx.
        # Example: 2025-01-01_weight_E000x_ S00xx.mkv
        # Or: 01-01-25_hearing_E000x-S000x.mkv
        r"\d{2,4}-\d{2}-\d{2}[ \d]*_(?P<name>.*)_E0.*?S0.*?\.(mp4|mkv)",
        basename,
    )
    if not m:
        raise ValueError(
            f"Could not also deduce metadata from file name: {file_path!r}"
        )
    name = m["name"]
    return f"Pediatric {name} training session"


@functools.lru_cache(maxsize=2048)
def sort_key(file_name: str) -> int:
    """Returns the index of the first matching keyword, or a large number if none is found."""
    lower_file_name = os.path.basename(file_name).lower()
    for index, keyword in enumerate(_FILE_SORT_ORDER):
        if keyword in lower_file_name:
            return index
    logging.warning(f"No sorting keyword found in file name: {file_name}")
    return len(_FILE_SORT_ORDER)  # Put files with no matching keyword at the end.
