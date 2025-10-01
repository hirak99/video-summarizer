import functools
import logging
import os

import regex

_FILE_REGEX_TO_TASK = {
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


def filename_to_task(file_path: str) -> str:
    basename = os.path.basename(file_path)
    for regex_str, description in _FILE_REGEX_TO_TASK.items():
        if regex_str in basename:
            return description

    # Fall back to heuristically deducing the metadata.
    logging.info(f"No metadata found for file: {file_path}")
    m = regex.fullmatch(
        # It's not always E000x-S00xx.
        # TODO: Add test for each of the cases below.
        # Examples:
        # 2025-01-01_weight_E000x_ S000x.mkv
        # 01-01-25_hearing_E000x-S000x.mkv
        # 2025-01-30 13-20-12_PPE_E000x-S000x.mkv
        r"\d{2,4}-\d{2}-\d{2}[ \d]*_(?P<name>.*)_E0.*?S0.*?\.(mp4|mkv)",
        basename,
    )
    if not m:
        raise ValueError(
            f"Could not also deduce metadata from file name: {file_path!r}"
        )
    name = m["name"]
    return f"pediatric {name} training"


@functools.lru_cache(maxsize=2048)
def sort_key(file_name: str) -> int:
    """Returns the index of the first matching keyword, or a large number if none is found."""
    lower_file_name = os.path.basename(file_name).lower()
    for index, keyword in enumerate(_FILE_SORT_ORDER):
        if keyword in lower_file_name:
            return index
    logging.warning(f"No sorting keyword found in file name: {file_name}")
    return len(_FILE_SORT_ORDER)  # Put files with no matching keyword at the end.
