import functools
import hashlib
import os
import re
import unicodedata

from typing import TypeVar

_SLUGIFY_REPLACEMENT_CHAR = "_"

_T = TypeVar("_T")


# Make a text safe to be part of filename.
def slugify(text: str) -> str:
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")  # Remove non-ASCII
    # Replace unwanted characters with the replacement
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    # Replace whitespace and repeated separators with a single separator
    text = re.sub(r"[-\s]+", _SLUGIFY_REPLACEMENT_CHAR, text)
    return text


@functools.cache
def file_stem_to_log_stem(file_stem: str) -> str:
    """Converts /path/to/stem -> /path/to/logs/stem"""
    log_dir = os.path.join(os.path.dirname(file_stem), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return os.path.join(log_dir, os.path.basename(file_stem))


def ensure_not_none(value: _T | None, *, err: str) -> _T:
    if value is None:
        raise ValueError(err)
    return value


def fingerprint(data: str) -> str:
    # Compute a unique 7-char fingerprint.
    # Used in saved outputs and filenames.
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:7]
