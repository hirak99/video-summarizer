import os
import pathlib
import re
import unicodedata

_SLUGIFY_REPLACEMENT_CHAR = "_"


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


def get_output_stem(
    source_file: str, old_root: pathlib.Path, new_root: pathlib.Path
) -> str:
    out_path = os.path.join(
        new_root,
        os.path.relpath(os.path.dirname(source_file), old_root),
    )
    return os.path.join(out_path, os.path.splitext(os.path.basename(source_file))[0])
