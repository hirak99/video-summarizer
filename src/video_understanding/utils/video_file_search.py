import dataclasses
import logging
import os
import pathlib
import re

from . import file_conventions
from ..video_config import VIDEOS_DIR
from ..video_flow_nodes import video_flow_types

# Top directory of VIDEOS_DIR to search within.
_PROGRAM_DIRS: dict[video_flow_types.ProgramType, str] = {
    video_flow_types.ProgramType.PMA: "Medical Assistant Recordings",
    video_flow_types.ProgramType.FTP: "First Time Parent",
}


# TODO: Move to file_conventions.
@dataclasses.dataclass
class _FileNameComponents:
    date: str
    student: str
    teacher: str
    session: str

    @classmethod
    def from_pathname(cls, pathname: str) -> "_FileNameComponents | None":
        file_re = r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<session>.+)_(?P<uid1>[ESP][0-9]+?)-(?P<uid2>[ESP][0-9]+?)\.(?P<ext>[a-zA-Z0-9]+)"
        match = re.fullmatch(file_re, os.path.basename(pathname))
        relative_name = os.path.relpath(pathname, VIDEOS_DIR)  # For messages.
        if not match:
            logging.warning(f"Invalid file name: {relative_name}")
            return None
        parent_basename = os.path.basename(os.path.dirname(pathname))
        if parent_basename != match.group("uid2"):
            logging.warning(f"Parent dir doesn't match student name: {relative_name}")
            return None

        return cls(
            date=match.group("date"),
            student=match.group("uid2"),
            teacher=match.group("uid1"),
            session=match.group("session"),
        )


def all_video_files(
    *,
    program: video_flow_types.ProgramType,
    regex: str | None = None,
    students_list: list[str] | None = None,
    teachers_list: list[str] | None = None,
) -> list[str]:
    """Returns a list of all video files in the VIDEOS_DIR.

    Args:
        regex: If provided, only files that match the regex will be returned.
        students: If provided, only files that contain any of the students will be returned.
        teachers: If provided, only files that contain any of the teachers will be returned.

    Returns:
        A list of paths to all video files.
    """
    if students_list is None and teachers_list is None:
        raise ValueError(
            "Sanity check: Specify at least one of students or teachers. Specify empty to run all files."
        )

    students: set[str] = set(students_list) if students_list else set()
    teachers: set[str] = set(teachers_list) if teachers_list else set()
    seen_students: set[str] = set()
    seen_teachers: set[str] = set()

    program_dir = os.path.join(VIDEOS_DIR, _PROGRAM_DIRS[program])
    if not os.path.isdir(program_dir):
        raise ValueError(f"Path for {program=} is not a directory: {program_dir=}")

    video_files: list[pathlib.Path] = []
    for root, _, files in os.walk(program_dir):
        for filename in files:
            if not (filename.endswith(".mkv") or filename.endswith(".mp4")):
                continue

            components = _FileNameComponents.from_pathname(os.path.join(root, filename))
            if not components:
                continue

            if regex and not re.search(regex, filename):
                continue

            if students and components.student not in students:
                continue
            if teachers and components.teacher not in teachers:
                continue

            seen_students.add(components.student)
            seen_teachers.add(components.teacher)

            video_files.append(
                VIDEOS_DIR / os.path.relpath(root, VIDEOS_DIR) / filename
            )

    # Show an error if there's students or teachers wanted but not found.
    for entity, wanted, seen in [
        ("students", students, seen_students),
        ("teachers", teachers, seen_teachers),
    ]:
        unseen = wanted - seen
        if unseen:
            raise ValueError(
                f"Following {entity} don't appear in any of the video files: {unseen}"
            )

    return sorted((str(x) for x in video_files), key=file_conventions.sort_key)
