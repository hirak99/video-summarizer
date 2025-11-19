import logging
import os
import pathlib
import re

from . import file_conventions
from .. import video_config
from ..video_flow_nodes import video_flow_types

# Top directory of VIDEOS_DIR to search within.
_PROGRAM_DIRS: dict[video_flow_types.ProgramType, str] = {
    video_flow_types.ProgramType.PMA: "Medical Assistant Recordings",
    video_flow_types.ProgramType.FTP: "First Time Parent",
}


def all_video_files(
    *,
    program: video_flow_types.ProgramType,
    regex: str | None = None,
    students_list: list[str] | None = None,
    teachers_list: list[str] | None = None,
    bad_files_errs: list[str] | None = None,
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

    program_dir = os.path.join(video_config.VIDEOS_DIR, _PROGRAM_DIRS[program])
    if not os.path.isdir(program_dir):
        raise ValueError(f"Path for {program=} is not a directory: {program_dir=}")

    video_files: list[pathlib.Path] = []
    for root, _, files in os.walk(program_dir):
        for filename in files:
            if not (filename.endswith(".mkv") or filename.endswith(".mp4")):
                continue

            file_path = os.path.join(root, filename)
            try:
                components = file_conventions.FileNameComponents.from_pathname(
                    os.path.join(root, filename)
                )
            except ValueError as e:
                rel_path = os.path.relpath(file_path, video_config.VIDEOS_DIR)
                logging.warning(f"Error parsing filename {rel_path}: {e}")
                if bad_files_errs is not None:
                    bad_files_errs.append(rel_path)
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
                video_config.VIDEOS_DIR
                / os.path.relpath(root, video_config.VIDEOS_DIR)
                / filename
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
