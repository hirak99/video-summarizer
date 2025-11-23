import argparse

import dotenv

from . import video_config
from . import video_flow_graph
from .utils import logging_utils
from .utils import video_file_search
from .video_flow_nodes import video_flow_types


def _main(
    *,
    regex: str | None,
    limit_files: int,
    makeviz: bool,
    dry_run: bool,
    program: video_flow_types.ProgramType,
    students: list[str] | None,
    teachers: list[str] | None,
):
    all_files_to_process = video_file_search.all_video_files(
        program=program, regex=regex, students_list=students, teachers_list=teachers
    )

    if limit_files:
        all_files_to_process = all_files_to_process[:limit_files]

    video_graph = video_flow_graph.VideoFlowGraph(
        program=program, makeviz=makeviz, dry_run=dry_run
    )
    video_graph.run(all_files_to_process=all_files_to_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files")

    program_types_str = ", ".join(
        [f"'{x.value}'" for x in video_flow_types.ProgramType]
    )
    parser.add_argument(
        "--program",
        type=video_flow_types.ProgramType,
        required=True,
        help=f"Session type. Must be one of {program_types_str}.",
    )

    parser.add_argument(
        "--students",
        type=str,
        nargs="*",
        help="Space delimited students to process.",
    )
    parser.add_argument(
        "--teachers",
        type=str,
        nargs="*",
        help="Space delimited teachers to process.",
    )

    parser.add_argument("--regex", type=str, help="Regex to filter filenames.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Max number of files to process. Value of 0 indicates no limit.",
        default=0,
    )
    parser.add_argument(
        "--makeviz",
        action="store_true",
        help="If set, creates a demo video.",
    )
    parser.add_argument(
        "--disable-vision",
        action="store_true",
        help="Disables vision pipeline. Dev flag.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Disables actual execution of the graph..",
    )
    args = parser.parse_args()

    video_config.ENABLE_VISION = not args.disable_vision

    logging_utils.setup_logging()
    dotenv.load_dotenv()
    _main(
        regex=args.regex,
        limit_files=args.limit,
        makeviz=args.makeviz,
        dry_run=args.dry_run,
        program=args.program,
        students=args.students,
        teachers=args.teachers,
    )
