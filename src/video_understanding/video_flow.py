import argparse

import dotenv

from . import video_config
from . import video_flow_graph
from .utils import logging_utils


def _main(
    *,
    regex: str | None,
    limit_files: int,
    makeviz: bool,
    dry_run: bool,
    file_search_terms: list[str]
):
    all_files_to_process = video_config.all_video_files(
        regex=regex, words=file_search_terms
    )

    if limit_files:
        all_files_to_process = all_files_to_process[:limit_files]

    video_graph = video_flow_graph.VideoFlowGraph(makeviz=makeviz, dry_run=dry_run)
    video_graph.run(all_files_to_process=all_files_to_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files")

    parser.add_argument(
        "--students",
        type=str,
        nargs="*",
        default=[],
        help="Space delimited students to process.",
    )
    parser.add_argument(
        "--teachers",
        type=str,
        nargs="*",
        default=[],
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

    # Words that a file must contain to be eligible.
    all_words = args.students + args.teachers

    video_config.ENABLE_VISION = not args.disable_vision

    logging_utils.setup_logging()
    dotenv.load_dotenv()
    _main(
        regex=args.regex,
        limit_files=args.limit,
        makeviz=args.makeviz,
        dry_run=args.dry_run,
        file_search_terms=all_words,
    )
