import argparse

import dotenv

from . import video_config
from . import video_flow_executor
from .utils import logging_utils


def _main(iregex: str | None, limit_files: int, makeviz: bool, dry_run: bool):
    video_graph = video_flow_executor.VideoFlowExecutor(
        makeviz=makeviz, dry_run=dry_run
    )
    video_graph.process(iregex, limit_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files")
    parser.add_argument(
        "--iregex", type=str, help="Case-insensitive regex to filter filenames."
    )
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
        iregex=args.iregex,
        limit_files=args.limit,
        makeviz=args.makeviz,
        dry_run=args.dry_run,
    )
