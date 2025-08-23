import argparse
import logging
import os

from . import video_config
from ..domain_specific import domain_config
from ..flow import process_graph
from ..flow import process_node
from .student_flow_nodes import compile_options
from .student_flow_nodes import eval_template_maker
from .student_flow_nodes import hiring_highlight_curator as hhc
from .student_flow_nodes import hiring_movie_compiler
from .utils import logging_utils
from .video_flow_nodes import student_eval_type

_OUTDIR = video_config.VIDEO_SUMMARIES_DIR / "StudentHighlights"


def main():
    persist_dir = _OUTDIR / "logs" / compile_options.COMPILATION_TYPE.value

    graph = process_graph.ProcessGraph()
    student_const = graph.add_node(0, process_node.constant("Student"), {"value": None})
    highlight_curate_node = graph.add_node(
        1,
        hhc.HighlightCurator,
        {
            "file_search_term": student_const,
            "out_dir": str(_OUTDIR),
            "log_dir": str(persist_dir),
        },
        version=4,
        # TODO: Instead of always forcing, should check if any of the sources in video_flow changed.
        force=True,
    )
    eval_template_node = graph.add_node(
        2,
        eval_template_maker.EvalTemplateMaker,
        {
            "highlights_log_file": highlight_curate_node,
            "out_dir": str(_OUTDIR),
        },
        version=2,
    )
    movie_compile_node = graph.add_node(
        3,
        hiring_movie_compiler.HiringMovieCompiler,
        {
            "highlights_log_file": highlight_curate_node,
        },
    )

    os.makedirs(persist_dir, exist_ok=True)

    for student_id in domain_config.STUDENTS:
        logging.info(f"Processing student: {student_id}")
        graph.persist(str(persist_dir / f"{student_id}.process_graph.json"))
        student_const.set("value", student_id)
        graph.run_upto([movie_compile_node, eval_template_node])

        if video_config.TESTING_MODE:
            # Just do 1 for debugging.
            break

    video_config.repeated_warnings()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Flow Pipeline")
    # List the options from the student_eval_type.CompilationType enum.
    valid_types: list[str] = [
        member.value
        for member in student_eval_type.CompilationType.__members__.values()
        if member != student_eval_type.CompilationType.UNKNOWN
    ]
    parser.add_argument(
        "--movie-type",
        type=str,
        required=True,
        help=f"Type of movie compilation to perform. Can be one of: {valid_types}.",
    )
    args = parser.parse_args()

    logging_utils.setup_logging()
    compile_options.set_compilation_type(args.movie_type)
    main()
