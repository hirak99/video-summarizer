import logging
import os

from . import video_config
from ..domain_specific import domain_config
from ..flow import process_graph
from ..flow import process_node
from .student_flow_nodes import eval_template_maker
from .student_flow_nodes import hiring_highlight_curator as hhc
from .student_flow_nodes import hiring_movie_compiler
from .utils import logging_utils

_OUTDIR = video_config.VIDEO_SUMMARIES_DIR / "StudentHighlights"
_PERSIST_DIR = _OUTDIR / "logs"


def main():
    graph = process_graph.ProcessGraph()
    student_const = graph.add_node(0, process_node.constant("Student"), {"value": None})
    highlight_curate_node = graph.add_node(
        1,
        hhc.HighlightCurator,
        {
            "file_search_term": student_const,
            "out_dir": str(_OUTDIR),
            "log_dir": str(_PERSIST_DIR),
        },
        version=3,
    )
    eval_template_node = graph.add_node(
        2,
        eval_template_maker.EvalTermplateMaker,
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

    os.makedirs(_PERSIST_DIR, exist_ok=True)

    for student_id in domain_config.STUDENTS:
        logging.info(f"Processing student: {student_id}")
        graph.persist(str(_PERSIST_DIR / f"{student_id}.process_graph.json"))
        student_const.set("value", student_id)
        graph.run_upto([movie_compile_node, eval_template_node])

        if video_config.TESTING_MODE:
            # Just do 1 for debugging.
            break

    video_config.repeated_warnings()


if __name__ == "__main__":
    logging_utils.setup_logging()
    main()
