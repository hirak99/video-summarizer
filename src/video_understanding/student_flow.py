# TODO: Rename student_flow.py and related shell scripts, and update all docs.
import argparse
import itertools
import logging
import os

from . import video_config
from ..flow import process_graph
from ..flow import process_node
from .student_flow_nodes import compile_options
from .student_flow_nodes import eval_template_maker
from .student_flow_nodes import hiring_highlight_curator as hhc
from .student_flow_nodes import hiring_movie_compiler
from .utils import logging_utils
from .video_flow_nodes import student_eval_type

from typing import Literal

_OUTDIR = video_config.VIDEO_SUMMARIES_DIR / "CompiledHighlights"

# Don't change the Literal type.
# It is set to nag you to change it back to False, after you temporarily set this to True.
_FORCE: Literal[False] = False


def _main(students: list[str], teachers: list[str]):
    persist_dir = _OUTDIR / "logs" / compile_options.COMPILATION_TYPE.value

    # Next Node ID: 5
    graph = process_graph.ProcessGraph()
    student_const = graph.add_node(
        0, process_node.constant("students_const"), {"value": None}
    )
    teacher_const = graph.add_node(
        4, process_node.constant("teachers_const"), {"value": None}
    )
    highlight_curate_node = graph.add_node(
        1,
        hhc.HighlightCurator,
        {
            "student": student_const,
            "teacher": teacher_const,
            "out_dir": str(_OUTDIR),
            "log_dir": str(persist_dir),
        },
        version=4,
        force=True,  # DO NOT change this. Instead use the _FORCE variable.
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

    # Exactly one of student, teaacher will be populated.
    for student, teacher in itertools.chain(
        ((student, None) for student in students),
        ((None, teacher) for teacher in teachers),
    ):
        persist_fname = f"{student or teacher}.process_graph.json"

        logging.info(f"Processing: {persist_fname}")
        graph.persist(str(persist_dir / persist_fname))

        result_timestamp = highlight_curate_node.result_timestamp
        source_timestamp = hhc.HighlightCurator.source_timestamp(
            student=student, teacher=teacher
        )
        logging.info(f"{result_timestamp=}, {source_timestamp=}")
        # Check if computation should be skipped.
        if not _FORCE:
            if result_timestamp is not None and source_timestamp is not None:
                if source_timestamp <= result_timestamp:
                    logging.info("Skipping because up to date.")
                    continue

        student_const.set("value", student)
        teacher_const.set("value", teacher)
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
    parser.add_argument(
        "--movie-type",
        type=str,
        required=True,
        help=f"Type of movie compilation to perform. Can be one of: {valid_types}.",
    )
    args = parser.parse_args()

    logging_utils.setup_logging()
    compile_options.set_compilation_type(args.movie_type)
    if (
        compile_options.COMPILATION_TYPE
        in [
            student_eval_type.CompilationType.STUDENT_HIRING,
            student_eval_type.CompilationType.STUDENT_RESUME,
        ]
        and args.teachers
    ) or (
        compile_options.COMPILATION_TYPE
        in [
            student_eval_type.CompilationType.TEACHER_HIRING,
        ]
        and args.students
    ):
        raise ValueError(
            f"Invalid teachers/students specified for {compile_options.COMPILATION_TYPE}"
        )

    _main(students=args.students, teachers=args.teachers)
