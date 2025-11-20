# TODO: Rename student_flow.py and related shell scripts, and update all docs.
import argparse
import itertools
import logging
import os

from . import video_config
from ..flow import process_graph
from .student_flow_nodes import highlights_persister
from .student_flow_nodes import hiring_highlight_curator as hhc
from .student_flow_nodes import hiring_movie_compiler
from .utils import logging_utils
from .video_flow_nodes import video_flow_types

_OUTDIR = video_config.VIDEO_SUMMARIES_DIR / "CompiledHighlights"


def _main(
    program: video_flow_types.ProgramType,
    movie_type: video_flow_types.CompilationType,
    students: list[str],
    teachers: list[str],
    force_rerun: bool,
    target_duration: float,
):
    if (
        movie_type
        in [
            video_flow_types.CompilationType.STUDENT_HIRING,
            video_flow_types.CompilationType.STUDENT_RESUME,
        ]
        and args.teachers
    ) or (
        movie_type
        in [
            video_flow_types.CompilationType.TEACHER_HIRING,
        ]
        and args.students
    ):
        raise ValueError(f"Invalid teachers/students specified for {movie_type}")

    persist_dir = _OUTDIR / "logs" / movie_type.value

    # Next Node ID: 6
    graph = process_graph.ProcessGraph()
    student_const = graph.add_constant_node(0, name="students_const", type=str | None)
    teacher_const = graph.add_constant_node(4, name="teachers_const", type=str | None)

    # Simply persists the evals from video_flow.
    # Can be useful to recreate any previous video.
    evals_persisted_node = graph.add_node(
        5,
        highlights_persister.EvalsPersister,
        {
            "program": program,
            "movie_type": movie_type,
            "student": student_const,
            "teacher": teacher_const,
            "log_dir": str(persist_dir),
        },
        force=True,  # DO NOT change this. Instead use the force_rerun arg.
    )

    # The HighlightCurator analyzes video content to find important segments,
    # filters them based on criteria like importance score and speaker time,
    # removes overlapping segments, and selects the best highlights for compilation.
    highlight_curate_node = graph.add_node(
        1,
        hhc.HighlightCurator,
        {
            # TODO: Fill evals_out from optional arg pointing to previous file to recreate movie from same highlights.
            "program": program,
            "movie_type": movie_type,
            "evals_out": evals_persisted_node,
            "student": student_const,
            "teacher": teacher_const,
            "out_dir": str(_OUTDIR),
            "log_dir": str(persist_dir),
            "target_duration": target_duration,
        },
        version=4,
    )
    movie_compile_node = graph.add_node(
        3,
        hiring_movie_compiler.HiringMovieCompiler,
        {
            "movie_type": movie_type,
            "highlights_log_file": highlight_curate_node,
        },
    )

    os.makedirs(persist_dir, exist_ok=True)

    # Process each student or teacher (exactly one of student, teacher will be populated).
    for student, teacher in itertools.chain(
        ((student, None) for student in students),
        ((None, teacher) for teacher in teachers),
    ):
        persist_fname = f"{student or teacher}.process_graph.json"

        logging.info(f"Processing: {persist_fname}")
        graph.persist(str(persist_dir / persist_fname))

        result_timestamp = highlight_curate_node.result_timestamp
        logging.info(f"{result_timestamp=}")

        # Check if computation should be skipped (if up to date).
        if result_timestamp is not None:
            source_timestamp = (
                highlights_persister.EvalsPersister.check_source_timestamp(
                    program=program,
                    movie_type=movie_type,
                    student=student,
                    teacher=teacher,
                )
            )
            logging.info(f"{source_timestamp=}")
            if source_timestamp <= result_timestamp:
                if not force_rerun:
                    logging.info("Skipping because up to date.")
                    continue
                else:
                    logging.info("Forcing a rerun despite being up to date.")

        student_const.set_value(student)
        teacher_const.set_value(teacher)
        graph.run_upto([movie_compile_node])

        if video_config.TESTING_MODE:
            # Just do 1 for debugging.
            break

    video_config.repeated_warnings()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Flow Pipeline")
    # List the options from the video_flow_types.CompilationType enum.
    valid_types: list[str] = [
        member.value
        for member in video_flow_types.CompilationType.__members__.values()
        if member != video_flow_types.CompilationType.UNKNOWN
    ]
    parser.add_argument(
        "--program",
        type=video_flow_types.ProgramType,
        required=True,
        help=f"Program type.",
    )
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
        type=video_flow_types.CompilationType,
        required=True,
        help=f"Type of movie compilation to perform. Can be one of: {valid_types}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of the pipeline even if results are up to date.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,  # 5 minutes
        help=f"Number of seconds to target for the movie.",
    )
    args = parser.parse_args()

    logging_utils.setup_logging()

    _main(
        program=args.program,
        movie_type=args.movie_type,
        students=args.students,
        teachers=args.teachers,
        force_rerun=args.force,
        target_duration=args.duration,
    )
