import json

from . import student_eval_type
from ...flow import process_node
from ..utils import movie_compiler

from typing import override


class StudentMovieCompiler(process_node.ProcessNode):
    @override
    def process(
        self, source_file: str, highlights_file: str, out_file_stem: str
    ) -> str:
        out_file = out_file_stem + ".student_highlights.mp4"

        with open(highlights_file) as file:
            evals: list[student_eval_type.StudentEvalT] = json.load(file)

        compiler = movie_compiler.MovieCompiler()

        highlights: list[movie_compiler.HighlightsT] = []
        for eval in sorted(evals, key=lambda x: x["start"]):
            eval_sign = "+" if eval["example_of"] == "strength" else "-"
            highlights.append(
                {
                    "description": f"AI comment: ({eval_sign}{eval['importance']}) {eval['comment']}",
                    "start_time": eval["start"],
                    "end_time": eval["end"],
                    "captions": [],  # TODO: Can add captions here.
                }
            )

        compiler.add_highlight_group(source_file, "", highlights)
        compiler.combine(out_file)

        return out_file
