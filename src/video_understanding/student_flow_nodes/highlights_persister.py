import functools
import hashlib
import json
import os

import pydantic

from . import video_graph_node_getter
from ...flow import process_node
from ..utils import misc_utils
from ..utils import video_file_search
from ..video_flow_nodes import role_based_captioner
from ..video_flow_nodes import video_flow_types

from typing import override


class HighlightData(pydantic.BaseModel):
    movie: str
    evaluation: video_flow_types.HighlightsT
    captions_file: str

    @functools.cached_property
    def captions(self) -> list[role_based_captioner.RoleAwareCaptionT]:
        with open(self.captions_file, "r") as file:
            return json.load(file)

    @property
    def duration(self) -> float:
        return self.evaluation["end"] - self.evaluation["start"]

    def _speaker_time(self, speaker: str) -> float:
        speaking_time = 0.0
        for caption in self.captions:
            if caption["speaker"] == speaker:
                # speaking_time for duration intersected with self.evaluation time.
                start = max(caption["interval"][0], self.evaluation["start"])
                end = min(caption["interval"][1], self.evaluation["end"])
                speaking_time += max(0, end - start)

        return speaking_time

    @functools.cached_property
    def student_speaking(self) -> float:
        return self._speaker_time("Student")

    @functools.cached_property
    def teacher_speaking(self) -> float:
        return self._speaker_time("Teacher")

    # Point system.
    @property
    def points(self) -> float:
        student_ratio: float
        if self.student_speaking == 0:
            student_ratio = 0
        else:
            student_ratio = self.student_speaking / (
                self.student_speaking + self.teacher_speaking
            )

        return (
            self.evaluation["importance"]
            + (10 * student_ratio)
            # Penalize small durations. Smaller durations get even more penalty.
            - min(-5 + self.duration, 0) ** 2 / 2.5 * 2  # From -20 to 0 over first 5s.
        )

    @functools.cached_property
    def fingerprint(self) -> str:
        # Compute a unique 7-char fingerprint for run_id.
        # We will use this for marking and logging manual reviews.
        fingerprint_info = self.model_dump_json()
        return hashlib.sha256(
            json.dumps(fingerprint_info, sort_keys=True).encode("utf-8")
        ).hexdigest()[:7]


HighlightsListT = pydantic.RootModel[list[HighlightData]]


@functools.lru_cache(maxsize=1)
def _get_all_video_fnames(
    *, program: video_flow_types.ProgramType, student: str | None, teacher: str | None
) -> list[str]:
    if student is None and teacher is None:
        raise ValueError("Either student or teacher must be specified.")
    students = [student] if student is not None else []
    teachers = [teacher] if teacher is not None else []
    if students and teachers:
        raise ValueError("Cannot have both students and teachers specified.")
    return video_file_search.all_video_files(
        program=program, students_list=students, teachers_list=teachers
    )


class EvalsPersister(process_node.ProcessNode):

    @classmethod
    def check_source_timestamp(
        cls,
        program: video_flow_types.ProgramType,
        movie_type: video_flow_types.CompilationType,
        *,
        student: str | None,
        teacher: str | None,
    ) -> float:
        """Ensures all highlights exists, and returns latest timestamp."""
        latest_timestamp = 0.0
        for video_fname in _get_all_video_fnames(
            program=program, student=student, teacher=teacher
        ):
            video_nodes = video_graph_node_getter.get_video_graph_nodes(
                movie_type=movie_type, video_fname=video_fname
            )
            highlights_node = video_nodes.current_highlights_node
            result_timestamp = misc_utils.ensure_not_none(
                highlights_node.result_timestamp,
                err="Highlights node not computed",
            )
            latest_timestamp = max(latest_timestamp, result_timestamp)
        return latest_timestamp

    @override
    def process(
        self,
        program: video_flow_types.ProgramType,
        movie_type: video_flow_types.CompilationType,
        student: str | None,
        teacher: str | None,
        log_dir: str,
    ) -> str:
        eval_segments = HighlightsListT([])

        for video_fname in _get_all_video_fnames(
            program=program, student=student, teacher=teacher
        ):
            video_nodes = video_graph_node_getter.get_video_graph_nodes(
                movie_type=movie_type, video_fname=video_fname
            )

            captions_file = misc_utils.ensure_not_none(
                video_nodes.graph.role_based_caption_node.result,
                err="Role based captions not computed",
            )
            if not isinstance(captions_file, str):
                raise ValueError(
                    f"Role based captions file not computed for {video_fname}"
                )

            highlights_node_result = misc_utils.ensure_not_none(
                video_nodes.current_highlights_node.result,
                err="Highlights node not computed",
            )

            with open(highlights_node_result, "r") as file:
                evaluations: list[video_flow_types.HighlightsT] = json.load(file)
                for evaluation in evaluations:
                    # Sometimes the comment is not capitalized in LLM output.
                    evaluation["comment"] = evaluation["comment"].capitalize()
                    eval_segments.root.append(
                        HighlightData(
                            movie=video_fname,
                            captions_file=captions_file,
                            evaluation=evaluation,
                        )
                    )

        fingerprint = misc_utils.fingerprint(eval_segments.model_dump_json())
        out_file_basename = f"segments_{student or teacher}_{fingerprint}.json"
        out_fname = os.path.join(log_dir, out_file_basename)

        with open(out_fname, "w") as f:
            json.dump(
                {
                    "segments": eval_segments.model_dump_json(),
                    "fingerprint": fingerprint,
                },
                f,
            )

        return out_fname
