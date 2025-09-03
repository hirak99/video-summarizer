import collections
import functools
import hashlib
import itertools
import json
import logging
import os

import pydantic

from . import compile_options
from . import video_graph_node_getter
from .. import video_config
from ...domain_specific import manual_overrides
from ...flow import process_node
from ..utils import file_conventions
from ..utils import misc_utils
from ..utils import movie_compiler
from ..video_flow_nodes import role_based_captioner
from ..video_flow_nodes import video_flow_types

from typing import override

# Remove clips if it's entirely within X second of beginning or end.
# Typically this may include convo on previous or next lesson.
_REMOVE_EDGES_SECS = 5


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


class HighlightsLog(pydantic.BaseModel):
    # This is the log of the chosen highlights.
    highlights: list[HighlightData]
    manual_labels_dir: str
    compiled_movie: str


# Detects overlapping highlights, and drops the ones with least scores.
# TODO: This should be unit-tested.
def _disjointify_highlights_in_same_file(
    highlights: list[HighlightData], overlap_threshold: float
) -> list[HighlightData]:
    highlights.sort(key=lambda x: x.evaluation["start"])

    result: list[HighlightData] = [highlights[0]]

    for cur_highlight in highlights[1:]:

        cur_start, cur_end = (
            cur_highlight.evaluation["start"],
            cur_highlight.evaluation["end"],
        )

        prev_start, prev_end = (
            result[-1].evaluation["start"],
            result[-1].evaluation["end"],
        )

        # Check if there is an overlap, and lengt of the overlap is more than overlap_threshold.
        # Check if there is an overlap.
        if cur_start < prev_end:
            # Calculate the length of the overlap.
            overlap_length = min(cur_end, prev_end) - max(cur_start, prev_start)

            assert overlap_length >= 0, "Check overlap_length computation"

            if overlap_length >= overlap_threshold:
                logging.info(f"Overlap of length {overlap_length} detected, handling.")
                # Overlap detected. Keep the highlight with higher points.
                if cur_highlight.points > result[-1].points:
                    result[-1] = cur_highlight
                # Else, the previous one is better or equal, so we drop the current one.
            else:
                # Overlap is less than threshold, consider them distinct.
                result.append(cur_highlight)
        else:
            # No overlap.
            result.append(cur_highlight)

    return result


def _disjointify_highlights(highlights: list[HighlightData]) -> list[HighlightData]:
    # Disjointify per path.
    highlights.sort(key=lambda x: x.movie)
    result: list[HighlightData] = []
    for path, group in itertools.groupby(highlights, key=lambda x: x.movie):
        del path  # Unused.
        result += _disjointify_highlights_in_same_file(
            list(group), overlap_threshold=0.0
        )
    return result


def _choose_highlights(
    highlights: list[HighlightData], target_duration: float
) -> list[HighlightData]:
    # Keep only highlights with importance.
    highlights = [x for x in highlights if x.evaluation["importance"] >= 5]
    logging.info(f"# important highlights: {len(highlights)}")

    # Manual exclusions.
    # Use manual_overrides.highlight_exclusion(filename, start, end).
    highlights = [
        x
        for x in highlights
        if not manual_overrides.is_clip_ineligible(
            x.movie, x.evaluation["start"], x.evaluation["end"]
        )
    ]

    # Exclude highlights at the edge.
    highlights = [
        x
        for x in highlights
        if not (
            x.evaluation["end"] <= _REMOVE_EDGES_SECS
            or x.evaluation["start"]
            >= movie_compiler.get_movie_duration(x.movie) - _REMOVE_EDGES_SECS
        )
    ]
    logging.info(f"# highlights not at the edge of movie: {len(highlights)}")

    # Remove overlaps.
    highlights = _disjointify_highlights(highlights)
    logging.info(f"# highlights after disjointifying: {len(highlights)}")

    # Now we come to the heart of the selection.

    # Choose from the top.
    chosen: list[HighlightData] = []
    total_duration = 0.0
    # To penalize if there is not enough variety.
    session_durations: collections.defaultdict[str, float] = collections.defaultdict(
        float
    )
    n_sessions = len(set(x.movie for x in highlights))

    remaining = highlights.copy()
    while remaining:
        # Sort by most points. Also prioritize sessions with low counts, as 10 (unseen), 5 (1 times), 3.33 (2 times) etc..
        def session_based_points(x: HighlightData) -> float:
            if total_duration == 0:
                return 0
            # Compute total time for this session / averaged time per sessions.
            # Representation is between 0 and n_sessions.
            # 1 means evenly spaced.
            session_representation = session_durations[x.movie] / (
                total_duration / n_sessions
            )
            # If session is not represented, prioritize it. Or, if represented more, suppress it.
            return -15 * session_representation

        remaining.sort(key=lambda x: x.points + session_based_points(x), reverse=True)
        highlight = remaining.pop(0)
        chosen.append(highlight)
        total_duration = (
            total_duration + highlight.duration + 2 * movie_compiler.DEFAULT_FADE_TIME
        )
        session_durations[highlight.movie] += highlight.duration

        # Break after adding - allow it to be slightly longer than target duration.
        if total_duration >= target_duration:
            break

    logging.info(f"Total duration: {total_duration:0.2f}")
    return chosen


@functools.lru_cache(maxsize=1)
def _get_all_video_fnames(*, student: str | None, teacher: str | None) -> list[str]:
    if student is None and teacher is None:
        raise ValueError("Either student or teacher must be specified.")
    students = [student] if student is not None else []
    teachers = [teacher] if teacher is not None else []
    if students and teachers:
        raise ValueError("Cannot have both students and teachers specified.")
    return video_config.all_video_files(students=students, teachers=teachers)


class HighlightCurator(process_node.ProcessNode):

    @classmethod
    def check_source_timestamp(
        cls, *, student: str | None, teacher: str | None
    ) -> float:
        """Ensures all highlights exists, and returns latest timestamp."""
        latest_timestamp = 0.0
        for video_fname in _get_all_video_fnames(student=student, teacher=teacher):
            video_nodes = video_graph_node_getter.get_video_graph_nodes(video_fname)
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
        student: str | None,
        teacher: str | None,
        log_dir: str,
        out_dir: str,
        target_duration: float,
    ) -> str:
        eval_segments: list[HighlightData] = []

        for video_fname in _get_all_video_fnames(student=student, teacher=teacher):
            video_nodes = video_graph_node_getter.get_video_graph_nodes(video_fname)

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
                    eval_segments.append(
                        HighlightData(
                            movie=video_fname,
                            captions_file=captions_file,
                            evaluation=evaluation,
                        )
                    )

        logging.info(f"# highlights curated by LLM: {len(eval_segments)}")
        highlights = _choose_highlights(eval_segments, target_duration)
        logging.info(f"# highlights chosen: {len(highlights)}")

        # Sort by required order of the session files.
        highlights.sort(
            key=lambda x: (file_conventions.sort_key(x.movie), x.evaluation["start"])
        )

        # Drop segments with manually blocked hashes.
        highlights = [
            x
            for x in highlights
            if x.fingerprint not in manual_overrides.BAD_HIRING_SEGMENTS
        ]

        # Compute fingerprint to trace this file.
        full_fingerprint_info = "_".join(x.fingerprint for x in highlights)
        full_fingerprint = hashlib.sha256(
            json.dumps(full_fingerprint_info, sort_keys=True).encode("utf-8")
        ).hexdigest()[:7]

        # Compiled movie name.
        movie_type_str = compile_options.COMPILATION_TYPE.value
        out_file_basename = f"{student or teacher}_{movie_type_str}_v{video_config.VERSION}_{full_fingerprint}"
        os.makedirs(out_dir, exist_ok=True)
        movie_name = os.path.join(out_dir, f"{out_file_basename}.mp4")

        # Save the highlights and info in log.
        highlights_log = HighlightsLog(
            highlights=highlights,
            manual_labels_dir=os.path.realpath(video_config.MANUAL_LABELS_DIR),
            compiled_movie=movie_name,
        )

        highlights_out = os.path.join(
            log_dir, f"{out_file_basename}.curated_highlights.json"
        )
        with open(highlights_out, "w") as file:
            json.dump(highlights_log.model_dump(), file, indent=2)

        return highlights_out
