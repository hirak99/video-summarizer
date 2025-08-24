import collections
import dataclasses
import functools
import hashlib
import itertools
import json
import logging
import os

import pydantic

from . import compile_options
from .. import video_config
from .. import video_flow_graph
from ...domain_specific import manual_overrides
from ...flow import process_node
from ..utils import file_conventions
from ..utils import misc_utils
from ..utils import movie_compiler
from ..video_flow_nodes import role_based_captioner
from ..video_flow_nodes import student_eval_type

from typing import Any, override

_TARGET_DURATION = 5 * 60

# Remove clips if it's entirely within X second of beginning or end.
# Typically this may include convo on previous or next lesson.
_REMOVE_EDGES_SECS = 5


class HighlightData(pydantic.BaseModel):
    movie: str
    evaluation: student_eval_type.StudentEvalT
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


def _choose_highlights(highlights: list[HighlightData]) -> list[HighlightData]:
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
        if total_duration >= _TARGET_DURATION:
            break

    logging.info(f"Total duration: {total_duration:0.2f}")
    return chosen


@functools.lru_cache(maxsize=1)
def _get_all_video_fnames(file_search_term: str) -> list[str]:
    return video_config.all_video_files(students=[file_search_term])


@dataclasses.dataclass(frozen=True)
class _NodeResult:
    result: Any
    result_timestamp: float | None


class HighlightCurator(process_node.ProcessNode):

    # Class variable.
    _video_graph = video_flow_graph.VideoFlowGraph(makeviz=False, dry_run=True)

    @classmethod
    @functools.lru_cache(maxsize=256)
    def _get_highlights_node(cls, video_fname: str) -> _NodeResult:
        cls._video_graph.persist_graph_for(video_fname)

        match compile_options.COMPILATION_TYPE:
            case compile_options.COMPILATION_TYPE.HIRING:
                highlights_node = cls._video_graph.highlights_student_hiring
            case compile_options.COMPILATION_TYPE.RESUME:
                highlights_node = cls._video_graph.highlights_student_resume
            case _:
                raise ValueError(
                    f"Unknown compilation type: {compile_options.COMPILATION_TYPE}"
                )
        if highlights_node.result is None:
            raise ValueError(f"Highlights node result not computed for {video_fname}")
        # It's important to not return the highlights_node directly, since it is
        # going to be overwritten on another call to persist_graph_for(...).
        return _NodeResult(highlights_node.result, highlights_node.result_timestamp)

    @classmethod
    def source_timestamp(cls, file_search_term: str) -> float | None:
        latest_timestamp = 0.0
        for video_fname in _get_all_video_fnames(file_search_term):
            highlights_node = cls._get_highlights_node(video_fname)
            if highlights_node.result_timestamp is None:
                return None
            latest_timestamp = max(latest_timestamp, highlights_node.result_timestamp)
        return latest_timestamp

    @override
    def process(self, file_search_term: str, log_dir: str, out_dir: str) -> str:
        eval_segments: list[HighlightData] = []

        for video_fname in _get_all_video_fnames(file_search_term):
            out_stem = misc_utils.get_output_stem(
                video_fname, video_config.VIDEOS_DIR, video_config.WORKSPACE_DIR
            )

            captions_file = out_stem + role_based_captioner.FILE_SUFFIX

            highlights_node = self._get_highlights_node(video_fname)

            with open(highlights_node.result, "r") as file:
                evaluations: list[student_eval_type.StudentEvalT] = json.load(file)
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

        movie_type_str = compile_options.COMPILATION_TYPE.value
        out_file_basename = (
            f"{file_search_term}_{movie_type_str}_v{video_config.VERSION}"
        )

        logging.info(f"# highlights curated by LLM: {len(eval_segments)}")
        highlights = _choose_highlights(eval_segments)
        logging.info(f"# highlights chosen: {len(highlights)}")

        # Sort by required order of the session files.
        highlights.sort(
            key=lambda x: (file_conventions.sort_key(x.movie), x.evaluation["start"])
        )

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{out_file_basename}.mp4")

        # Drop segments with manually blocked hashes.
        highlights = [
            x
            for x in highlights
            if x.fingerprint not in manual_overrides.BAD_HIRING_SEGMENTS
        ]

        # Save the highlights and info in log.
        highlights_log = HighlightsLog(
            highlights=highlights,
            manual_labels_dir=os.path.realpath(video_config.MANUAL_LABELS_DIR),
            compiled_movie=out_file,
        )

        highlights_log_file = os.path.join(
            log_dir, f"{out_file_basename}.hiring_highlight_compiler.json"
        )
        with open(highlights_log_file, "w") as file:
            json.dump(highlights_log.model_dump(), file, indent=2)

        return highlights_log_file
