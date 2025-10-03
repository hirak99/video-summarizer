import collections
import datetime
import itertools
import json
import logging
import os

import pydantic

from . import compile_options
from . import highlights_persister
from .. import video_config
from ...domain_specific import manual_overrides
from ...flow import process_node
from ..utils import file_conventions
from ..utils import movie_compiler

from typing import override

# Remove clips if it's entirely within X second of beginning or end.
# Typically this may include convo on previous or next lesson.
_REMOVE_EDGES_SECS = 5

# All sections with importance below this will be dropped.
_IMPORTANCE_THRESHOLD = 6


class HighlightsLog(pydantic.BaseModel):
    # This is the log of the chosen highlights.
    highlights: list[highlights_persister.HighlightData]
    manual_labels_dir: str
    compiled_movie: str


# Detects overlapping highlights, and drops the ones with least scores.
# TODO: This should be unit-tested.
def _disjointify_highlights_in_same_file(
    highlights: list[highlights_persister.HighlightData], overlap_threshold: float
) -> list[highlights_persister.HighlightData]:
    highlights.sort(key=lambda x: x.evaluation["start"])

    result: list[highlights_persister.HighlightData] = [highlights[0]]

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


def _disjointify_highlights(
    highlights: list[highlights_persister.HighlightData],
) -> list[highlights_persister.HighlightData]:
    # Disjointify per path.
    highlights.sort(key=lambda x: x.movie)
    result: list[highlights_persister.HighlightData] = []
    for path, group in itertools.groupby(highlights, key=lambda x: x.movie):
        del path  # Unused.
        result += _disjointify_highlights_in_same_file(
            list(group), overlap_threshold=0.0
        )
    return result


def _choose_highlights(
    highlights: list[highlights_persister.HighlightData], target_duration: float
) -> list[highlights_persister.HighlightData]:
    # Keep only highlights with importance.
    highlights = [
        x for x in highlights if x.evaluation["importance"] >= _IMPORTANCE_THRESHOLD
    ]
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
    chosen: list[highlights_persister.HighlightData] = []
    total_duration = 0.0
    # To penalize if there is not enough variety.
    session_durations: collections.defaultdict[str, float] = collections.defaultdict(
        float
    )
    n_sessions = len(set(x.movie for x in highlights))

    remaining = highlights.copy()
    while remaining:
        # Sort by most points. Also prioritize sessions with low counts, as 10 (unseen), 5 (1 times), 3.33 (2 times) etc..
        def session_based_points(x: highlights_persister.HighlightData) -> float:
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


class HighlightCurator(process_node.ProcessNode):

    @override
    def process(
        self,
        evals_out: str,
        student: str | None,
        teacher: str | None,
        log_dir: str,
        out_dir: str,
        target_duration: float,
    ) -> str:

        with open(evals_out) as f:
            data = json.load(f)
            eval_segments = highlights_persister.HighlightsListT.model_validate_json(
                data["segments"]
            )
            evals_fingerprint: str = data["fingerprint"]

        logging.info(f"# highlights curated by LLM: {len(eval_segments.root)}")
        highlights = _choose_highlights(eval_segments.root, target_duration)
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

        # Compiled movie name.
        movie_type_str = compile_options.COMPILATION_TYPE.value
        time_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
        out_file_basename = f"{student or teacher}_{movie_type_str}_v{video_config.VERSION}_{evals_fingerprint}_{time_str}"
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
