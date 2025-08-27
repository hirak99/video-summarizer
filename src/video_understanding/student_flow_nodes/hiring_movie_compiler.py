import logging
import pathlib

from . import compile_options
from . import hiring_highlight_curator as hhc
from .. import video_config
from ...flow import process_node
from ..utils import manual_labels_manager
from ..utils import misc_utils
from ..utils import movie_compiler
from ..video_flow_nodes import ocr_detector

from typing import override

_EPSILON = 1e-3


class HiringMovieCompiler(process_node.ProcessNode):

    @override
    def process(self, highlights_log_file: str) -> str:
        with open(highlights_log_file, "r") as file:
            highlights_log = hhc.HighlightsLog.model_validate_json(file.read())

        self._compile_movie(
            highlights_log.highlights, out_file=highlights_log.compiled_movie
        )

        return highlights_log.compiled_movie

    def _compile_movie(
        self, highlights: list[hhc.HighlightData], out_file: str
    ) -> None:
        """Compile the chosen highlights into a movie."""
        compiler = movie_compiler.MovieCompiler(compile_options.get_movie_options())

        for index, segment in enumerate(highlights):
            logging.info(f"{index + 1} of {len(highlights)}: {segment.movie}")
            # title = video_globals.metadata_for_file(segment.path).description
            to_mmss = lambda secs: f"{int(secs // 60):02d}:{int(secs % 60):02d}"

            title = (
                "Session: "
                + pathlib.Path(segment.movie).stem
                + f" [{to_mmss(segment.evaluation['start'])}"
                + f" - {to_mmss(segment.evaluation['end'])}]"
            )

            eval_sign = " +" if segment.evaluation["example_of"] == "strength" else ""
            highlight: movie_compiler.HighlightsT = {
                "description": f"{index+1}.{segment.fingerprint} AI comment:{eval_sign} {segment.evaluation['comment']}",
                "start_time": segment.evaluation["start"],
                "end_time": segment.evaluation["end"],
                "captions": segment.captions,
            }

            # Following two blocks ensure -
            # 1. Title fade is skipped if the same movie plays consecutively.
            # 2. Fade is removed if there is no gap between segments.
            # Note: Reducing fade time to 0 causes a slight but audible glitch.
            title_fade_out = True
            fade_out_time = movie_compiler.DEFAULT_FADE_TIME
            next_segment = (
                highlights[index + 1] if index + 1 < len(highlights) else None
            )
            if next_segment is not None:
                if next_segment.movie == segment.movie:
                    logging.info("Skipping title fade out.")
                    title_fade_out = False
                    overlap_time = (
                        next_segment.evaluation["start"] - segment.evaluation["end"]
                    )
                    if overlap_time <= 2 * movie_compiler.DEFAULT_FADE_TIME + _EPSILON:
                        # If the overlap is too long, then there is some error.
                        if overlap_time > 2 * movie_compiler.DEFAULT_FADE_TIME + 1:
                            raise RuntimeError(
                                f"Overlap time too long: {overlap_time}s"
                            )
                        logging.info("Lowering fade out time.")
                        next_segment.evaluation["start"] = segment.evaluation["end"]
                        fade_out_time = 0.0

            title_fade_in = True
            fade_in_time = movie_compiler.DEFAULT_FADE_TIME
            last_segment = highlights[index - 1] if index > 0 else None
            if last_segment is not None:
                if last_segment.movie == segment.movie:
                    logging.info("Skipping title fade in.")
                    title_fade_in = False
                    # The time should have already been adjusted to match.
                    if (
                        abs(
                            last_segment.evaluation["end"] - segment.evaluation["start"]
                        )
                        <= _EPSILON
                    ):
                        logging.info("Lowering fade in time.")
                        fade_in_time = 0.0

            out_stem = misc_utils.get_output_stem(
                segment.movie, video_config.VIDEOS_DIR, video_config.WORKSPACE_DIR
            )

            annotation_blur = manual_labels_manager.AnnotationBlur(segment.movie)

            compiler.add_highlight(
                source_movie_file=segment.movie,
                title=title,
                highlight=highlight,
                title_fade_in=title_fade_in,
                title_fade_out=title_fade_out,
                fade_in_time=fade_in_time,
                fade_out_time=fade_out_time,
                temp_clip_hash=segment.fingerprint,  # To invalidate any cached clips.
                blur_json_file=out_stem + ocr_detector.FILE_SUFFIX,
                frame_processor=annotation_blur.process_frame,
            )
            if video_config.TESTING_MODE:
                break

        compiler.combine(str(out_file))
