import logging
import pathlib

from . import hiring_highlight_curator as hhc
from .. import video_config
from ...flow import process_node
from ..utils import manual_labels_manager
from ..utils import misc_utils
from ..utils import movie_compiler
from ..video_flow_nodes import ocr_detector

from typing import override


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
        compiler = movie_compiler.MovieCompiler()
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

            last_path = highlights[index - 1].movie if index > 0 else None
            next_path = (
                highlights[index + 1].movie if index + 1 < len(highlights) else None
            )

            out_stem = misc_utils.get_output_stem(
                segment.movie, video_config.VIDEOS_DIR, video_config.WORKSPACE_DIR
            )

            annotation_blur = manual_labels_manager.AnnotationBlur(segment.movie)

            compiler.add_highlight(
                source_movie_file=segment.movie,
                title=title,
                highlight=highlight,
                title_fade_in=segment.movie != last_path,
                title_fade_out=segment.movie != next_path,
                temp_clip_hash=segment.fingerprint,  # To invalidate any cached clips.
                blur_json_file=out_stem + ocr_detector.FILE_SUFFIX,
                frame_processor=annotation_blur.process_frame,
            )
            if video_config.TESTING_MODE:
                break

        compiler.combine(str(out_file))
