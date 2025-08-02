import argparse
import logging
import os
import re

import dotenv

from . import video_config
from ..domain_specific import domain_config
from ..flow import internal_graph_node
from ..flow import process_graph
from ..flow import process_node
from .utils import logging_utils
from .utils import misc_utils
from .video_flow_nodes import caption_visualizer
from .video_flow_nodes import ocr_detector
from .video_flow_nodes import role_based_captioner
from .video_flow_nodes import role_identifier
from .video_flow_nodes import speaker_assigner
from .video_flow_nodes import student_evaluator
from .video_flow_nodes import student_movie_compiler
from .video_flow_nodes import transcriber
from .video_flow_nodes import vision_processor
from .video_flow_nodes import voice_separator


def main(iregex: str | None, limit_files: int, makeviz: bool, dry_run: bool):
    graph = process_graph.ProcessGraph(dry_run=dry_run)

    # Don't re-use purged node ids.
    # Next Id: 14
    # 3 is deprecated.
    source_file_const = graph.add_node(
        0, process_node.constant("Source Video"), {"value": ""}
    )
    out_stem_const = graph.add_node(1, process_node.constant(), {"value": ""})
    transcribe_node = graph.add_node(
        2,
        transcriber.WhisperTranscribe,
        {"source_file": source_file_const, "out_file_stem": out_stem_const},
        invalidate_before=1753438637,
    )
    diarize_node = graph.add_node(
        7,
        voice_separator.VoiceSeparator,
        {"source_file": source_file_const, "out_file_stem": out_stem_const},
        invalidate_before=1751002018,
    )
    speaker_assign_node = graph.add_node(
        6,
        speaker_assigner.SpeakerAssigner,
        {
            "captions_file": transcribe_node,
            "diarization_file": diarize_node,
            "out_file_stem": out_stem_const,
        },
        invalidate_before=1748629535,
    )
    role_identify_node = graph.add_node(
        8,
        role_identifier.RoleIdentifier,
        {
            "word_captions_file": speaker_assign_node,
            "out_file_stem": out_stem_const,
        },
        invalidate_before=1749284285,
    )
    visualize_node = graph.add_node(
        5,
        caption_visualizer.CaptionVisualizer,
        {
            "source_file": source_file_const,
            "word_captions_file": speaker_assign_node,
            "diarization_file": diarize_node,
            "identified_roles": role_identify_node,
            "out_file_stem": out_stem_const,
        },
        invalidate_before=1749101896,
    )
    role_based_caption_node = graph.add_node(
        9,
        role_based_captioner.RoleBasedCaptionsNode,
        {
            "word_captions_file": speaker_assign_node,
            "identified_roles": role_identify_node,
            "out_file_stem": out_stem_const,
        },
    )
    vision_process_node = graph.add_node(
        13,
        vision_processor.VisionProcess,
        {
            "source_file": source_file_const,
            "role_aware_summary_file": role_based_caption_node,
            "out_file_stem": out_stem_const,
        },
        version=4,
    )
    student_evaluate_node = graph.add_node(
        10,
        student_evaluator.StudentEvaluator,
        {
            "source_file": source_file_const,
            "role_aware_summary_file": role_based_caption_node,
            "scene_understanding_file": vision_process_node,
            "out_file_stem": out_stem_const,
        },
        version=3,
    )
    student_movie_compile_node = graph.add_node(
        11,
        student_movie_compiler.StudentMovieCompiler,
        {
            "source_file": source_file_const,
            "highlights_file": student_evaluate_node,
            "out_file_stem": out_stem_const,
        },
    )
    del student_movie_compile_node  # TBR
    ocr_detect_node = graph.add_node(
        12,
        ocr_detector.OcrDetector,
        {"source_file": source_file_const, "out_file_stem": out_stem_const},
        invalidate_before=1749315690,
    )

    # Final target node(s) for all files.
    final_nodes: list[internal_graph_node.AddedNode] = [
        student_evaluate_node,
        ocr_detect_node,
    ]
    if makeviz:
        final_nodes.append(visualize_node)

    all_files_to_process: list[str] = []
    for dirpath, dirnames, filenames in os.walk(video_config.VIDEOS_DIR):
        del dirnames  # Unused.
        for filename in filenames:
            if filename.endswith(".mkv") or filename.endswith(".mp4"):
                path = os.path.join(dirpath, filename)
                if iregex is not None and not re.search(iregex, path, re.IGNORECASE):
                    continue
                for student_id in domain_config.STUDENTS:
                    if re.search(rf"(\b|_){student_id}\b", path):
                        # Add this path.
                        break
                else:
                    if iregex is not None:
                        logging.info(
                            f"Ignoring iregex match {path!r}, because not in STUDENTS."
                        )
                    continue
                all_files_to_process.append(path)
    if limit_files:
        all_files_to_process = all_files_to_process[:limit_files]

    def prep_fn(file_no: int, path: str):
        logging.info(
            f"Processing {file_no + 1} of {len(all_files_to_process)}:"
            f" {os.path.relpath(path, video_config.VIDEOS_DIR)}"
        )
        out_stem = misc_utils.get_output_stem(
            path, video_config.VIDEOS_DIR, video_config.WORKSPACE_DIR
        )

        graph.persist(out_stem + ".process_graph_state.json")
        source_file_const.set("value", path)
        out_stem_const.set("value", out_stem)

    graph.process_batch(
        batch_items=all_files_to_process,
        final_nodes=final_nodes,
        prep_fn=prep_fn,
        post_fn=lambda file_no, path: video_config.repeated_warnings(),
        release_after_nodes=[
            transcribe_node,
            diarize_node,
            role_identify_node,
        ],
        # As there is little to no errors, we want to fail immediately.
        fault_tolerant=False,
    )

    video_config.repeated_warnings()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files")
    parser.add_argument(
        "--iregex", type=str, help="Case-insensitive regex to filter filenames."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max number of files to process. Value of 0 indicates no limit.",
        default=0,
    )
    parser.add_argument(
        "--makeviz",
        action="store_true",
        help="If set, creates a demo video.",
    )
    parser.add_argument(
        "--disable-vision",
        action="store_true",
        help="Disables vision pipeline. Dev flag.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Disables actual execution of the graph..",
    )
    args = parser.parse_args()

    video_config.ENABLE_VISION = not args.disable_vision

    logging_utils.setup_logging()
    dotenv.load_dotenv()
    main(
        iregex=args.iregex,
        limit_files=args.limit,
        makeviz=args.makeviz,
        dry_run=args.dry_run,
    )
