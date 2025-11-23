import functools
import hashlib
import json
import logging
import os

from . import prompt_templates
from . import video_config
from ..flow import internal_graph_node
from ..flow import process_graph
from .video_flow_nodes import caption_visualizer
from .video_flow_nodes import custom_yolo_detector
from .video_flow_nodes import highlights_selector
from .video_flow_nodes import ocr_detector
from .video_flow_nodes import role_based_captioner
from .video_flow_nodes import role_identifier
from .video_flow_nodes import session_summarizer
from .video_flow_nodes import speaker_assigner
from .video_flow_nodes import transcriber
from .video_flow_nodes import transcription_refiner
from .video_flow_nodes import video_flow_types
from .video_flow_nodes import video_quality_profiler
from .video_flow_nodes import vision_processor
from .video_flow_nodes import voice_separator


def _file_checksum(source_file: str) -> str:
    hasher = hashlib.sha1()
    with open(source_file, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            hasher.update(byte_block)
    return hasher.hexdigest()


class VideoFlowGraph:
    def __init__(
        self, *, program: video_flow_types.ProgramType, makeviz: bool, dry_run: bool
    ):
        """Makes the graph but does not execute anything."""
        graph = process_graph.ProcessGraph(dry_run=dry_run)

        # Don't re-use purged node ids.
        # Next Id: 23
        # Id(s) deprecated: 3, 11, 16.
        self._source_file_const = graph.add_constant_node(
            0, name="Source Video", type=str
        )
        self._out_stem_const = graph.add_constant_node(1, name="Out Stem", type=str)

        video_quality_profile_node = graph.add_node(
            17,
            video_quality_profiler.VideoQualityProfiler,
            {
                "source_file": self._source_file_const,
                "out_file_stem": self._out_stem_const,
            },
            version=2,
        )
        custom_yolo_detect_node = graph.add_node(
            18,
            custom_yolo_detector.CustomYoloDetector,
            {
                "source_file": self._source_file_const,
                "out_file_stem": self._out_stem_const,
            },
            version=0,
        )
        del custom_yolo_detect_node  # Incomplete and likely to get dropped.
        transcribe_node = graph.add_node(
            2,
            transcriber.WhisperTranscribe,
            {
                "source_file": self._source_file_const,
                "out_file_stem": self._out_stem_const,
            },
            invalidate_before=1753438637,
        )
        diarize_node = graph.add_node(
            7,
            voice_separator.VoiceSeparator,
            {
                "source_file": self._source_file_const,
                "out_file_stem": self._out_stem_const,
            },
            version=2,
        )
        transcription_refine_node = graph.add_node(
            14,
            transcription_refiner.TranscriptionRefiner,
            {
                "captions_file": transcribe_node,
                "diarization_file": diarize_node,
                "out_file_stem": self._out_stem_const,
            },
            version=5,
        )
        speaker_assign_node = graph.add_node(
            6,
            speaker_assigner.SpeakerAssigner,
            {
                "captions_file": transcription_refine_node,
                "diarization_file": diarize_node,
                "out_file_stem": self._out_stem_const,
            },
            version=3,
        )
        role_identify_node = graph.add_node(
            8,
            role_identifier.RoleIdentifier,
            {
                "source_file": self._source_file_const,
                "word_captions_file": speaker_assign_node,
                "out_file_stem": self._out_stem_const,
            },
            version=4,
        )
        visualize_node = graph.add_node(
            5,
            caption_visualizer.CaptionVisualizer,
            {
                "source_file": self._source_file_const,
                "word_captions_file": speaker_assign_node,
                "diarization_file": diarize_node,
                "identified_roles": role_identify_node,
                "out_file_stem": self._out_stem_const,
            },
            invalidate_before=1749101896,
        )
        self.role_based_caption_node = graph.add_node(
            9,
            role_based_captioner.RoleBasedCaptionsNode,
            {
                "word_captions_file": speaker_assign_node,
                "identified_roles": role_identify_node,
                "out_file_stem": self._out_stem_const,
            },
            version=2,
        )
        self._vision_process_node = graph.add_node(
            13,
            vision_processor.VisionProcess,
            {
                "source_file": self._source_file_const,
                "role_aware_summary_file": self.role_based_caption_node,
                "out_file_stem": self._out_stem_const,
            },
            version=7,
        )
        if not video_config.ENABLE_VISION:
            self._vision_process_node = None

        self._session_summary_node = graph.add_node(
            22,
            session_summarizer.SessionSummarizer,
            {
                "program": program.value,
                "source_file": self._source_file_const,
                "role_aware_summary_file": self.role_based_caption_node,
                "scene_understanding_file": self._vision_process_node,
                "bad_video_segments_file": video_quality_profile_node,
            },
            version=2,
        )

        # Increase this if the HighlightsSelector logic changes.
        highlights_logic_ver = 0
        self.highlights_student_hiring = graph.add_node(
            10,
            highlights_selector.HighlightsSelector,
            {
                "compilation_type": video_flow_types.CompilationType.STUDENT_HIRING,
                "source_file": self._source_file_const,
                "role_aware_summary_file": self.role_based_caption_node,
                "scene_understanding_file": self._vision_process_node,
                "bad_video_segments_file": video_quality_profile_node,
                "out_file_stem": self._out_stem_const,
            },
            version=f"{highlights_logic_ver}.{prompt_templates.STUDENT_HIRING_PROMPT_VERSION}",
        )
        self.highlights_student_resume = graph.add_node(
            15,
            highlights_selector.HighlightsSelector,
            {
                "compilation_type": video_flow_types.CompilationType.STUDENT_RESUME,
                "source_file": self._source_file_const,
                "role_aware_summary_file": self.role_based_caption_node,
                "scene_understanding_file": self._vision_process_node,
                "bad_video_segments_file": video_quality_profile_node,
                "out_file_stem": self._out_stem_const,
            },
            version=f"{highlights_logic_ver}.{prompt_templates.STUDENT_RESUME_PROMPT_VERSION}",
        )
        self.highlights_teacher_hiring = graph.add_node(
            19,
            highlights_selector.HighlightsSelector,
            {
                "compilation_type": video_flow_types.CompilationType.TEACHER_HIRING,
                "source_file": self._source_file_const,
                "role_aware_summary_file": self.role_based_caption_node,
                "scene_understanding_file": self._vision_process_node,
                "bad_video_segments_file": video_quality_profile_node,
                "out_file_stem": self._out_stem_const,
            },
            version=f"{highlights_logic_ver}.{prompt_templates.TEACHER_HIRING_PROMPT_VERSION}",
        )
        # We are not yet generating teacher highlights.
        del self.highlights_teacher_hiring

        self.highlights_ftp = graph.add_node(
            20,
            highlights_selector.HighlightsSelector,
            {
                "compilation_type": video_flow_types.CompilationType.FTP_HIGHLIGHTS,
                "source_file": self._source_file_const,
                "role_aware_summary_file": self.role_based_caption_node,
                "scene_understanding_file": self._vision_process_node,
                "bad_video_segments_file": video_quality_profile_node,
                "out_file_stem": self._out_stem_const,
            },
            version=f"{highlights_logic_ver}.{prompt_templates.FTP_PROMPT_VERSION}",
        )

        self.ocr_detect_node = graph.add_node(
            12,
            ocr_detector.OcrDetector,
            {
                "source_file": self._source_file_const,
                "out_file_stem": self._out_stem_const,
            },
            version=2,
        )

        # Common nodes for all programs.
        # Program-specific nodes are added while running it.
        final_nodes: list[internal_graph_node.AddedNode] = [
            self.ocr_detect_node,
            self._session_summary_node,
        ]
        if makeviz:
            final_nodes.append(visualize_node)
        match program:
            case video_flow_types.ProgramType.FTP:
                final_nodes.append(self.highlights_ftp)
            case video_flow_types.ProgramType.PMA:
                final_nodes.append(self.highlights_student_hiring)
                final_nodes.append(self.highlights_student_resume)
            case _:
                raise ValueError(f"Unknown program: {program}")

        self.graph = graph
        self._final_nodes = final_nodes
        self._release_resources_after = [
            self.ocr_detect_node,
            transcribe_node,
            diarize_node,
            role_identify_node,
        ]

    def persist_graph_for(self, video_path: str):
        video_name_wo_ext = os.path.splitext(os.path.basename(video_path))[0]

        # Store each version in its dir as checksum.
        results_dir = os.path.join(
            video_config.WORKSPACE_DIR, video_name_wo_ext, _file_checksum(video_path)
        )

        os.makedirs(results_dir, exist_ok=True)

        persist_path = os.path.join(results_dir, "graph_state.json")

        self.graph.persist(persist_path)

        # Stuff will be appended to the stem, like `out_stem + ".transcription.json"`.
        out_stem = os.path.join(results_dir, "video_flow_data")

        # Set the constants.
        self._source_file_const.set_value(video_path)
        self._out_stem_const.set_value(out_stem)

    def run(self, *, all_files_to_process: list[str]):

        def prep_fn(file_no: int, video_path: str, count: int):
            logging.info(
                f"Processing {file_no + 1} of {count}:"
                f" {os.path.relpath(video_path, video_config.VIDEOS_DIR)}"
            )
            self.persist_graph_for(video_path)

        self.graph.process_batch(
            batch_items=all_files_to_process,
            run_nodes=self._final_nodes,
            prep_fn=functools.partial(prep_fn, count=len(all_files_to_process)),
            post_fn=lambda file_no, path: video_config.repeated_warnings(),
            release_resources_after=self._release_resources_after,
            # As there is little to no errors, we want to fail immediately.
            fault_tolerant=False,
        )

        video_config.repeated_warnings()

    def role_aware_captions(self) -> list[role_based_captioner.RoleAwareCaptionT]:
        result = self.role_based_caption_node.result
        if result is None:
            raise ValueError("Role aware captions not computed")
        with open(result) as f:
            return json.load(f)

    def scene_understanding_result(self) -> vision_processor.SceneListT | None:
        if self._vision_process_node is None:
            # Will not be initialized if ENABLE_VISION is False.
            return None
        result = self._vision_process_node.result
        if result is None:
            raise ValueError("Scene understanding not computed")
        with open(result) as f:
            return vision_processor.SceneListT.model_validate_json(f.read())
