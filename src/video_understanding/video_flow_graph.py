import functools
import json
import logging
import os

from . import video_config
from ..flow import internal_graph_node
from ..flow import process_graph
from ..flow import process_node
from .utils import misc_utils
from .video_flow_nodes import caption_visualizer
from .video_flow_nodes import highlights_selector
from .video_flow_nodes import ocr_detector
from .video_flow_nodes import role_based_captioner
from .video_flow_nodes import role_identifier
from .video_flow_nodes import speaker_assigner
from .video_flow_nodes import student_eval_type
from .video_flow_nodes import transcriber
from .video_flow_nodes import transcription_refiner
from .video_flow_nodes import vision_processor
from .video_flow_nodes import voice_separator


class VideoFlowGraph:
    def __init__(self, *, makeviz: bool, dry_run: bool):
        """Makes the graph but does not execute anything."""

        graph = process_graph.ProcessGraph(dry_run=dry_run)

        # Don't re-use purged node ids.
        # Next Id: 17
        # Id(s) deprecated: 3, 11.
        self._source_file_const = graph.add_node(
            0, process_node.constant("Source Video"), {"value": ""}
        )
        self._out_stem_const = graph.add_node(1, process_node.constant(), {"value": ""})
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
            invalidate_before=1751002018,
        )
        transcription_refine_node = graph.add_node(
            14,
            transcription_refiner.TranscriptionRefiner,
            {
                "captions_file": transcribe_node,
                "diarization_file": diarize_node,
                "out_file_stem": self._out_stem_const,
            },
        )
        speaker_assign_node = graph.add_node(
            6,
            speaker_assigner.SpeakerAssigner,
            {
                "captions_file": transcription_refine_node,
                "diarization_file": diarize_node,
                "out_file_stem": self._out_stem_const,
            },
            invalidate_before=1748629535,
        )
        role_identify_node = graph.add_node(
            8,
            role_identifier.RoleIdentifier,
            {
                "word_captions_file": speaker_assign_node,
                "out_file_stem": self._out_stem_const,
            },
            invalidate_before=1749284285,
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
        self._role_based_caption_node = graph.add_node(
            9,
            role_based_captioner.RoleBasedCaptionsNode,
            {
                "word_captions_file": speaker_assign_node,
                "identified_roles": role_identify_node,
                "out_file_stem": self._out_stem_const,
            },
        )
        self._vision_process_node = graph.add_node(
            13,
            vision_processor.VisionProcess,
            {
                "source_file": self._source_file_const,
                "role_aware_summary_file": self._role_based_caption_node,
                "out_file_stem": self._out_stem_const,
            },
            version=4,
        )
        if not video_config.ENABLE_VISION:
            self._vision_process_node = None
        self.highlights_student_hiring = graph.add_node(
            10,
            highlights_selector.HighlightsSelector,
            {
                "compilation_type": student_eval_type.CompilationType.HIRING,
                "source_file": self._source_file_const,
                "role_aware_summary_file": self._role_based_caption_node,
                "scene_understanding_file": self._vision_process_node,
                "out_file_stem": self._out_stem_const,
            },
            version=4,
        )
        self.highlights_student_resume = graph.add_node(
            15,
            highlights_selector.HighlightsSelector,
            {
                "compilation_type": student_eval_type.CompilationType.RESUME,
                "source_file": self._source_file_const,
                "role_aware_summary_file": self._role_based_caption_node,
                "scene_understanding_file": self._vision_process_node,
                "out_file_stem": self._out_stem_const,
            },
            version=3,
        )
        self.highlights_teacher_hiring = graph.add_node(
            16,
            highlights_selector.HighlightsSelector,
            {
                "compilation_type": student_eval_type.CompilationType.TEACHER_HIRING,
                "source_file": self._source_file_const,
                "role_aware_summary_file": self._role_based_caption_node,
                "scene_understanding_file": self._vision_process_node,
                "out_file_stem": self._out_stem_const,
            },
        )
        ocr_detect_node = graph.add_node(
            12,
            ocr_detector.OcrDetector,
            {
                "source_file": self._source_file_const,
                "out_file_stem": self._out_stem_const,
            },
            invalidate_before=1749315690,
        )

        # Final target node(s) for all files.
        final_nodes: list[internal_graph_node.AddedNode] = [
            self.highlights_student_hiring,
            self.highlights_student_resume,
            self.highlights_teacher_hiring,
            ocr_detect_node,
        ]
        if makeviz:
            final_nodes.append(visualize_node)

        self.graph = graph
        self._final_nodes = final_nodes
        self._release_resources_after = [
            transcribe_node,
            diarize_node,
            role_identify_node,
        ]

    def persist_graph_for(self, video_path: str):
        out_stem = misc_utils.get_output_stem(
            video_path, video_config.VIDEOS_DIR, video_config.WORKSPACE_DIR
        )
        self.graph.persist(out_stem + ".process_graph_state.json")
        self._source_file_const.set("value", video_path)
        self._out_stem_const.set("value", out_stem)

    def run(self, *, all_files_to_process: list[str]):

        def prep_fn(file_no: int, video_path: str, count: int):
            logging.info(
                f"Processing {file_no + 1} of {count}:"
                f" {os.path.relpath(video_path, video_config.VIDEOS_DIR)}"
            )
            self.persist_graph_for(video_path)

        self.graph.process_batch(
            batch_items=all_files_to_process,
            final_nodes=self._final_nodes,
            prep_fn=functools.partial(prep_fn, count=len(all_files_to_process)),
            post_fn=lambda file_no, path: video_config.repeated_warnings(),
            release_resources_after=self._release_resources_after,
            # As there is little to no errors, we want to fail immediately.
            fault_tolerant=False,
        )

        video_config.repeated_warnings()

    def role_aware_captions(self) -> list[role_based_captioner.RoleAwareCaptionT]:
        result = self._role_based_caption_node.result
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
