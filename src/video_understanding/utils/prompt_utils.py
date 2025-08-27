import logging

from .. import video_config
from ...domain_specific import manual_overrides
from ..utils import interval_scanner
from ..video_flow_nodes import role_based_captioner
from ..video_flow_nodes import video_quality_assessor
from ..video_flow_nodes import vision_processor


def caption_lines_for_prompt(
    source_file: str,
    role_aware_summary: list[role_based_captioner.RoleAwareCaptionT],
    scene_understanding: vision_processor.SceneListT | None,
    bad_segments: list[video_quality_assessor.BadSegment] | None = None,
    start: float | None = None,
    end: float | None = None,
) -> list[str]:

    def _caption_to_text(caption: role_based_captioner.RoleAwareCaptionT) -> str:
        return f"[{caption['interval'][0]:.1f} - {caption['interval'][1]:.1f}] {caption['speaker']}: {caption['text']}"

    lines: list[str] = []

    # Empty scanner if there is no bad segments.
    bad_segments_scanner = interval_scanner.IntervalScanner(bad_segments or [])

    scene_index = -1
    lines_skipped: bool = False
    for caption_idx, caption in enumerate(role_aware_summary):
        if manual_overrides.is_clip_ineligible(
            source_file, caption["interval"][0], caption["interval"][1]
        ):
            logging.info(
                f"Skipping due to manual override - {caption['speaker']}: {caption['text']}"
            )
            lines_skipped = True
            continue

        if bad_segments_scanner.overlapping_intervals(
            caption["interval"][0], caption["interval"][1]
        ):
            logging.info(
                f"Skipping due to bad segment - {caption['speaker']}: {caption['text']}"
            )
            lines_skipped = True
            continue

        if lines_skipped:
            lines.append("[INADMISSIBLE SEGMENT SKIPPED]")
            lines_skipped = False

        if video_config.ENABLE_VISION and scene_understanding is not None:
            time_end = caption["interval"][1]
            # For the last caption, iterate all remaining scenes.
            is_last_caption = caption_idx == len(role_aware_summary) - 1
            while scene_index + 1 < len(scene_understanding.chronology) and (
                scene_understanding.chronology[scene_index + 1].time < time_end
                or is_last_caption
            ):
                scene_index += 1
                this_scene = scene_understanding.chronology[scene_index]
                if not this_scene.actions:
                    continue

                if start is not None and this_scene.time < start:
                    continue
                if end is not None and this_scene.time > end:
                    break

                last_scene_time = (
                    scene_understanding.chronology[scene_index - 1].time
                    if scene_index > 0
                    else 0
                )

                lines.append(
                    f"[{last_scene_time:0.1f} - {this_scene.time:0.1f}] _[Visual: "
                    + " ".join(this_scene.actions)
                    + "]_"
                )

        if start is not None and caption["interval"][1] < start:
            continue
        if end is not None and caption["interval"][0] > end:
            break

        lines.append(_caption_to_text(caption))

    return lines
