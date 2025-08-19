# Sometimes. Whisper appears to create incorrect starting point, spanning 10s of seconds.
# This will attempt to correct those silences.
import json
import logging

from . import transcriber
from . import voice_separator
from ...flow import process_node
from ..utils import interval_scanner

from typing import override

# Only silences bigger than this will be corrected.
# I think at about 4 it starts to become uncomfortably long silence.
_SILENCE_THRESHOLD = 4.0

# Amount of silence retained after correction. Should be <= _ALLOWED_SILENCE.
_MAINTAIN_SILENCE = 1.0

# Will not trim if any word less than this.
_MIN_WORD_LENGTH = 0.5


def _union_intervals(intervals: list[tuple[float, float]]):
    intervals.sort()
    result: list[tuple[float, float]] = []
    for start, end in intervals:
        if not result:
            result.append((start, end))
            continue
        last_start, last_end = result[-1]
        if start <= last_end:
            result[-1] = (last_start, max(last_end, end))
        else:
            result.append((start, end))
    return result


# TODO: Add unit tests for the following two methods.
def _trim_start(caption: transcriber.TranscriptionT, new_start: float):
    new_start = round(new_start, 2)
    # The new start cannot be after the end of the first word.
    first_word_end = caption["words"][0]["end"]
    new_start = min(new_start, first_word_end - _MIN_WORD_LENGTH)
    caption["interval"] = (new_start, caption["interval"][1])
    caption["words"][0]["start"] = new_start


def _trim_end(caption: transcriber.TranscriptionT, new_end: float):
    new_end = round(new_end, 2)
    # The new end cannot be before the start of the last word.
    last_word_start = caption["words"][-1]["start"]
    new_end = max(new_end, last_word_start + _MIN_WORD_LENGTH)
    caption["interval"] = (caption["interval"][0], new_end)
    caption["words"][-1]["end"] = new_end


class TranscriptionRefiner(process_node.ProcessNode):
    @override
    def process(
        self,
        captions_file: str,
        diarization_file: str,
        out_file_stem: str,
    ) -> str:
        """Stores assigned captions as a json file and returns the file name."""
        logging.info(f"Correcting transcriptions for {captions_file!r}...")
        with open(captions_file) as f:
            captions: list[transcriber.TranscriptionT] = json.load(f)
        with open(diarization_file) as f:
            diarizations = voice_separator.DiarizationListT.model_validate_json(
                f.read()
            )

        speech_times: list[tuple[float, float]] = [
            x.interval for x in diarizations.root
        ]
        speech_times = _union_intervals(speech_times)

        speech_scanner = interval_scanner.IntervalScanner(
            [{"interval": x} for x in speech_times]
        )

        for caption in captions:
            start, end = caption["interval"]
            speech_intervals = speech_scanner.overlapping_intervals(start, end)
            if not speech_intervals:
                # Too many of these to make it a logging.warning().
                # logging.info(f"No speech time intersects with {caption=}")
                continue
            # If caption starts or ends with a long silence, trim it.
            # Note: This doesn't take into account if there is long silence after a first short speech.
            # However, I have not seen that kind of error out of Whisper yet.
            speech_start = speech_intervals[0]["interval"][0]
            speech_end = speech_intervals[-1]["interval"][1]
            if start < speech_start - _SILENCE_THRESHOLD:
                _trim_start(caption, speech_start - _MAINTAIN_SILENCE)
                logging.info(
                    f"Trimming start, because caption {start=}, {speech_start=}; corrected {caption=}"
                )
            if end > speech_end + _SILENCE_THRESHOLD:
                _trim_end(caption, speech_end + _MAINTAIN_SILENCE)
                logging.info(
                    f"Trimming end, because caption {end=}, {speech_end=}; corrected {caption=}"
                )

        out_file = out_file_stem + ".captions_corrected.json"
        with open(out_file, "w") as f:
            json.dump(captions, f)

        return out_file
