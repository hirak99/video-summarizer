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

_MIN_SENTENCE_LENGTH = 0.5


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


# Combined function to trim start or end.
def _trim_caption(
    caption: transcriber.TranscriptionT,
    *,
    new_start: float = float("-inf"),
    new_end: float = float("inf"),
) -> None:
    new_start = round(new_start, 2)
    new_start = min(new_start, caption["interval"][1] - _MIN_SENTENCE_LENGTH)
    new_start = max(new_start, caption["interval"][0])

    new_end = round(new_end, 2)
    # new_end = max(new_end, caption["interval"][0] + _MIN_SENTENCE_LENGTH)
    new_end = max(new_end, new_start + _MIN_SENTENCE_LENGTH)
    new_end = min(new_end, caption["interval"][1])

    caption["interval"] = (new_start, new_end)

    # Combine words whose intervals no longer overlap with (new_start, new_end).
    new_words: list[transcriber.TranscriptionWordT] = []
    for word in caption["words"]:
        word["start"] = max(word["start"], new_start)
        word["end"] = min(word["end"], new_end)
        if not new_words:
            new_words.append(word)
            continue
        last_word = new_words[-1]
        if last_word["end"] <= new_start or word["start"] >= new_end:
            last_word["text"] += " " + word["text"]
            last_word["end"] = word["end"]
            continue
        new_words.append(word)
    caption["words"] = new_words


def _trim_start(caption: transcriber.TranscriptionT, new_start: float) -> None:
    _trim_caption(caption, new_start=new_start)


def _trim_end(caption: transcriber.TranscriptionT, new_end: float) -> None:
    _trim_caption(caption, new_end=new_end)


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
