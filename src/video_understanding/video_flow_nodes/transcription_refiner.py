# Sometimes. Whisper appears to create incorrect starting point, spanning 10s of seconds.
# This will attempt to correct those silences.
import json
import logging

from . import transcriber
from . import voice_separator
from ...flow import process_node
from ..utils import interval_scanner
from ..utils import misc_utils

from typing import Iterable, Iterator, override

# Only silences bigger than this will be corrected.
# I think at about 4 it starts to become uncomfortably long silence.
_SILENCE_THRESHOLD = 4.0

# Amount of silence retained after correction. Should be <= _ALLOWED_SILENCE.
_MAINTAIN_SILENCE = 1.0

_MIN_SENTENCE_LENGTH = 0.5

# Controls for splitting long captions.
# Splitting will be considered after these many words. The word must end with sentence-ending char to split.
_SPLIT_AFTER_WORDS = 15
# Will not split if remaining words are <= this count.
_NO_SPLIT_IF_REMAINING = 6


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


# TODO: Add unit test.
def _split_long_captions(
    captions: Iterable[transcriber.TranscriptionT],
) -> Iterator[transcriber.TranscriptionT]:
    for caption in captions:
        words = caption["words"]
        if len(words) <= _SPLIT_AFTER_WORDS:
            yield caption
            continue

        # Reconstruct using the words.
        def reconstruct(start_index: int, end_index: int) -> transcriber.TranscriptionT:
            if start_index == 0 and end_index == len(words):
                # Avoid unnecessary processing.
                logging.info(f"Returning full caption at {caption['interval']=}")
                return caption

            full_text = "".join(
                # The " " prefix is how Whisper does it, because there is always a space before sentence starts.
                " " + x["text"]
                for x in words[start_index:end_index]
            )
            result: transcriber.TranscriptionT = {
                "interval": (words[start_index]["start"], words[end_index - 1]["end"]),
                "text": full_text,
                "words": words[start_index:end_index],
            }
            if start_index == 0:
                if "cut_reason" in caption:
                    result["cut_reason"] = caption["cut_reason"]
            else:
                result["cut_reason"] = "split long caption"
            logging.info(
                f"Splitting caption from {start_index} to {end_index} ({len(words)=}): {result['text']}"
            )
            return result

        start_i = 0
        for end_i in range(len(words) - _NO_SPLIT_IF_REMAINING):
            word = words[end_i]["text"]
            if end_i + 1 - start_i >= _SPLIT_AFTER_WORDS:
                if word[-1] in "?.":
                    yield reconstruct(start_i, end_i + 1)
                    start_i = end_i + 1
        if start_i < len(words) - 1:
            yield reconstruct(start_i, len(words))


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

        corrected_captions: list[transcriber.TranscriptionT] = []
        for caption in _split_long_captions(captions):
            start, end = caption["interval"]
            logging.info(f"{start=} {end=} {caption['text']=}")
            speech_intervals = speech_scanner.overlapping_intervals(start, end)
            if not speech_intervals:
                # Too many of these to make it a logging.warning().
                # logging.info(f"No speech time intersects with {caption=}")
                continue
            # If caption starts or ends with a long silence, trim it.
            # Note: This doesn't take into account if there is long silence after a first short speech.
            # However, I have not seen that kind of error out of Whisper yet.

            # Note: Intervals are not sorted! So following won't work -
            # speech_start = speech_intervals[0]["interval"][0]
            # speech_end = speech_intervals[-1]["interval"][1]
            speech_start = min(x["interval"][0] for x in speech_intervals)
            speech_end = max(x["interval"][1] for x in speech_intervals)

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
            corrected_captions.append(caption)

        del captions  # Prevent inadvertent usage after this point.

        out_file = (
            f"{out_file_stem}.captions_corrected.{misc_utils.timestamp_str()}.json"
        )
        with open(out_file, "w") as f:
            json.dump(corrected_captions, f)

        return out_file
