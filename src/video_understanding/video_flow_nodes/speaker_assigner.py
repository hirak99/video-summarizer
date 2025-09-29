import json
import logging

from . import transcriber
from ...flow import process_node
from ..utils import interval_scanner

import typing
from typing import override, TypedDict

_BREAK_SENTENCE_TIME = 0.5  # Seconds of silence to break sentences.


class _SpeakerWordT(TypedDict):
    interval: tuple[float, float]
    speaker: str
    text: str
    confidence: float


# This is the output type of the data that we will write.
class SpeakerCaptionT(TypedDict):
    interval: tuple[float, float]
    text: str
    words: list[_SpeakerWordT]


def _split_to_sentences(
    words: list[transcriber.TranscriptionWordT],
) -> list[list[transcriber.TranscriptionWordT]]:
    sentences: list[list[transcriber.TranscriptionWordT]] = []
    current: list[transcriber.TranscriptionWordT] = []  # Current sentence.
    last_word: transcriber.TranscriptionWordT | None = None
    for word_dict in words:
        word = word_dict["text"]
        current.append(word_dict)
        break_here = False

        if last_word is not None:
            if last_word["end"] + _BREAK_SENTENCE_TIME < word_dict["start"]:
                break_here = True

        if word.endswith(".") or word.endswith("?") or word.endswith("!"):
            break_here = True

        if break_here:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


class SpeakerAssigner(process_node.ProcessNode):
    @override
    def process(
        self,
        captions_file: str,
        diarization_file: str,
        out_file_stem: str,
    ) -> str:
        """Stores assigned captions as a json file and returns the file name."""
        out_file = out_file_stem + ".assigned_captions.json"
        with open(captions_file) as f:
            captions: list[transcriber.TranscriptionT] = json.load(f)
        with open(diarization_file) as f:
            diarizations = json.load(f)

        diarization_scanner = interval_scanner.IntervalScanner(diarizations)

        for caption in captions:
            # Caption has the format -
            # {
            #   "interval": [433.64, 434.66],
            #   "text": " I like that.",
            #   "words": [
            #     {
            #       "text": "I",
            #       "start": 433.64,
            #       "end": 433.9,
            #       "confidence": 0.414
            #     },
            #     {
            #       "text": "like",
            #       "start": 433.9,
            #       "end": 434.2,
            #       "confidence": 0.235
            #     },
            #     {
            #       "text": "that.",
            #       "start": 434.2,
            #       "end": 434.66,
            #       "confidence": 0.4
            #     }
            #   ]
            # }

            # We change to this -
            #   "words": [
            #     {
            #       "interval": [ 433.64, 433.9 ],
            #       "speaker": "SPEAKER_01",
            #       "text": "I",
            #       "confidence": 0.414
            #     },
            #     {
            #       "interval": [ 433.9, 434.2 ],
            #       "speaker": "SPEAKER_01",
            #       ...
            #     },
            #     {
            #       "interval": [ 434.2, 434.66 ],
            #       "speaker": "SPEAKER_01",
            #       ...
            #     }
            #   ]

            # TODO: This is moderately complicated. Should add unittest.
            sentences = _split_to_sentences(caption["words"])
            for sentence in sentences:
                # Duration of the whole sentence.
                start, end = sentence[0]["start"], sentence[-1]["end"]

                speaker_weights: dict[str, float] = {}
                for segment in diarization_scanner.overlapping_intervals(start, end):
                    # Consider the lengt of intersection between [start, end] and the diarized interval.
                    dia_length = min(end, segment["interval"][1]) - max(
                        start, segment["interval"][0]
                    )
                    speaker_weights[segment["speaker"]] = (
                        speaker_weights.get(segment["speaker"], 0) + dia_length
                    )
                if speaker_weights:
                    # Assign the speaker with most amount of time spoken.
                    sentence_speaker = max(
                        speaker_weights, key=lambda x: speaker_weights[x]
                    )
                else:
                    sentence_speaker = ""

                for word in sentence:
                    start, end = word["start"], word["end"]
                    # Note: We are changing word here, and it will no longer be captioning.TranscriptionWordT.
                    word["interval"] = (start, end)  # type: ignore
                    del word["start"]  # type: ignore
                    del word["end"]  # type: ignore
                    word["speaker"] = sentence_speaker  # type: ignore

        # This line actually does nothing, except to tell the reader the type of
        # the data we will write.
        out_captions = typing.cast(list[SpeakerCaptionT], captions)

        with open(out_file, "w") as f:
            json.dump(out_captions, f)
        logging.info(f"Written to {out_file}")
        return out_file
