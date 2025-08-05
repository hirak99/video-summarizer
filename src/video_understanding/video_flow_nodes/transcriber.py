import json
import logging
import os
import subprocess

import whisper_timestamped as whisper  # type: ignore

from . import voice_separator
from .. import video_config
from ...flow import process_node

from typing import NotRequired, override, TypedDict

# Minimum number of seconds to skip if there is an error or a bad transition.
# Skipping a minimum ensures that we don't get stuck in a loop.
_ON_ERROR_SKIP = 0.5


# E.g. [{"interval": [0.0, 2.5], "text": "Hi there"}, ...]
class TranscriptionWordT(TypedDict):
    text: str
    start: float
    end: float
    confidence: float


class TranscriptionT(TypedDict):
    # If present, transcription was cut and restarted here to fix some issue.
    # The string indicates that issue.
    cut_reason: NotRequired[str]

    interval: tuple[float, float]
    text: str
    words: list[TranscriptionWordT]


def _text_is_rolling(text: str) -> bool:
    # A 'rolling' caption is when the transcriber fails to detect sentence breaks.
    words = [w.strip() for w in text.split()]
    # Checked that the word length is <= 51 for good captioning.
    # So > 60 triggering should be precise.
    return len(words) > 60 and not any(word and word[-1] in ",?." for word in words)


def _find_rolling_segment_index(transcription: list[TranscriptionT]) -> int | None:
    for index, segment in enumerate(transcription):
        text = segment["text"]
        if _text_is_rolling(text):
            return index
    return None


def _find_repetition_index(transcription: list[TranscriptionT]) -> int | None:
    # If text starts repeating, returns the index at which repetition starts.
    for i in range(len(transcription) - 2):
        if (
            transcription[i]["text"]
            == transcription[i + 1]["text"]
            == transcription[i + 2]["text"]
        ):
            logging.info(
                f"3-Repetition detected at index {i}, time {transcription[i]['interval'][0]} -"
            )
            logging.info(f"  {transcription[i]=}")
            logging.info(f"  {transcription[i+1]=}")
            logging.info(f"  {transcription[i+2]=}")
            return i
    return None


def _find_bad_index(
    transcription: list[TranscriptionT],
) -> tuple[int | None, str | None]:
    """Returns bad index and reason."""
    for tester_fn, reason in [
        (_find_repetition_index, "repetition"),
        (_find_rolling_segment_index, "rolling"),
    ]:
        index = tester_fn(transcription)
        if index is not None:
            return index, reason
    return None, None


# Cuts audio to a tempfile.
def _cut_audio(input_file: str, start: float) -> str:
    temp_file = video_config.tempdir() / "temp_cut_for_transcription.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-ss",
        str(start),
        str(temp_file),
    ]
    subprocess.run(command, check=True)
    return str(temp_file)


# TODO: Unittest this.
def _shift_transcription_timestamp(
    transcription: list[TranscriptionT], shift: float
) -> list[TranscriptionT]:
    return [
        {
            "interval": (
                round(segment["interval"][0] + shift, 2),
                round(segment["interval"][1] + shift, 2),
            ),
            "text": segment["text"],
            "words": [
                {
                    "text": word["text"],
                    "start": round(word["start"] + shift, 2),
                    "end": round(word["end"] + shift, 2),
                    "confidence": word["confidence"],
                }
                for word in segment["words"]
            ],
        }
        for segment in transcription
    ]


class WhisperTranscribe(process_node.ProcessNode):
    def __init__(self):
        # The large-v3-turbo appears to do on par with large-v3 for English.
        # See https://github.com/openai/whisper/discussions/2363
        # For the original models, see also pg. 22 here: https://arxiv.org/abs/2212.04356
        self._model = whisper.load_model("large-v3-turbo")
        self._model.to("cuda")

    def _transcribe_raw(self, local_path: str) -> list[TranscriptionT]:
        logging.info(f"Transcribing {local_path!r}...")
        # Note: The initial_prompt is in an attempt to make it output sentence
        # structure. See https://github.com/openai/whisper/discussions/194
        model_result = whisper.transcribe(
            self._model, local_path, language="en", initial_prompt="Okay."
        )
        transcription: list[TranscriptionT] = []
        for segment in model_result["segments"]:
            if "words" in segment:
                words: list[TranscriptionWordT] = segment["words"]
            else:
                # Sometimes, DTW may not work.
                # As a fallback use the full sentence.
                # These captions can be searched with confidence = 0.0.
                logging.warning(f'No "words" transcription in {segment=}')
                words: list[TranscriptionWordT] = [
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "confidence": 0.0,
                    }
                ]

            transcription.append(
                {
                    "interval": (segment["start"], segment["end"]),
                    "text": segment["text"],
                    "words": words,
                }
            )
        return transcription

    # Transcribe with safeguards.
    # Note: Though infrequently, Whisper can get stuck on some words. Using
    # `condition_on_previous_text=False` helps avoid this, but it will lose
    # sentence structure. Here is an alternative way to do this.
    # 1. Transcribe.
    # 2. Detect where it went bad. If not, return.
    # 3. Transcribe from where it went bad, and repeat.
    def _transcribe(self, local_path: str) -> list[TranscriptionT]:
        start_time: float = 0.0
        transcription_so_far: list[TranscriptionT] = []
        cut_reason: str | None = None
        while True:
            path = local_path
            temp_file_to_delete: str | None = None
            if start_time > 0:
                temp_file_to_delete = _cut_audio(local_path, start_time)
                path = temp_file_to_delete
            try:
                transcription = self._transcribe_raw(path)
            except AssertionError as e:
                error_message = str(e)
                if "Inconsistent number of segments" in error_message:
                    # Following error was seen when processing a file -
                    #   File "/path/to/python3.10/site-packages/whisper_timestamped/transcribe.py", line 310, in transcribe_timestamped
                    #     (transcription, words) = _transcribe_timestamped_efficient(model, audio,
                    #   File "/path/to/python3.10/site-packages/whisper_timestamped/transcribe.py", line 936, in _transcribe_timestamped_efficient
                    #     assert l1 == l2 or l1 == 0, f"Inconsistent number of segments: whisper_segments ({l1}) != timestamped_word_segments ({l2})"
                    # AssertionError: Inconsistent number of segments: whisper_segments (429) != timestamped_word_segments (428)
                    logging.warning(
                        f"Caught assertion error, retrying: {error_message}"
                    )
                    start_time += _ON_ERROR_SKIP
                    continue
                else:
                    raise
            if temp_file_to_delete is not None:
                os.remove(temp_file_to_delete)
            if start_time > 0:
                transcription = _shift_transcription_timestamp(
                    transcription, start_time
                )
                # Since timestamp is not 0, this was a cut.
                # Store the reson for the cut, i.e. what it fixes.
                if transcription:
                    assert cut_reason is not None
                    transcription[0]["cut_reason"] = cut_reason

            bad_index, cut_reason = _find_bad_index(transcription)

            if bad_index is None:
                transcription_so_far += transcription
                return transcription_so_far

            logging.info(f"Bad transcription detected. Reason: {cut_reason}")
            # Redo the transcription from where it is broken.
            transcription_so_far += transcription[:bad_index]
            # Avoid infinite loop by making sure start_time increases.
            start_time = max(
                transcription[bad_index]["interval"][0], start_time + _ON_ERROR_SKIP
            )
            logging.info(f"Restarting transcription at time {start_time}")

    @override
    def process(self, source_file: str, out_file_stem: str) -> str:
        out_file = out_file_stem + ".transcription.json"
        with voice_separator.get_wav(source_file) as source_file:
            transcription = self._transcribe(source_file)
        with open(out_file, "w") as f:
            json.dump(transcription, f)
        return out_file
