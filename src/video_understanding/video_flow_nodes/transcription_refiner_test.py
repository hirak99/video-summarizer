import copy
import unittest

from . import transcriber
from . import transcription_refiner
from typing import Final

# Access private methods for tests.
# pyright: reportPrivateUsage=false

_TEST_CAPTION: transcriber.TranscriptionT = {
    "text": "Hello, how are you?",
    "interval": (5.0, 10.0),
    "words": [
        {"text": "Hello", "start": 5.0, "end": 6.0, "confidence": 0.9},
        {"text": "how", "start": 6.0, "end": 7.0, "confidence": 0.9},
        {"text": "are", "start": 7.0, "end": 8.0, "confidence": 0.9},
        {"text": "you?", "start": 8.0, "end": 9.0, "confidence": 0.9},
    ],
}

# Convenient shorthand.
_MIN_WORD_LENGTH: Final[float] = transcription_refiner._MIN_WORD_LENGTH


class TestTranscriptionRefiner(unittest.TestCase):
    def test_trim_start_normal(self):
        caption = copy.deepcopy(_TEST_CAPTION)
        transcription_refiner._trim_start(caption, 5.1)
        self.assertEqual(caption["interval"][0], 5.1)
        self.assertEqual(caption["words"][0]["start"], 5.1)

    def test_trim_start_past_first_word_end(self):
        caption = copy.deepcopy(_TEST_CAPTION)
        transcription_refiner._trim_start(caption, 8.1)
        self.assertEqual(caption["interval"][0], 6.0 - _MIN_WORD_LENGTH)
        self.assertEqual(caption["words"][0]["start"], 6.0 - _MIN_WORD_LENGTH)

    def test_trim_end_normal(self):
        caption = copy.deepcopy(_TEST_CAPTION)
        # Try to trim to 8.5, which is within the last word ("you?": start=8.0, end=9.0)
        transcription_refiner._trim_end(caption, 8.9)
        self.assertEqual(caption["interval"][1], 8.9)
        self.assertEqual(caption["words"][-1]["end"], 8.9)

    def test_trim_end_before_last_word_start(self):
        caption = copy.deepcopy(_TEST_CAPTION)
        # Try to trim to 8.1, which is before last word's start (8.0), so should clamp to 8.5 (8.0 + _MIN_WORD_LENGTH)
        transcription_refiner._trim_end(caption, 7.1)
        self.assertEqual(caption["interval"][1], 8.0 + _MIN_WORD_LENGTH)
        self.assertEqual(caption["words"][-1]["end"], 8.0 + _MIN_WORD_LENGTH)
