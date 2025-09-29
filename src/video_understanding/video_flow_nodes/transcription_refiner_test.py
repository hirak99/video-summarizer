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
    def _check_words(
        self,
        caption: transcriber.TranscriptionT,
        expected: list[tuple[str, tuple[float, float]]],
    ) -> None:
        """
        Helper to check all words' text and (start, end).
        expected: list of (text, (start, end))
        """
        words = caption["words"]
        self.assertEqual(len(words), len(expected))
        for word, (exp_text, (exp_start, exp_end)) in zip(words, expected):
            self.assertEqual(word["text"], exp_text)
            self.assertEqual(word["start"], exp_start)
            self.assertEqual(word["end"], exp_end)

    def test_trim_start_normal(self):
        caption = copy.deepcopy(_TEST_CAPTION)
        transcription_refiner._trim_start(caption, 5.1)
        self.assertEqual(caption["interval"][0], 5.1)
        expected = [
            ("Hello", (5.1, 6.0)),
            ("how", (6.0, 7.0)),
            ("are", (7.0, 8.0)),
            ("you?", (8.0, 9.0)),
        ]
        self._check_words(caption, expected)

    def test_trim_start_past_first_word_end(self):
        caption = copy.deepcopy(_TEST_CAPTION)
        transcription_refiner._trim_start(caption, 8.1)
        self.assertEqual(caption["interval"][0], 6.0 - _MIN_WORD_LENGTH)
        expected = [
            ("Hello", (6.0 - _MIN_WORD_LENGTH, 6.0)),
            ("how", (6.0, 7.0)),
            ("are", (7.0, 8.0)),
            ("you?", (8.0, 9.0)),
        ]
        self._check_words(caption, expected)

    def test_trim_end_normal(self):
        caption = copy.deepcopy(_TEST_CAPTION)
        # Try to trim to 8.9, which is within the last word ("you?": start=8.0, end=9.0)
        transcription_refiner._trim_end(caption, 8.9)
        self.assertEqual(caption["interval"][1], 8.9)
        expected = [
            ("Hello", (5.0, 6.0)),
            ("how", (6.0, 7.0)),
            ("are", (7.0, 8.0)),
            ("you?", (8.0, 8.9)),
        ]
        self._check_words(caption, expected)

    def test_trim_end_before_last_word_start(self):
        caption = copy.deepcopy(_TEST_CAPTION)
        # Try to trim to 7.1, which is before last word's start (8.0), so should clamp to 8.5 (8.0 + _MIN_WORD_LENGTH)
        transcription_refiner._trim_end(caption, 7.1)
        self.assertEqual(caption["interval"][1], 8.0 + _MIN_WORD_LENGTH)
        expected = [
            ("Hello", (5.0, 6.0)),
            ("how", (6.0, 7.0)),
            ("are", (7.0, 8.0)),
            ("you?", (8.0, 8.0 + _MIN_WORD_LENGTH)),
        ]
        self._check_words(caption, expected)
