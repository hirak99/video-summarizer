import unittest

from . import word_caption_utils

_TEST_CAPTIONS = [
    {
        "text": "Hello. How are you?",
        "words": [
            {"text": "Hello.", "interval": [0.0, 0.5], "speaker": "SPEAKER_00"},
            {"text": "How", "interval": [0.6, 0.8], "speaker": "SPEAKER_01"},
            {"text": "are", "interval": [0.9, 1.1], "speaker": "SPEAKER_01"},
            {"text": "you?", "interval": [1.2, 1.4], "speaker": "SPEAKER_01"},
        ],
    },
    {
        "text": "I am fine, thank you. Great. Nice to see you.",
        "words": [
            {"text": "I", "interval": [1.5, 1.6], "speaker": "SPEAKER_00"},
            {"text": "am", "interval": [1.7, 1.8], "speaker": "SPEAKER_00"},
            {"text": "fine,", "interval": [1.9, 2.1], "speaker": "SPEAKER_00"},
            {"text": "thank", "interval": [2.2, 2.4], "speaker": "SPEAKER_00"},
            {"text": "you.", "interval": [2.5, 2.7], "speaker": "SPEAKER_00"},
            {"text": "Great.", "interval": [2.8, 3.0], "speaker": "SPEAKER_01"},
            {"text": "Nice", "interval": [3.1, 3.3], "speaker": "SPEAKER_01"},
            {"text": "to", "interval": [3.4, 3.5], "speaker": "SPEAKER_01"},
            {"text": "see", "interval": [3.6, 3.8], "speaker": "SPEAKER_01"},
            {"text": "you.", "interval": [3.9, 4.1], "speaker": "SPEAKER_01"},
        ],
    },
    {
        "text": "Great.",
        "words": [
            {"text": "Cool.", "interval": [4.2, 4.3], "speaker": ""},
        ],
    },
]


_TEST_CAPTIONS2 = [
    {
        "text": "Hello.",
        "words": [
            {"text": "Hello.", "interval": [0.0, 0.5], "speaker": "SPEAKER_00"},
        ],
    },
    {
        "text": "How are you?",
        "words": [
            {"text": "How", "interval": [0.6, 0.8], "speaker": "SPEAKER_00"},
            {"text": "are", "interval": [0.9, 1.1], "speaker": "SPEAKER_00"},
            {"text": "you?", "interval": [1.2, 1.4], "speaker": "SPEAKER_00"},
        ],
    },
    {
        "text": "Fine.",
        "words": [
            {"text": "Fine.", "interval": [1.5, 1.6], "speaker": "SPEAKER_01"},
        ],
    },
]


class TestCaptionUtils(unittest.TestCase):
    def test_merge_word_captions(self):
        self.assertEqual(
            word_caption_utils.merge_word_captions(_TEST_CAPTIONS, None, None),
            [
                {"speaker": "SPEAKER_00", "text": "Hello.", "interval": (0.0, 0.5)},
                {
                    "speaker": "SPEAKER_01",
                    "text": "How are you?",
                    "interval": (0.6, 1.4),
                },
                {
                    "speaker": "SPEAKER_00",
                    "text": "I am fine, thank you.",
                    "interval": (1.5, 2.7),
                },
                {
                    "speaker": "SPEAKER_01",
                    "text": "Great. Nice to see you.",
                    "interval": (2.8, 4.1),
                },
                {"speaker": "", "text": "Cool.", "interval": (4.2, 4.3)},
            ],
        )

    def test_merge_word_captions_with_replacements(self):
        self.assertEqual(
            word_caption_utils.merge_word_captions(
                _TEST_CAPTIONS,
                {"SPEAKER_00": "Person A", "SPEAKER_01": "Person B"},
                "Unknown",
            ),
            [
                {"speaker": "Person A", "text": "Hello.", "interval": (0.0, 0.5)},
                {
                    "speaker": "Person B",
                    "text": "How are you?",
                    "interval": (0.6, 1.4),
                },
                {
                    "speaker": "Person A",
                    "text": "I am fine, thank you.",
                    "interval": (1.5, 2.7),
                },
                {
                    "speaker": "Person B",
                    "text": "Great. Nice to see you.",
                    "interval": (2.8, 4.1),
                },
                {"speaker": "Unknown", "text": "Cool.", "interval": (4.2, 4.3)},
            ],
        )

    def test_all_speakers(self):
        self.assertEqual(
            word_caption_utils.all_speakers(_TEST_CAPTIONS),
            ["SPEAKER_00", "SPEAKER_01"],
        )

    def test_speaker_continues(self):
        self.assertEqual(
            word_caption_utils.merge_word_captions(_TEST_CAPTIONS2, None, None),
            [
                {"speaker": "SPEAKER_00", "text": "Hello.", "interval": (0.0, 0.5)},
                {
                    "speaker": "SPEAKER_00",
                    "text": "How are you?",
                    "interval": (0.6, 1.4),
                },
                {"speaker": "SPEAKER_01", "text": "Fine.", "interval": (1.5, 1.6)},
            ],
        )
