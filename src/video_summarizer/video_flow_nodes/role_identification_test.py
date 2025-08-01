import unittest

from . import role_identifier

# pyright: reportPrivateUsage=false

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


class TestCaptionUtils(unittest.TestCase):
    def test_combine_word_captions(self):
        lines, aliases = role_identifier._caption_to_str(_TEST_CAPTIONS)
        self.assertEqual(
            lines.splitlines(),
            [
                "Person A: Hello.",
                "Person B: How are you?",
                "Person A: I am fine, thank you.",
                "Person B: Great. Nice to see you.",
                "Either: Cool.",
            ],
        )
        self.assertEqual(aliases, {"SPEAKER_00": "Person A", "SPEAKER_01": "Person B"})
