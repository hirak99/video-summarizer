import unittest

from . import role_based_captioner
from . import vision_processor

# pyright: reportPrivateUsage=false

_EXAMPLE_CAPTIONS: list[role_based_captioner.RoleAwareCaptionT] = [
    {
        "speaker": "Teacher",
        "text": "Hello. How are you?",
        "interval": (30.0, 30.5),
    },
    {
        "speaker": "Student",
        "text": "What will we learn today?",
        "interval": (30.6, 30.8),
    },
    {
        "speaker": "Teacher",
        "text": "Do you have any questions?",
        "interval": (35.1, 35.8),
    },
    {
        "speaker": "Student",
        "text": "No, thank you.",
        "interval": (36.6, 36.8),
    },
]


class VisionProcessorTest(unittest.TestCase):
    def test_caption(self):
        last_ressults = vision_processor.SceneListT(model="test", chronology=[])
        self.assertEqual(
            vision_processor._get_prompt(
                source_movie="/path/to/MovieFile.mkv",
                captions=_EXAMPLE_CAPTIONS,
                last_results=last_ressults,
            ).splitlines(),
            [
                "This is a frame taken from a video session named '/path/to/MovieFile.mkv', showing the student's view.",
                "The teacher is remote, and the student may also be interacting with a mirror or collaborators.",
                "Summarize the frame's key content relevant to the session, ignoring PII (e.g., phone numbers) or UI elements (e.g., cursors, dialog boxes).",
                "Focus on the core visual content for analysis.",
                "For additional context, below is an excerpt of conversation immediately before this frame -",
                '30.0s | Teacher: "Hello. How are you?"',
                '30.6s | Student: "What will we learn today?"',
                '35.1s | Teacher: "Do you have any questions?"',
                '36.6s | Student: "No, thank you."',
                "",
                "Please output a JSON, of the form -",
                "{",
                '  "scene": ["BULLET_POINT_1", "BULLET_POINT_2", ...]',
                "}",
            ],
        )

    def test_caption_with_last(self):
        last_ressults = vision_processor.SceneListT(
            model="test",
            chronology=[
                vision_processor.SceneDescriptionT(
                    time=33.5,
                    context_hash="abc",
                    scene=[
                        "Medical equipment laid out on a table.",
                        "A person is talking to the camera.",
                    ],
                    actions=["Student picked up the doll."],
                )
            ],
        )
        prompt = vision_processor._get_prompt(
            source_movie="/path/to/MovieFile.mkv",
            captions=_EXAMPLE_CAPTIONS,
            last_results=last_ressults,
        )
        self.assertEqual(
            prompt.splitlines(),
            [
                "This is a frame taken from a video session named '/path/to/MovieFile.mkv', showing the student's view.",
                "The teacher is remote, and the student may also be interacting with a mirror or collaborators.",
                "Summarize the frame's key content relevant to the session, ignoring PII (e.g., phone numbers) or UI elements (e.g., cursors, dialog boxes).",
                "Focus on the core visual content for analysis.",
                "For additional context, below is an excerpt of conversation immediately before this frame -",
                '30.0s | Teacher: "Hello. How are you?"',
                '30.6s | Student: "What will we learn today?"',
                "33.5s | Student's View: Medical equipment laid out on a table. A person is talking to the camera. | Student's Actions: Student picked up the doll.",
                '35.1s | Teacher: "Do you have any questions?"',
                '36.6s | Student: "No, thank you."',
                "",
                "Here is the summary of the immediate last frame again:",
                "(Time: 33.5s)",
                "[",
                '  "Medical equipment laid out on a table.",',
                '  "A person is talking to the camera."',
                "]",
                "",
                "Summarize the current scene, and any relevant actions done since the previous frame by the student.",
                "If there is no important actions, then output an empty list.",
                "",
                "Please output a JSON, of the form -",
                "{",
                '  "scene": ["BULLET_POINT_1", "BULLET_POINT_2", ...]',
                '  "actions": ["SIGNIFICANT_ACTION_1", "SIGNIFICANT_ACTION_2", ...]',
                "}",
            ],
        )


if __name__ == "__main__":
    unittest.main()
