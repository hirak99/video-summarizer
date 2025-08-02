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
                "This is a frame taken from a video session named '/path/to/MovieFile.mkv'.",
                "The perspective is of the student's view. Teacher is remote, and the student may look at mirror or other collaborators.",
                "Please summarize the frame for future analysis.",
                "Ignore any PII such as phone number, user-interface elements like cursor or dialog box - and focus on content relevant for the session.",
                "",
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
                "This is a frame taken from a video session named '/path/to/MovieFile.mkv'.",
                "The perspective is of the student's view. Teacher is remote, and the student may look at mirror or other collaborators.",
                "Please summarize the frame for future analysis.",
                "Ignore any PII such as phone number, user-interface elements like cursor or dialog box - and focus on content relevant for the session.",
                "",
                "For additional context, below is an excerpt of conversation immediately before this frame -",
                '30.0s | Teacher: "Hello. How are you?"',
                '30.6s | Student: "What will we learn today?"',
                "33.5s | Scene: Medical equipment laid out on a table. A person is talking to the camera. | Actions: Student picked up the doll.",
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
                "Summarize the current scene, and actions done since the previous frame by the student.",
                "If there is no important action then output an empty list.",
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
