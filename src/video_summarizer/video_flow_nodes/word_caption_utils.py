import dataclasses

from typing import Any


@dataclasses.dataclass
class _CaptionedText:
    words: list[str]
    interval: tuple[float, float]
    speaker: str


def merge_word_captions(
    word_captions: list[dict[str, Any]],
    speaker_aliases: dict[str, str] | None,
    unknown: str | None,
) -> list[dict[str, Any]]:
    """Merged individual speaker-captioned words from each segment of captions.

    If and as the speaker changes in words, new lines are added.
    """
    by_speaker: list[_CaptionedText] = []
    for segment in word_captions:

        # Note: For forcing a break after a segment.
        # Other than loosing resolution, very long segments produce very long lines which can throw off LLMs.
        new_segment = True

        for word in segment["words"]:

            speaker = word["speaker"]
            if speaker == "":
                if unknown is not None:
                    speaker = unknown
            elif speaker_aliases is not None:
                speaker = speaker_aliases[speaker]

            if new_segment:
                last_speaker = None
                new_segment = False
            else:
                last_speaker = by_speaker[-1].speaker if by_speaker else None
            if speaker != last_speaker:
                by_speaker.append(
                    _CaptionedText(
                        words=[],
                        interval=word["interval"],
                        speaker=speaker,
                    )
                )

            by_speaker[-1].words.append(word["text"])
            by_speaker[-1].interval = (by_speaker[-1].interval[0], word["interval"][1])

    # Convert to our standard dictionary format.
    return [
        {
            "speaker": item.speaker,
            "text": " ".join(item.words),
            "interval": item.interval,
        }
        for item in by_speaker
    ]


def all_speakers(captions: list[dict[str, Any]]) -> list[str]:
    """Get all unique speakers from the captions."""
    speakers = set()
    for caption in captions:
        for word in caption["words"]:
            speakers.add(word["speaker"])
    if "" in speakers:
        speakers.remove("")
    return sorted(speakers)
