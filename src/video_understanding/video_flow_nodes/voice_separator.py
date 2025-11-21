import contextlib
import json
import logging
import os
import subprocess

from pyannote import audio  # type: ignore
import pydantic
import torch

from .. import video_config
from ...flow import process_node
from ..utils import file_conventions

from typing import override

_HF_AUTH_ENV = "HUGGING_FACE_AUTH"


# This is type of saved data.
# TODO: Add validation before saving.
# TODO: Use in speaker_assigner.py.
class _Diarization(pydantic.BaseModel):
    interval: tuple[float, float]
    speaker: str


DiarizationListT = pydantic.RootModel[list[_Diarization]]


@contextlib.contextmanager
def get_wav(source_file: str):
    out_file = None
    try:
        out_file = video_config.random_temp_fname("_wav_temp", ".wav")
        command = [
            "ffmpeg",
            "-y",
            "-i",
            source_file,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",  # Single channel
            "-ar",
            "16000",  # Sample rate 16kHz
            out_file,
        ]
        subprocess.run(command, check=True)
        yield out_file
    finally:
        if out_file is not None:
            os.remove(out_file)
            logging.info(f"Removed {out_file!r}")


class VoiceSeparator(process_node.ProcessNode):
    def __init__(self) -> None:
        self._pipeline = audio.Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv(_HF_AUTH_ENV),
        )
        self._pipeline.to(torch.device("cuda"))

    @override
    def process(
        self,
        source_file: str,
        out_file_stem: str,
    ) -> str:
        out_file = out_file_stem + ".diarized.json"
        logging.info(f"Diatrizing to {out_file}")

        max_speakers = 2
        file_parts = file_conventions.FileNameComponents.from_pathname(source_file)
        if file_parts.student.startswith("G"):
            # Group of students.
            max_speakers = 3

        with get_wav(source_file) as source_wav:
            diarization = self._pipeline(source_wav, max_speakers=max_speakers)

        result = []
        seen_speakers: set[str] = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            seen_speakers.add(speaker)
            result.append(
                {
                    "interval": (turn.start, turn.end),
                    "speaker": speaker,
                }
            )

        logging.info(f"Seen speakers: {seen_speakers} with {max_speakers=}.")

        # Validate.
        all(_Diarization.model_validate(x) for x in result)

        with open(out_file, "w") as f:
            json.dump(result, f)
        return out_file
