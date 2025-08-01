import contextlib
import json
import logging
import os
import subprocess

from pyannote import audio  # type: ignore
import torch

from .. import video_config
from ...flow import process_node

from typing import override

_HF_AUTH_ENV = "HUGGING_FACE_AUTH"

# Since we know there are two speakers, we can use this info to guide diarization.
_NUM_SPEAKERS = 2


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
    def process(self, source_file: str, out_file_stem: str) -> str:
        out_file = out_file_stem + ".diarized.json"
        logging.info(f"Diatrizing to {out_file}")
        with get_wav(source_file) as source_wav:
            diarization = self._pipeline(source_wav, max_speakers=_NUM_SPEAKERS)
        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append(
                {
                    "interval": (turn.start, turn.end),
                    "speaker": speaker,
                }
            )
        with open(out_file, "w") as f:
            json.dump(result, f)
        return out_file
